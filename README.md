# Customer Lifetime Value — End-to-End ML Project

Proiect pentru cursul **eVantage Data Science V1.0, Lecția 5** (Classic ML
End-to-End Project). Predicție de Customer Lifetime Value (CLV) pe 90 de
zile, folosind XGBoost cu hyperparameter tuning + un layer de clasificare
pentru risc de churn — expuse printr-un API, containerizate, deployate pe
Kubernetes (k3s), cu CI/CD pe GitHub Actions și monitoring Prometheus/Grafana.

## Ce face proiectul

Pentru un client de retail online (dataset [Online
Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail), UCI), modelul
prezice:
- **CLV pe 90 de zile** (regresie, XGBoost cu hyperparameter tuning via
  `RandomizedSearchCV`)
- **Probabilitatea ca respectivul client să rămână activ** în aceeași
  fereastră (clasificare — Logistic Regression / Decision Tree / Random
  Forest, ales automat cel cu ROC-AUC mai bun)
- Combinate într-un **Expected Value** = `CLV_prezis × P(activ)`, folosit ca
  scor de business pentru prioritizare de retenție

Rezultate pe test set (ultima rulare cunoscută): MAE ≈ 52.7 GBP (vs. baseline
naiv ≈ 88.7 GBP → ~40% reducere a erorii), R² ≈ 0.70, churn ROC-AUC ≈ 0.745.

## Arhitectură

```
                          ┌─────────────────┐
                          │  Online Retail   │
                          │   (CSV, UCI)     │
                          └────────┬─────────┘
                                   │
                    src/data_cleaning.py + features.py
                                   │
                          ┌────────▼─────────┐
                          │   src/train.py    │  (tuning XGBoost +
                          │                    │   comparație clasificatori)
                          └────────┬───────────┘
                                   │ salvează
                          ┌────────▼─────────┐
                          │     models/       │
                          └────────┬───────────┘
                                   │
                          ┌────────▼─────────┐
                          │ src/inference.py  │
                          │  + main.py (API)  │──── /predict, /health, /metrics
                          └────────┬───────────┘
                                   │
                    Docker (GitHub Actions CI) → GHCR
                                   │
                          ┌────────▼─────────┐
                          │  Helm → k3s        │
                          │  (helm/clv-api)    │
                          └────────┬───────────┘
                                   │
                    Prometheus (scrape /metrics) → Grafana
                                   │
                    CronJob (periodic) → evaluate_and_push_mae.py
                                   │
                           Pushgateway → Prometheus (MAE/RMSE/R2/AUC)
```

## Structură repo

```
.
├── .github/workflows/ci.yml     # test -> train -> build & push imagine (GHCR)
├── data/                        # online_retail.csv (NU e în git, vezi mai jos)
├── models/                      # artefacte antrenate (NU e în git, generat de train.py)
├── src/
│   ├── data_cleaning.py         # load_and_clean()
│   ├── features.py              # build_features(), FEATURE_COLS
│   ├── models.py                 # tuning XGBoost + comparație clasificatori churn
│   ├── train.py                  # entry point — rulează tot pipeline-ul
│   ├── inference.py              # predict_customer() — folosit și de API
│   └── evaluate_and_push_mae.py  # job periodic — trimite metrici către Pushgateway
├── tests/                        # pytest — rulat automat în CI
├── helm/clv-api/                 # Helm chart pentru deploy pe k3s
├── monitoring/                   # manifests Prometheus + Grafana + Pushgateway + CronJob
├── main.py                       # FastAPI wrapper (+ /metrics via Instrumentator)
├── Dockerfile
└── requirements.txt
```

## Setup — de la zero (clonat de pe GitHub)

### 1. Clonează și instalează

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Ia dataset-ul

CSV-ul nu e în git (45MB, CI-ul îl descarcă fresh la fiecare build). Local:
```bash
mkdir -p data
curl -L -o data/online_retail.csv \
  "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
```

### 3. Antrenează modelul

```bash
python -m src.train
```
Durează sub un minut, produce `models/xgb_clv.json`, `models/churn_model.pkl`,
`models/metadata.json`.

### 4. Pornește API-ul local

```bash
uvicorn main:app --reload --port 8000
```
- Swagger UI interactiv: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Metrici Prometheus: `http://localhost:8000/metrics`

Test rapid:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_cal": 5, "recency_cal": 120, "T_cal": 200,
    "monetary_value_cal": 45.5, "avg_basket_value": 45.5,
    "n_unique_products": 12, "avg_quantity": 8.3, "tenure_days": 200
  }'
```

### 5. Rulează testele

```bash
pytest tests/ -v
```

## CI/CD — GitHub Actions

`.github/workflows/ci.yml`, la fiecare push pe `main`:
1. **test** — `pytest tests/` (orice push/PR, rapid, nu antrenează)
2. **build-and-push** (doar pe `main`, după ce testele trec) — descarcă
   dataset-ul, rulează `python -m src.train` (imaginea conține mereu un
   model proaspăt, nu unul static comis în git), construiește imaginea
   Docker, o urcă în **GitHub Container Registry**
   (`ghcr.io/<user>/<repo>:latest` + `:<commit-sha>`)

Nu trebuie configurat niciun secret manual — folosește `GITHUB_TOKEN`-ul
implicit al Actions.

**Notă importantă:** pachetele noi din GHCR sunt **private by default**.
Fă pachetul public din Settings (GitHub → contul tău → Packages →
pachetul → Package settings → Change visibility), altfel `kubectl` nu va
putea trage imaginea fără un `imagePullSecret`.

## Deploy pe Kubernetes (k3s + Helm)

Presupune un cluster k3s funcțional local (`kubectl get nodes` trebuie să
arate un nod `Ready`) și Helm instalat.

```bash
helm install clv helm/clv-api
kubectl get pods
kubectl get svc
```

Testare locală (port-forward):
```bash
kubectl port-forward svc/clv-clv-api 8000:80
curl http://localhost:8000/health
```

Upgrade după o imagine nouă în GHCR:
```bash
helm upgrade clv helm/clv-api
# sau, dacă tag-ul rămâne "latest" și vrei rollout forțat:
kubectl rollout restart deployment clv-clv-api
```

Rollback:
```bash
helm rollback clv
```

Detalii complete (imagePullSecret, valori configurabile) în
`helm/clv-api/README.md`.

## Monitoring — Prometheus + Grafana + Pushgateway

Manifests simple (nu Prometheus Operator), în `monitoring/`:

```bash
kubectl apply -f monitoring/prometheus-rbac.yaml
kubectl apply -f monitoring/prometheus-configmap.yaml
kubectl apply -f monitoring/prometheus-deployment.yaml
kubectl apply -f monitoring/grafana-datasource-configmap.yaml
kubectl apply -f monitoring/grafana-deployment.yaml
kubectl apply -f monitoring/pushgateway-deployment.yaml
kubectl apply -f monitoring/mae-evaluation-cronjob.yaml
```

- **Prometheus** descoperă automat pod-uri cu adnotarea
  `prometheus.io/scrape: "true"` (deja setată pe pod-ul `clv-api` din chart)
  — vezi `http://localhost:9090/targets` (după `kubectl port-forward
  svc/prometheus 9090:9090`)
- **Grafana** — `kubectl port-forward svc/grafana 3000:3000`, user/parolă
  `admin`/`admin` (doar pentru local — schimbă dacă expui vreodată în
  afara mașinii tale). Datasource Prometheus e deja provizionat automat.
- **Metrici disponibile din API** (request count + latency, gratuit din
  instrumentare): `http_requests_total`, `http_request_duration_seconds_bucket`
  — vezi exemple de query PromQL mai jos.
- **Pushgateway + CronJob** — `evaluate_and_push_mae.py` rulează periodic
  (`*/10 * * * *` — interval de test; în producție reală ar avea sens mult
  mai rar, dat fiind orizontul de 90 zile al modelului) și trimite
  `clv_model_mae`, `clv_model_rmse`, `clv_model_r2`, `clv_churn_model_roc_auc`.

Query-uri PromQL utile în Grafana:
```
rate(http_requests_total{handler="/predict"}[1m])
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict"}[5m]))
clv_model_mae
```

### ⚠️ Status cunoscut / de reparat

Gauge-urile de business (`clv_model_mae`, `clv_model_rmse`, `clv_model_r2`,
`clv_churn_model_roc_auc`) **nu apar încă în Grafana la momentul acestui
commit** — mecanismul de push a fost testat și confirmat funcțional local
(Pushgateway local + script rulat manual), dar în cluster CronJob-ul nu a
fost încă declanșat/verificat capăt la capăt. De reparat/verificat:
- confirmă că CronJob-ul chiar rulează (`kubectl get jobs`, `kubectl get
  pods | grep clv-mae`)
- verifică log-urile job-ului (`kubectl logs <pod>`) pentru erori de
  conectare la `pushgateway:9091`
- dacă tot nu apare nimic, declanșează manual pentru debugging:
  ```bash
  kubectl create job clv-mae-manual-test --from=cronjob/clv-mae-evaluation
  kubectl wait --for=condition=complete job/clv-mae-manual-test --timeout=120s
  kubectl logs job/clv-mae-manual-test
  ```

**Notă despre MAE ca metrică:** spre deosebire de request count/latency
(care există live, din trafic real), MAE necesită ground truth — pe acest
dataset istoric (2010-2011), scriptul folosește MAE-ul deja calculat pe
test set la antrenare (din `models/metadata.json`), nu recalculează
"live" pe date noi (asta ar necesita trafic real + o fereastră de 90 zile
de așteptare, cum ar fi în producție reală).

## Pași următori

- Debug complet pentru gauge-urile de mai sus
- Alertare Grafana pe threshold (ex. MAE peste un anumit prag)
- Loki + Promtail pentru logging structurat al predicțiilor individuale