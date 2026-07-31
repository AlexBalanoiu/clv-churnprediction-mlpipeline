# Customer Lifetime Value — API

## Structură
```
.
├── .github/workflows/ci.yml # GitHub Actions: test -> train -> build & push image
├── data/
│   └── online_retail.csv    # nu e în git — CI îl descarcă la build
├── models/                  # generat de train.py (nu e în git)
├── src/
│   ├── data_cleaning.py     # load_and_clean()
│   ├── features.py          # build_features(), FEATURE_COLS
│   ├── models.py            # tuning XGBoost + comparație clasificatori churn
│   ├── train.py             # entry point — rulează tot pipeline-ul
│   └── inference.py         # predict_customer() — folosit și de API
├── tests/                   # pytest — rulat automat în CI
├── main.py                  # FastAPI wrapper
├── Dockerfile
└── requirements.txt
```

## Setup în VS Code

1. **Deschide folderul în VS Code** (`File > Open Folder`)

2. **Instalează extensia Python** (Microsoft) dacă n-o ai deja — VS Code o
   sugerează automat când deschide un `.py`.

3. **Creează un virtual environment** — din terminalul integrat (`` Ctrl+` ``):
   ```bash
   python3 -m venv .venv
   ```
   VS Code va detecta `.venv` și te va întreba dacă vrei să-l folosești ca
   interpreter — apasă "Yes" (sau `Ctrl+Shift+P` → `Python: Select Interpreter`
   → alege `.venv`).

4. **Activează venv-ul și instalează dependențele:**
   ```bash
   # Linux/Mac
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate

   pip install -r requirements.txt
   ```

5. **Rulează pipeline-ul de training** (generează `models/`):
   ```bash
   python -m src.train
   ```
   Ar trebui să vezi MAE, R², ROC-AUC afișate — durează sub un minut.

6. **Pornește API-ul:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Deschide http://localhost:8000/docs — Swagger UI interactiv, poți testa
   `/predict` direct din browser.

## Debugging în VS Code (opțional dar util)

Creează `.vscode/launch.json` cu configurația de mai jos, ca să poți pune
breakpoint-uri și rula cu F5 în loc de terminal:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload", "--port", "8000"],
      "jinja": true
    },
    {
      "name": "Train pipeline",
      "type": "debugpy",
      "request": "launch",
      "module": "src.train"
    }
  ]
}
```

## Test rapid din terminal

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_cal": 5, "recency_cal": 120, "T_cal": 200,
    "monetary_value_cal": 45.5, "avg_basket_value": 45.5,
    "n_unique_products": 12, "avg_quantity": 8.3, "tenure_days": 200
  }'
```

## CI/CD — GitHub Actions

`.github/workflows/ci.yml` face automat, la fiecare push pe `main`:
1. **test** — rulează `pytest tests/` (rulează la orice push/PR, rapid, nu antrenează modelul)
2. **build-and-push** (doar pe `main`, după ce testele trec):
   - descarcă dataset-ul
   - rulează `python -m src.train` (produce artefacte proaspete în `models/`)
   - construiește imaginea Docker și o urcă în **GitHub Container Registry**
     (`ghcr.io/<user>/<repo>:latest` și `:<commit-sha>`)

Nu ai nimic de configurat manual — `GITHUB_TOKEN`-ul e generat automat de GitHub
Actions și are deja drepturi de scriere în GHCR pentru acest repo (atâta timp
cât `packages: write` e setat în workflow, ceea ce e deja acolo).

**Ca să activezi:**
1. `git init && git add . && git commit -m "initial commit"`
2. Creează un repo nou pe GitHub, apoi:
   ```bash
   git remote add origin https://github.com/<user>/<repo>.git
   git branch -M main
   git push -u origin main
   ```
3. Verifică rularea în tab-ul **Actions** al repo-ului
4. După primul build reușit, imaginea apare în tab-ul **Packages** al contului tău

**Testare locală a imaginii Docker** (dacă ai Docker instalat):
```bash
docker build -t clv-api .
docker run -p 8000:8000 clv-api
```

## Pași următori
- Deploy pe k3s cu Helm chart (pull din GHCR)
- Monitoring: metrici expuse spre Prometheus, dashboard Grafana
