FROM python:3.11-slim

WORKDIR /app

# System deps needed to build xgboost/scikit-learn wheels are already covered
# by manylinux wheels on PyPI, so no extra apt packages needed here.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
