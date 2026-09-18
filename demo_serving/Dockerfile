# model_serving - the demo serving API (house_prices, bank_marketing, bbc_news).
# Build context is the REPO ROOT (see docker-compose.yml), since app.py's own
# path logic (BASE_DIR = two levels up from this file) expects to sit at
# <root>/demo_serving/app.py, with demo_models/checkpoints mounted as a
# sibling of demo_serving/ - so the folder layout is preserved inside the
# image rather than flattened.
FROM python:3.12-slim

WORKDIR /app

# CPU-only wheel - the full PyPI "torch" package pulls in CUDA dependencies
# that are unnecessary (and multiple GB larger) for this demo's encoder model.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY demo_serving/requirements.txt demo_serving/requirements.txt
RUN pip install --no-cache-dir -r demo_serving/requirements.txt

COPY demo_serving/app.py demo_serving/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "demo_serving"]
