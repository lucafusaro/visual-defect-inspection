FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir \
    numpy pandas Pillow matplotlib fastapi uvicorn python-multipart scikit-learn \
    && pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && pip install --no-cache-dir --no-deps .

COPY artifacts/bottle_knn_model.pt artifacts/bottle_knn_model.pt

EXPOSE 8000

CMD ["uvicorn", "visual_inspection.api:app", "--host", "0.0.0.0", "--port", "8000"]