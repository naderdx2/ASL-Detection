FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi==0.136.1 uvicorn==0.47.0 python-multipart==0.0.29 mediapipe==0.10.21 opencv-python-headless==4.11.0.86 numpy==1.24.3 Pillow==12.2.0 pydantic==2.13.4

COPY api.py .
COPY models/ ./models/
COPY labels.json .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]