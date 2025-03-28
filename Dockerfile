FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
