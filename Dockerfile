FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
CMD ["./data", "./pickle/model.pkl", "./output/predictions.csv"]
