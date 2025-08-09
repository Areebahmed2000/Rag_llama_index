# Use the official Python image from the Docker hub
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y gcc g++ libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    && apt-get clean

# Copy requirements.txt to container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to container
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app using Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
