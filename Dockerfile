# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
# CRITICAL MLOPS STEP: We use the --extra-index-url to force the CPU-only version of PyTorch.
# This prevents downloading gigabytes of unnecessary CUDA GPU drivers for an inference API.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy your application code and the saved model weights into the container
COPY app.py .
COPY model.py . 
COPY rain_nowcaster.pth .
COPY scaler.joblib .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
