# Dockerfile for StableLM Inference API
#
# FastAPI + vLLM powered inference server for the fine-tuned
# and aligned StableLM 1.6B model.
#
# Build:
#   docker build -t stablelm-api -f docker/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 --gpus all stablelm-api

FROM vllm/vllm-openai:latest

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source code
COPY api/ ./api/

# Expose the API port
EXPOSE 8000

# Start the inference server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
