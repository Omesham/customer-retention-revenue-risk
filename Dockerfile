# Use Python 3.12 as the base image.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Prevent Python from generating bytecode files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to container logs.
ENV PYTHONUNBUFFERED=1

# Install the project's Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code.
COPY src/ ./src/

# Document the API's internal port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["python", "-m", "uvicorn", "src.retention_lab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]