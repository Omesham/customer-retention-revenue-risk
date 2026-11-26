# Use Python 3.12 as the base image.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Prevent Python from generating bytecode files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to container logs.
ENV PYTHONUNBUFFERED=1

# Install the project's Python dependencies.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Verify Uvicorn works before completing the build.
RUN python -m uvicorn --version | grep -i uvicorn

# Copy the application source code.
COPY src/ ./src/

# Verify that FastAPI and its dependencies load successfully.
RUN python -c "from src.retention_lab.api.main import app; print(app.title)"
# Document the API's internal port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["python", "-m", "uvicorn", "src.retention_lab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]