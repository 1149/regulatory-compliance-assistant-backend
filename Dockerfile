# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster
# Use Python 3.11 or higher for numpy compatibility

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to leverage Docker's caching
COPY requirements.txt ./

# Install system dependencies required for some Python packages (e.g., psycopg2-binary, spacy)
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    build-essential \
    python3-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_md --no-deps

# Copy the rest of the application code
COPY . /app

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]