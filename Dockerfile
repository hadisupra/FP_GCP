# Use Python 3.13.2 image as base
FROM python:3.11

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    sqlite3 \
    libsqlite3-dev \
    vim \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify critical files and folders exist
RUN echo "=== Verifying deployed files ===" && \
    ls -la /app/ && \
    echo "=== Checking isi olist db folder ===" && \
    ls -la "/app/isi olist db/" && \
    echo "=== Counting CSV files ===" && \
    find "/app/isi olist db/" -name "*.csv" -type f && \
    echo "=== Verification complete ==="

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose port
EXPOSE 8080

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
