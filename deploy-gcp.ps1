# PowerShell script for deploying to GCP Cloud Run from Windows
# Run this script from PowerShell

# Configuration
$PROJECT_ID = "folkloric-stone-470515-t5"  # Change this to your GCP project ID
# Project configuration
# $PROJECT_ID = "olist-project-7431"  # update to your GCP project ID
# olist-project-7431
$REGION = "asia-southeast2"               # Change region if needed jakarta
# $REGION = "us-central1"            # update region as needed
$SERVICE_NAME = "llm-agent-api"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Read secrets from secrets.toml if it exists
$QDRANT_URL = ""
$QDRANT_API_KEY = ""
$QDRANT_COLLECTION = "olist_products"
$DISABLE_INGEST = "1"
$OPENAI_API_KEY = ""

if (Test-Path "secrets.toml") {
    Write-Host "Reading secrets from secrets.toml..." -ForegroundColor Yellow
    $secretsContent = Get-Content "secrets.toml" -Raw
    
    if ($secretsContent -match 'QDRANT_URL\s*=\s*"([^"]+)"') { $QDRANT_URL = $matches[1] }
    if ($secretsContent -match 'QDRANT_API_KEY\s*=\s*"([^"]+)"') { $QDRANT_API_KEY = $matches[1] }
    if ($secretsContent -match 'OPENAI_API_KEY\s*=\s*"([^"]+)"') { $OPENAI_API_KEY = $matches[1] }
}

# Override from environment if available
if ($Env:QDRANT_URL) { $QDRANT_URL = $Env:QDRANT_URL }
if ($Env:QDRANT_API_KEY) { $QDRANT_API_KEY = $Env:QDRANT_API_KEY }
if ($Env:QDRANT_COLLECTION) { $QDRANT_COLLECTION = $Env:QDRANT_COLLECTION }
if ($Env:DISABLE_INGEST) { $DISABLE_INGEST = $Env:DISABLE_INGEST }
if ($Env:OPENAI_API_KEY) { $OPENAI_API_KEY = $Env:OPENAI_API_KEY }

# Validate required vars
if (-not $QDRANT_URL -or -not $QDRANT_API_KEY) {
    Write-Warning 'QDRANT_URL or QDRANT_API_KEY is not set. Set them via $Env:QDRANT_URL and $Env:QDRANT_API_KEY before deploying.'
}
if (-not $OPENAI_API_KEY) {
    Write-Warning 'OPENAI_API_KEY is not set. Set it via $Env:OPENAI_API_KEY before deploying.'
}

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Deploying to Google Cloud Run" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Service: $SERVICE_NAME"
Write-Host ""

# Step 1: Build and push image via Cloud Build (using cloudbuild.yaml for no-cache)
Write-Host "Step 1: Building and pushing fresh image to GCR (no cache)..." -ForegroundColor Yellow
gcloud builds submit --config cloudbuild.yaml .

# Step 2: Deploy to Cloud Run
Write-Host "Step 2: Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
    --image gcr.io/$PROJECT_ID/llm-agent-api:latest `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --port 8080 `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 10 `
    --timeout 300 `
    --cpu-boost `
    --set-env-vars "QDRANT_URL=$QDRANT_URL,QDRANT_API_KEY=$QDRANT_API_KEY,QDRANT_COLLECTION=$QDRANT_COLLECTION,DISABLE_INGEST=$DISABLE_INGEST,OPENAI_API_KEY=$OPENAI_API_KEY"

# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test your API:"
Write-Host "GET: $SERVICE_URL/"
Write-Host "POST: $SERVICE_URL/ask with JSON body {\"query\":\"ringkasan review parfum\"}"
Write-Host ""