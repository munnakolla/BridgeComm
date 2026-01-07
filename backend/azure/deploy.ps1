# PowerShell Deployment Script for BridgeComm Backend
# Run this script from the backend directory

$ErrorActionPreference = "Stop"

Write-Host "🚀 BridgeComm Backend Azure Deployment Script" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Configuration - Update these values
$RESOURCE_GROUP = "bridgecomm-rg"
$LOCATION = "eastus"
$APP_NAME = "bridgecomm-api"
$SKU = "B1"

# Check if Azure CLI is installed
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Azure CLI is not installed. Please install it first." -ForegroundColor Red
    Write-Host "   Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
}

# Check if logged in
Write-Host "📝 Checking Azure login status..." -ForegroundColor Yellow
try {
    $account = az account show 2>$null | ConvertFrom-Json
} catch {
    Write-Host "⚠️  Not logged in to Azure. Running 'az login'..." -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}

# Show current subscription
Write-Host "📍 Current subscription:" -ForegroundColor Green
Write-Host "   Name: $($account.name)"
Write-Host "   ID: $($account.id)"

$continue = Read-Host "Continue with this subscription? (y/n)"
if ($continue -ne "y" -and $continue -ne "Y") {
    Write-Host "Run 'az account set --subscription <subscription-id>' to change subscription"
    exit 1
}

# Create resource group
Write-Host "📦 Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan
Write-Host "📋 Creating App Service Plan..." -ForegroundColor Yellow
az appservice plan create `
    --name "$APP_NAME-plan" `
    --resource-group $RESOURCE_GROUP `
    --sku $SKU `
    --is-linux

# Create Web App
Write-Host "🌐 Creating Web App..." -ForegroundColor Yellow
az webapp create `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --plan "$APP_NAME-plan" `
    --runtime "PYTHON:3.11"

# Configure startup command
Write-Host "⚙️  Configuring startup command..." -ForegroundColor Yellow
az webapp config set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --startup-file "uvicorn app.main:app --host 0.0.0.0 --port 8000"

Write-Host ""
Write-Host "⚠️  IMPORTANT: Configure environment variables" -ForegroundColor Red
Write-Host "================================================" -ForegroundColor Red
Write-Host "Run the following command with your actual Azure credentials:" -ForegroundColor Yellow
Write-Host ""
Write-Host @"
az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings ``
    AZURE_SPEECH_KEY='your-speech-key' ``
    AZURE_SPEECH_REGION='eastus' ``
    AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/' ``
    AZURE_OPENAI_API_KEY='your-openai-key' ``
    AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4' ``
    AZURE_STORAGE_CONNECTION_STRING='your-storage-connection-string' ``
    AZURE_COSMOS_ENDPOINT='https://your-cosmos.documents.azure.com:443/' ``
    AZURE_COSMOS_KEY='your-cosmos-key' ``
    AZURE_COSMOS_DATABASE_NAME='bridgecomm' ``
    AZURE_COSMOS_CONTAINER_NAME='users' ``
    APP_SECRET_KEY='your-random-secret-key'
"@

# Deploy code
Write-Host ""
Write-Host "📤 Deploying code..." -ForegroundColor Yellow
Write-Host "Run: az webapp up --name $APP_NAME --resource-group $RESOURCE_GROUP --runtime 'PYTHON:3.11'"
Write-Host ""
Write-Host "Or configure GitHub Actions / Azure DevOps for CI/CD"

Write-Host ""
Write-Host "✅ Base infrastructure created!" -ForegroundColor Green
Write-Host "🌐 Your app will be available at: https://$APP_NAME.azurewebsites.net" -ForegroundColor Cyan
Write-Host ""
Write-Host "📚 Next steps:" -ForegroundColor Yellow
Write-Host "1. Set environment variables (see above)"
Write-Host "2. Deploy your code"
Write-Host "3. Create Azure Cosmos DB database and container"
Write-Host "4. Create Azure Blob Storage container"
