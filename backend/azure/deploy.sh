#!/bin/bash
# Azure CLI Deployment Script for BridgeComm Backend
# Run this script from the backend directory

set -e

echo "🚀 BridgeComm Backend Azure Deployment Script"
echo "=============================================="

# Configuration - Update these values
RESOURCE_GROUP="bridgecomm-rg"
LOCATION="eastus"
APP_NAME="bridgecomm-api"
SKU="B1"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    echo "   Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
echo "📝 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "⚠️  Not logged in to Azure. Running 'az login'..."
    az login
fi

# Show current subscription
echo "📍 Current subscription:"
az account show --query "{Name:name, ID:id}" -o table

read -p "Continue with this subscription? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Run 'az account set --subscription <subscription-id>' to change subscription"
    exit 1
fi

# Create resource group
echo "📦 Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan
echo "📋 Creating App Service Plan..."
az appservice plan create \
    --name "${APP_NAME}-plan" \
    --resource-group $RESOURCE_GROUP \
    --sku $SKU \
    --is-linux

# Create Web App
echo "🌐 Creating Web App..."
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan "${APP_NAME}-plan" \
    --runtime "PYTHON:3.11"

# Configure startup command
echo "⚙️  Configuring startup command..."
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo ""
echo "⚠️  IMPORTANT: Configure environment variables"
echo "================================================"
echo "Run the following commands with your actual Azure credentials:"
echo ""
echo "az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings \\"
echo "    AZURE_SPEECH_KEY='your-speech-key' \\"
echo "    AZURE_SPEECH_REGION='eastus' \\"
echo "    AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/' \\"
echo "    AZURE_OPENAI_API_KEY='your-openai-key' \\"
echo "    AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4' \\"
echo "    AZURE_STORAGE_CONNECTION_STRING='your-storage-connection-string' \\"
echo "    AZURE_COSMOS_ENDPOINT='https://your-cosmos.documents.azure.com:443/' \\"
echo "    AZURE_COSMOS_KEY='your-cosmos-key' \\"
echo "    AZURE_COSMOS_DATABASE_NAME='bridgecomm' \\"
echo "    AZURE_COSMOS_CONTAINER_NAME='users' \\"
echo "    APP_SECRET_KEY='your-random-secret-key'"

# Deploy code
echo ""
echo "📤 Deploying code..."
echo "Run: az webapp up --name $APP_NAME --resource-group $RESOURCE_GROUP --runtime 'PYTHON:3.11'"
echo ""
echo "Or configure GitHub Actions / Azure DevOps for CI/CD"

echo ""
echo "✅ Base infrastructure created!"
echo "🌐 Your app will be available at: https://${APP_NAME}.azurewebsites.net"
echo ""
echo "📚 Next steps:"
echo "1. Set environment variables (see above)"
echo "2. Deploy your code"
echo "3. Create Azure Cosmos DB database and container"
echo "4. Create Azure Blob Storage container"
