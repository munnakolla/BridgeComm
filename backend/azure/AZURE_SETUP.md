# Azure Services Setup Guide

This guide helps you set up all required Azure services for the BridgeComm backend.

## Prerequisites

- Azure account with an active subscription
- Azure CLI installed (`az --version`)
- Logged into Azure CLI (`az login`)

## 1. Resource Group

Create a resource group to organize all BridgeComm resources:

```bash
az group create --name bridgecomm-rg --location eastus
```

## 2. Azure Speech Services

Create a Speech Services resource:

```bash
az cognitiveservices account create \
    --name bridgecomm-speech \
    --resource-group bridgecomm-rg \
    --kind SpeechServices \
    --sku S0 \
    --location eastus \
    --yes
```

Get the keys:

```bash
az cognitiveservices account keys list \
    --name bridgecomm-speech \
    --resource-group bridgecomm-rg
```

**Environment Variables:**
```env
AZURE_SPEECH_KEY=<key1 from above>
AZURE_SPEECH_REGION=eastus
```

## 3. Azure OpenAI

### Create OpenAI Resource

```bash
az cognitiveservices account create \
    --name bridgecomm-openai \
    --resource-group bridgecomm-rg \
    --kind OpenAI \
    --sku S0 \
    --location eastus \
    --yes
```

### Deploy GPT-4 Model

1. Go to [Azure OpenAI Studio](https://oai.azure.com/)
2. Select your resource
3. Go to **Deployments**
4. Click **Create new deployment**
5. Select **gpt-4** or **gpt-4-turbo**
6. Name it `gpt-4` (or your preferred name)
7. Deploy

Get the endpoint and keys:

```bash
# Get endpoint
az cognitiveservices account show \
    --name bridgecomm-openai \
    --resource-group bridgecomm-rg \
    --query "properties.endpoint"

# Get keys
az cognitiveservices account keys list \
    --name bridgecomm-openai \
    --resource-group bridgecomm-rg
```

**Environment Variables:**
```env
AZURE_OPENAI_ENDPOINT=https://bridgecomm-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<key1 from above>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
```

## 4. Azure Blob Storage

### Create Storage Account

```bash
az storage account create \
    --name bridgecommstorage \
    --resource-group bridgecomm-rg \
    --location eastus \
    --sku Standard_LRS \
    --kind StorageV2
```

### Create Container

```bash
az storage container create \
    --name bridgecomm-media \
    --account-name bridgecommstorage \
    --public-access blob
```

### Get Connection String

```bash
az storage account show-connection-string \
    --name bridgecommstorage \
    --resource-group bridgecomm-rg
```

**Environment Variables:**
```env
AZURE_STORAGE_CONNECTION_STRING=<connection string from above>
AZURE_STORAGE_CONTAINER_NAME=bridgecomm-media
```

## 5. Azure Cosmos DB

### Create Cosmos DB Account

```bash
az cosmosdb create \
    --name bridgecomm-cosmos \
    --resource-group bridgecomm-rg \
    --kind GlobalDocumentDB \
    --default-consistency-level Session \
    --locations regionName=eastus failoverPriority=0
```

### Create Database

```bash
az cosmosdb sql database create \
    --account-name bridgecomm-cosmos \
    --resource-group bridgecomm-rg \
    --name bridgecomm
```

### Create Container

```bash
az cosmosdb sql container create \
    --account-name bridgecomm-cosmos \
    --resource-group bridgecomm-rg \
    --database-name bridgecomm \
    --name users \
    --partition-key-path /user_id \
    --throughput 400
```

### Get Endpoint and Key

```bash
# Get endpoint
az cosmosdb show \
    --name bridgecomm-cosmos \
    --resource-group bridgecomm-rg \
    --query "documentEndpoint"

# Get key
az cosmosdb keys list \
    --name bridgecomm-cosmos \
    --resource-group bridgecomm-rg \
    --query "primaryMasterKey"
```

**Environment Variables:**
```env
AZURE_COSMOS_ENDPOINT=https://bridgecomm-cosmos.documents.azure.com:443/
AZURE_COSMOS_KEY=<primary key from above>
AZURE_COSMOS_DATABASE_NAME=bridgecomm
AZURE_COSMOS_CONTAINER_NAME=users
```

## 6. (Optional) Azure Key Vault

For production, store secrets in Key Vault:

```bash
# Create Key Vault
az keyvault create \
    --name bridgecomm-kv \
    --resource-group bridgecomm-rg \
    --location eastus

# Store secrets
az keyvault secret set --vault-name bridgecomm-kv --name "speech-key" --value "<your-key>"
az keyvault secret set --vault-name bridgecomm-kv --name "openai-key" --value "<your-key>"
az keyvault secret set --vault-name bridgecomm-kv --name "cosmos-key" --value "<your-key>"
az keyvault secret set --vault-name bridgecomm-kv --name "storage-connection" --value "<your-string>"
```

**Environment Variables:**
```env
AZURE_KEY_VAULT_URL=https://bridgecomm-kv.vault.azure.net/
```

## Complete .env File

After setting up all services, your `.env` file should look like:

```env
# Azure Speech Services
AZURE_SPEECH_KEY=abc123...
AZURE_SPEECH_REGION=eastus

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://bridgecomm-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=xyz789...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=bridgecommstorage;...
AZURE_STORAGE_CONTAINER_NAME=bridgecomm-media

# Azure Cosmos DB
AZURE_COSMOS_ENDPOINT=https://bridgecomm-cosmos.documents.azure.com:443/
AZURE_COSMOS_KEY=abc...==
AZURE_COSMOS_DATABASE_NAME=bridgecomm
AZURE_COSMOS_CONTAINER_NAME=users

# Application
APP_SECRET_KEY=your-random-32-character-secret-key
APP_DEBUG=false
APP_CORS_ORIGINS=["https://your-frontend-url.com"]
```

## Cost Estimation (Monthly)

| Service | SKU | Estimated Cost |
|---------|-----|----------------|
| Azure Speech | S0 | ~$1-10 (pay per use) |
| Azure OpenAI | S0 | ~$10-100 (depends on usage) |
| Azure Storage | Standard LRS | ~$1-5 |
| Azure Cosmos DB | 400 RU/s | ~$24 |
| Azure App Service | B1 | ~$13 |

**Total estimated: $50-150/month** (varies based on usage)

## Verification

After setup, verify all services are working:

```bash
# Test Speech service
curl -X POST "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1" \
  -H "Ocp-Apim-Subscription-Key: YOUR_SPEECH_KEY" \
  --data-binary @test-audio.wav

# Test OpenAI
curl "https://bridgecomm-openai.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01" \
  -H "Content-Type: application/json" \
  -H "api-key: YOUR_OPENAI_KEY" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Test your API
curl http://localhost:8000/health
```

## Troubleshooting

### "Resource not found" errors
- Ensure the resource is in the correct region
- Verify the resource name is spelled correctly

### "Unauthorized" errors
- Check that API keys are correct
- Ensure keys haven't been rotated
- Verify the endpoint URL is correct

### Cosmos DB errors
- Ensure the database and container exist
- Check partition key configuration matches code

### Speech errors
- Verify audio format is supported (WAV, MP3)
- Check language code is valid

## Next Steps

1. ✅ All Azure services configured
2. 🔜 Update `.env` file with your credentials
3. 🔜 Deploy the backend to Azure App Service
4. 🔜 Test all API endpoints
5. 🔜 Configure CORS for your frontend domain
