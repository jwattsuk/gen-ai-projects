# RAG Chatbot Application

A Retrieval Augmented Generation (RAG) chatbot application built with Azure OpenAI and Cosmos DB that supports **two distinct vector search modes** for educational and production use.

## How the RAG Application Works

The RAG application operates as follows:

**Knowledge Base Creation**: Documents are ingested and preprocessed, breaking them down into smaller chunks, transforming them into text embeddings, and storing them in a Vector Database (CosmosDB in this case).

**User Query**: The user asks a question.

**Retrieval**: The embedding model retrieves relevant information from our knowledge base to provide context that will be incorporated into the prompt.

**Augmented Generation**: The LLM enhances its response based on the retrieved data. This allows the response to be based not only on pre-trained data but also on relevant information from the added context.

## Vector Search Modes

This application supports **two vector search approaches** to demonstrate different vector database patterns:

### 🚀 Cached Mode (Default)
- **How it works**: All embeddings are loaded from CosmosDB into memory as a pandas DataFrame and vector similarity search is performed locally using scikit-learn
- **Best for**: 
  - Educational purposes and demonstrating vector similarity concepts
  - Smaller datasets that fit comfortably in memory
  - Faster search performance (in-memory operations)
- **Trade-offs**: Uses more memory, requires loading entire dataset at startup

### 🗄️ CosmosDB Mode 
- **How it works**: Vector similarity search is performed directly in CosmosDB without loading all data into memory
- **Best for**:
  - Production deployments with large knowledge bases
  - True vector database approach
  - Memory-constrained environments
- **Trade-offs**: Slightly slower search (database queries), more scalable

**Quick Mode Comparison:**
```bash
# Cached mode - traditional approach with in-memory search
python main.py quick-test --mode cached

# CosmosDB mode - direct vector database queries  
python main.py quick-test --mode cosmosdb
```

## Features

- **Dual Vector Search Modes**: Both cached (in-memory) and direct CosmosDB vector search
- **Modular Architecture**: Separated into distinct, reusable classes
- **Knowledge Base Management**: Load and process documents with text chunking
- **Vector Embeddings**: Generate embeddings using Azure OpenAI
- **Similarity Search**: Fast vector search using scikit-learn or CosmosDB
- **RAG Chatbot**: Combines retrieval and generation for informed responses
- **Mode Selection**: Easy switching between search approaches
- **Testing Framework**: Comprehensive evaluation with metrics like MAP
- **Interactive CLI**: Command-line interface for easy interaction

## Project Structure

```
├── main.py                 # Main orchestration script
├── knowledge_base.py       # Data loading and Cosmos DB integration
├── embeddings_search.py    # Embedding generation and vector search
├── chatbot.py              # RAG chatbot implementation
├── test_chatbot.py         # Testing and evaluation framework
├── requirements.txt        # Python dependencies
├── data/                   # Knowledge base documents
│   ├── frameworks.md
│   ├── own_framework.md
│   └── perceptron.md
└── README.md              # This file
```

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory with your Azure credentials:

```bash
# Azure Cosmos DB
COSMOS_DB_ENDPOINT=https://your-cosmosdb-account.documents.azure.com:443/
COSMOS_DB_KEY=your-64-character-cosmos-db-primary-key-here==

# Azure OpenAI - Chat Deployment
AZURE_OPENAI_API_KEY=your-32-character-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini

# Azure OpenAI - Embeddings Deployment (can be same or different instance)
AZURE_OPENAI_EMBEDDINGS_API_KEY=your-embeddings-api-key-here
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=https://your-embeddings-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
```

**Sample .env file:**
```bash
# Replace these dummy values with your actual Azure resource values
COSMOS_DB_ENDPOINT=https://myrag-cosmosdb.documents.azure.com:443/
COSMOS_DB_KEY=ABC123def456ghi789jkl012mno345pqr678stu901vwx234yz567890abcdef12==

# Azure OpenAI - Chat
AZURE_OPENAI_API_KEY=1234567890abcdef1234567890abcdef
AZURE_OPENAI_ENDPOINT=https://myrag-openai.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini

# Azure OpenAI - Embeddings (separate instance example)
AZURE_OPENAI_EMBEDDINGS_API_KEY=abcdef1234567890abcdef1234567890
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=https://eastus.api.cognitive.microsoft.com/
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Azure Resources Setup

You'll need to create the following Azure resources. Here are the Azure CLI commands to set them up:

#### Prerequisites
```bash
# Login to Azure
az login

# Set your subscription (replace with your subscription ID)
az account set --subscription "your-subscription-id"

# Create a resource group
az group create --name rag-chatbot-rg --location eastus
```

#### 1. Create Azure OpenAI Service
```bash
# Create Azure OpenAI resource
az cognitiveservices account create \
  --name myrag-openai \
  --resource-group rag-chatbot-rg \
  --location eastus \
  --kind OpenAI \
  --sku S0

# Deploy GPT-4o-mini model
az cognitiveservices account deployment create \
  --name myrag-openai \
  --resource-group rag-chatbot-rg \
  --deployment-name gpt-4o-mini \
  --model-name gpt-4o-mini \
  --model-version "2024-07-18" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard

# Deploy text-embedding-ada-002 model
az cognitiveservices account deployment create \
  --name myrag-openai \
  --resource-group rag-chatbot-rg \
  --deployment-name text-embedding-ada-002 \
  --model-name text-embedding-ada-002 \
  --model-version "2" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard

# Get your OpenAI endpoint and key
az cognitiveservices account show \
  --name myrag-openai \
  --resource-group rag-chatbot-rg \
  --query "properties.endpoint" --output tsv

az cognitiveservices account keys list \
  --name myrag-openai \
  --resource-group rag-chatbot-rg \
  --query "key1" --output tsv
```

#### 2. Create Cosmos DB
```bash
# Create Cosmos DB account
az cosmosdb create \
  --name myrag-cosmosdb \
  --resource-group rag-chatbot-rg \
  --default-consistency-level Session \
  --locations regionName=eastus

# Create database
az cosmosdb sql database create \
  --account-name myrag-cosmosdb \
  --resource-group rag-chatbot-rg \
  --name rag-cosmos-db

# Create container
az cosmosdb sql container create \
  --account-name myrag-cosmosdb \
  --database-name rag-cosmos-db \
  --resource-group rag-chatbot-rg \
  --name data \
  --partition-key-path "/id" \
  --throughput 400

# Get your Cosmos DB endpoint and key
az cosmosdb show \
  --name myrag-cosmosdb \
  --resource-group rag-chatbot-rg \
  --query "documentEndpoint" --output tsv

az cosmosdb keys list \
  --name myrag-cosmosdb \
  --resource-group rag-chatbot-rg \
  --query "primaryMasterKey" --output tsv
```

#### 3. Update Your .env File
Use the endpoints and keys from the commands above to populate your `.env` file:

**If using a single Azure OpenAI instance:**
- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_EMBEDDINGS_ENDPOINT`: Same endpoint from OpenAI resource
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_EMBEDDINGS_API_KEY`: Same key from OpenAI resource
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: `gpt-4o-mini` 
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`: `text-embedding-ada-002`

**If using separate Azure OpenAI instances (recommended for production):**
- Create two separate OpenAI resources and use their respective endpoints and keys
- This allows for better resource management and scaling

**Required environment variables:**
- `COSMOS_DB_ENDPOINT`: Output from the Cosmos DB endpoint command
- `COSMOS_DB_KEY`: Output from the Cosmos DB keys command
- `AZURE_OPENAI_API_KEY`: API key for chat deployment
- `AZURE_OPENAI_ENDPOINT`: Endpoint for chat deployment
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Name of your chat model deployment
- `AZURE_OPENAI_EMBEDDINGS_API_KEY`: API key for embeddings deployment  
- `AZURE_OPENAI_EMBEDDINGS_ENDPOINT`: Endpoint for embeddings deployment
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`: Name of your embeddings model deployment

## Usage

### Quick Start

#### Option 1: Cached Mode (Default - Educational)
Run the complete setup process with in-memory vector search:

```bash
python main.py full-setup --mode cached
```

#### Option 2: CosmosDB Mode (Production-style)
Run the complete setup process with direct CosmosDB vector search:

```bash
python main.py full-setup --mode cosmosdb
```

Both commands will:
1. Check environment variables
2. Build the knowledge base from your documents
3. Create and store embeddings appropriately for the selected mode
4. Run a quick test
5. Provide instructions for starting the interactive chat

### Mode-Specific Commands

#### Cached Mode Commands
```bash
# Build knowledge base for cached mode
python main.py build-kb --mode cached

# Create in-memory embeddings and search index
python main.py create-embeddings --mode cached

# Start interactive chat with cached search
python main.py chat --mode cached

# Run tests with cached search
python main.py test --mode cached
```

#### CosmosDB Mode Commands  
```bash
# Build knowledge base with embeddings stored in CosmosDB
python main.py build-kb --mode cosmosdb

# For CosmosDB mode, embeddings are created during build-kb
# No separate create-embeddings step needed

# Start interactive chat with direct CosmosDB search
python main.py chat --mode cosmosdb

# Run tests with CosmosDB search
python main.py test --mode cosmosdb
```

### Universal Commands

Check environment setup:
```bash
python main.py setup
```

Quick test with mode selection:
```bash
python main.py quick-test --mode cached
python main.py quick-test --mode cosmosdb
```

### Comparing Both Modes

To see the difference between modes, run quick tests with both:

```bash
# Test cached mode
python main.py quick-test --mode cached

# Test CosmosDB mode  
python main.py quick-test --mode cosmosdb
```

The output will show which search approach was used and demonstrate the difference in implementation.

## Classes Overview

### `KnowledgeBase` (knowledge_base.py)
- Loads documents from files
- Processes text into chunks
- Stores data in Cosmos DB
- Manages the document processing pipeline

### `EmbeddingService` (embeddings_search.py)
- Generates embeddings using Azure OpenAI
- Handles batch processing of texts
- Manages OpenAI API configuration

### `SearchIndex` (embeddings_search.py)
- Creates vector similarity search index
- Performs k-nearest neighbor search
- Saves/loads search indices

### `EmbeddingManager` (embeddings_search.py)
- Combines embedding service and search index with **dual-mode support**
- **Cached mode**: Processes DataFrames with embeddings and builds in-memory search index
- **CosmosDB mode**: Performs direct vector search queries against CosmosDB
- Provides unified search interface regardless of underlying mode

### `RAGChatBot` (chatbot.py)
- Implements Retrieval Augmented Generation with **configurable search modes**
- Manages conversation history
- **Mode-aware**: Automatically adapts to cached or CosmosDB search mode
- Combines document retrieval with text generation

### `ChatBotEvaluator` (test_chatbot.py)
- Evaluates chatbot performance
- Calculates various metrics (keyword presence, source relevance, etc.)
- Generates test reports

## Extending the Application

### Adding New Documents

1. Place new markdown files in the `data/` directory
2. Update the `data_paths` in `KnowledgeBase` initialization
3. Rebuild the knowledge base for your chosen mode:
   ```bash
   # For cached mode
   python main.py build-kb --mode cached
   python main.py create-embeddings --mode cached
   
   # For CosmosDB mode
   python main.py build-kb --mode cosmosdb
   ```

### Switching Between Modes

You can easily switch between modes without changing code:

```python
from chatbot import RAGChatBot

# Start with cached mode
chatbot = RAGChatBot(search_mode="cached")
chatbot.load_knowledge_base("embeddings")

# Switch to CosmosDB mode
chatbot.set_search_mode("cosmosdb")
chatbot.load_knowledge_base()  # Reloads for new mode
```

### Adding Custom Search Modes

Extend the `EmbeddingManager` class to add new search backends:

```python
class EmbeddingManager:
    def __init__(self, search_mode="cached"):
        # Add your custom mode
        if search_mode == "your_custom_mode":
            self._setup_custom_search()
    
    def _search_custom(self, query_embedding, k):
        # Your custom search implementation
        pass
```

### Creating Custom Test Cases

```python
from test_chatbot import TestCase

custom_test = TestCase(
    query="Your question here",
    expected_keywords=["keyword1", "keyword2"],
    relevant_responses=["Good response example"],
    irrelevant_responses=["Bad response example"],
    expected_sources=["expected_file.md"]
)
```

### Adding New Evaluation Metrics

Extend the `ChatBotEvaluator` class to add custom evaluation methods:

```python
def evaluate_custom_metric(self, response: str, expected: Any) -> float:
    # Your custom evaluation logic
    return score
```

## Testing

The application includes comprehensive testing capabilities:

- **Keyword Evaluation**: Checks if expected keywords appear in responses
- **Source Relevance**: Validates that relevant documents are retrieved
- **Response Quality**: Evaluates semantic similarity to expected responses
- **Mean Average Precision (MAP)**: Standard IR evaluation metric

## API Integration

The classes are designed to be easily integrated into web applications with mode selection:

```python
from chatbot import RAGChatBot

# Initialize with preferred mode
chatbot = RAGChatBot(search_mode="cosmosdb")  # or "cached"
chatbot.load_knowledge_base("embeddings")

# Use in API endpoints
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    query = request.json['query']
    result = chatbot.chat(query)
    
    # Result includes mode information
    return jsonify({
        "response": result['response'],
        "sources": result['sources'],
        "search_mode": result['search_mode'],
        "search_info": result['search_info']
    })

# Endpoint to switch modes
@app.route('/switch-mode', methods=['POST']) 
def switch_mode():
    new_mode = request.json['mode']
    chatbot.set_search_mode(new_mode)
    chatbot.load_knowledge_base("embeddings")
    return jsonify({"message": f"Switched to {new_mode} mode"})
```

## Performance Considerations

### Cached Mode
- **Memory usage**: Proportional to dataset size (all embeddings in RAM)
- **Search speed**: Very fast (milliseconds)
- **Startup time**: Slower (loads all embeddings)
- **Scalability**: Limited by available memory

### CosmosDB Mode  
- **Memory usage**: Minimal (only query embedding in memory)
- **Search speed**: Fast (database query latency)
- **Startup time**: Faster (no bulk loading)
- **Scalability**: Scales with CosmosDB throughput

### Choosing the Right Mode

**Use Cached Mode when:**
- Dataset is small-to-medium (< 100k documents)
- Demonstrating vector similarity concepts
- Maximum search speed is critical
- Memory is abundant

**Use CosmosDB Mode when:**
- Dataset is large (> 100k documents)
- Memory is constrained
- Multiple application instances need shared access
- True production vector database patterns are desired

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Run `python main.py setup` to verify
2. **Import Errors**: Install requirements with `pip install -r requirements.txt`
3. **Azure Connection Issues**: Verify your Azure credentials and endpoints
4. **Empty Responses**: Check that your knowledge base has been built and embeddings created

### Debug Mode

For detailed debugging, you can modify the scripts to add more logging or run individual components:

```python
# Test knowledge base loading
from knowledge_base import KnowledgeBase
kb = KnowledgeBase()
data = kb.load_data()
print(data.head())

# Test embedding creation
from embeddings_search import EmbeddingService
service = EmbeddingService()
embedding = service.create_embedding("test text")
print(len(embedding))

# Test both search modes
from embeddings_search import EmbeddingManager

# Test cached mode
em_cached = EmbeddingManager(search_mode="cached")
# ... load data and test

# Test CosmosDB mode
em_cosmos = EmbeddingManager(search_mode="cosmosdb") 
# ... test direct search
```

### Mode-Specific Troubleshooting

**Cached Mode Issues:**
- Memory errors: Reduce dataset size or increase available RAM
- Slow loading: Check if embeddings files are corrupted, regenerate if needed
- Search errors: Verify search index was built correctly

**CosmosDB Mode Issues:**
- Connection errors: Verify CosmosDB credentials and network access
- Slow queries: Check CosmosDB throughput settings and query complexity
- Missing embeddings: Ensure knowledge base was built with `with_embeddings=True`

````
