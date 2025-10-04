# RAG Chatbot Application

A Retrieval Augmented Generation (RAG) chatbot application built with Azure OpenAI and Cosmos DB, converted from the Jupyter notebook into modular Python scripts for easier debugging and development.

## Features

- **Modular Architecture**: Separated into distinct, reusable classes
- **Knowledge Base Management**: Load and process documents with text chunking
- **Vector Embeddings**: Generate embeddings using Azure OpenAI
- **Similarity Search**: Fast vector search using scikit-learn
- **RAG Chatbot**: Combines retrieval and generation for informed responses
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

Create a `.env` file or set the following environment variables:

```bash
# Azure Cosmos DB
COSMOS_DB_ENDPOINT=your_cosmos_db_endpoint
COSMOS_DB_KEY=your_cosmos_db_key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Azure Resources

Make sure you have:
- Azure Cosmos DB database named `rag-cosmos-db` with container `data`
- Azure OpenAI service with text embedding and chat completion models

## Usage

### Quick Start

Run the complete setup process:

```bash
python main.py full-setup
```

This will:
1. Check environment variables
2. Build the knowledge base from your documents
3. Create embeddings and search index
4. Run a quick test
5. Save embeddings for future use

### Individual Commands

Check environment setup:
```bash
python main.py setup
```

Build knowledge base only:
```bash
python main.py build-kb
```

Create embeddings only:
```bash
python main.py create-embeddings
```

Start interactive chat:
```bash
python main.py chat
```

Run test suite:
```bash
python main.py test
```

Quick test:
```bash
python main.py quick-test
```

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
- Combines embedding service and search index
- Processes DataFrames with embeddings
- Provides high-level search interface

### `RAGChatBot` (chatbot.py)
- Implements Retrieval Augmented Generation
- Manages conversation history
- Combines document retrieval with text generation

### `ChatBotEvaluator` (test_chatbot.py)
- Evaluates chatbot performance
- Calculates various metrics (keyword presence, source relevance, etc.)
- Generates test reports

## Extending the Application

### Adding New Documents

1. Place new markdown files in the `data/` directory
2. Update the `data_paths` in `KnowledgeBase` initialization
3. Rebuild the knowledge base: `python main.py build-kb`
4. Recreate embeddings: `python main.py create-embeddings`

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

The classes are designed to be easily integrated into web applications:

```python
from chatbot import RAGChatBot

# Initialize once
chatbot = RAGChatBot()
chatbot.load_knowledge_base("embeddings")

# Use in API endpoints
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    query = request.json['query']
    result = chatbot.chat(query)
    return jsonify(result)
```

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
```

## Contributing

The modular design makes it easy to contribute:

1. Each class has a single responsibility
2. Dependencies are clearly defined
3. Test cases can be added easily
4. Configuration is externalized

Feel free to extend the functionality by adding new classes or enhancing existing ones!