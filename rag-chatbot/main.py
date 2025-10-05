"""
Main Application Script
Orchestrates the complete RAG application workflow
"""

import os
import argparse
from pathlib import Path
from chatbot import RAGChatBot, ChatInterface
from knowledge_base import KnowledgeBase
from embeddings_search import EmbeddingManager
from test_chatbot import ChatBotEvaluator, create_default_test_cases

# Auto-load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path('../.env')
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded environment variables from .env file")
    else:
        print("ℹ️  No .env file found, using system environment variables")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")


def setup_environment():
    """Check that required environment variables are set"""
    required_vars = [
        'COSMOS_DB_ENDPOINT',
        'COSMOS_DB_KEY', 
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_EMBEDDINGS_API_KEY',
        'AZURE_OPENAI_EMBEDDINGS_ENDPOINT'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    
    # Show configuration summary
    print(f"🔧 Configuration summary:")
    print(f"   Cosmos DB: {os.getenv('COSMOS_DB_ENDPOINT')}")
    print(f"   OpenAI Chat: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   Chat Deployment: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'Not set - will use default')}")
    print(f"   OpenAI Embeddings: {os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')}")
    print(f"   Embeddings Deployment: {os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT', 'text-embedding-ada-002')}")
    
    return True


def build_knowledge_base(search_mode="cached"):
    """Build the knowledge base from scratch"""
    print(f"🔧 Building knowledge base for {search_mode} mode...")
    
    try:
        kb = KnowledgeBase()
        
        if search_mode == "cosmosdb":
            # For CosmosDB mode, build with embeddings
            from embeddings_search import EmbeddingService
            embedding_service = EmbeddingService()
            data = kb.build_knowledge_base(with_embeddings=True, embedding_service=embedding_service)
        else:
            # For cached mode, build without embeddings (will be added later)
            data = kb.build_knowledge_base()
        
        print(f"✅ Knowledge base built with {len(data)} text chunks")
        return data
    
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n🔧 To fix this:")
        print("1. Check your Azure Cosmos DB account in the Azure Portal")
        print("2. Go to 'Keys' section")
        print("3. Copy the 'PRIMARY KEY' (not the connection string)")
        print("4. Update your .env file with the correct key")
        print("5. The key should be a long base64 string (usually 88 characters)")
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def create_embeddings(data=None, save_path="embeddings", search_mode="cached"):
    """Create embeddings and search index"""
    print(f"🧠 Creating embeddings and search index for {search_mode} mode...")
    
    if search_mode == "cosmosdb":
        print("📄 For CosmosDB mode, embeddings are stored directly in the database during knowledge base build.")
        return None
    
    # Cached mode - create in-memory embeddings
    try:
        embedding_manager = EmbeddingManager(search_mode="cached")
        
        if data is None:
            # Load knowledge base first
            kb = KnowledgeBase()
            data = kb.build_knowledge_base()
        
        data_with_embeddings = embedding_manager.process_dataframe(data)
        
        # Save embeddings
        embedding_manager.save_embeddings(save_path)
        
        print(f"✅ Embeddings created and saved to {save_path}")
        return embedding_manager
        
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        print("\n🔧 Common issues and solutions:")
        print("1. Check your Azure OpenAI embeddings API key and endpoint")
        print("2. Verify your embeddings deployment name is correct")
        print("3. Ensure your deployment has sufficient quota")
        print("4. Check that your data files exist and are readable")
        return None


def run_chatbot(embeddings_path="embeddings", search_mode="cached"):
    """Run the interactive chatbot"""
    print(f"🤖 Starting RAG Chatbot in {search_mode} mode...")
    
    # Initialize chatbot with specified mode
    chatbot = RAGChatBot(search_mode=search_mode)
    chatbot.load_knowledge_base(embeddings_path)
    
    # Start interactive interface
    interface = ChatInterface(chatbot)
    interface.run()


def run_tests(embeddings_path="embeddings", search_mode="cached"):
    """Run the test suite"""
    print(f"🧪 Running test suite in {search_mode} mode...")
    
    # Initialize chatbot with specified mode
    chatbot = RAGChatBot(search_mode=search_mode)
    chatbot.load_knowledge_base(embeddings_path)
    
    # Create evaluator and run tests
    evaluator = ChatBotEvaluator(chatbot)
    test_cases = create_default_test_cases()
    
    results = evaluator.run_test_suite(test_cases)
    
    # Calculate MAP score
    map_score = evaluator.calculate_map_score(test_cases)
    print(f"\n📊 Mean Average Precision (MAP) Score: {map_score:.3f}")
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluator.save_results(f"test_results_{search_mode}_{timestamp}.json")
    
    with open(f"test_report_{search_mode}_{timestamp}.txt", 'w') as f:
        f.write(report)
    
    print("✅ Test completed! Results and report saved.")


def quick_test(embeddings_path="embeddings", search_mode="cached"):
    """Run a quick test with a single query"""
    print(f"⚡ Quick test in {search_mode} mode...")
    
    # Initialize chatbot with specified mode
    chatbot = RAGChatBot(search_mode=search_mode)
    chatbot.load_knowledge_base(embeddings_path)
    
    # Test query
    test_query = "What is a perceptron?"
    result = chatbot.chat(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Search Mode: {result['search_info']['description']}")
    print(f"Response: {result['response']}")
    print(f"\nSources used:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source['path']} (similarity: {source['similarity_score']:.3f})")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="RAG Chatbot Application with Dual Search Modes")
    parser.add_argument('command', choices=[
        'setup', 'build-kb', 'create-embeddings', 'chat', 'test', 'quick-test', 'full-setup'
    ], help='Command to execute')
    parser.add_argument('--embeddings-path', default='embeddings', 
                       help='Path for saving/loading embeddings (default: embeddings)')
    parser.add_argument('--mode', choices=['cached', 'cosmosdb'], default='cached',
                       help='Search mode: cached (in-memory) or cosmosdb (direct vector search)')
    
    args = parser.parse_args()
    
    print(f"🔍 Using {args.mode.upper()} mode")
    if args.mode == "cached":
        print("   → Loads all embeddings into memory for fast search")
        print("   → Good for demonstrating vector similarity concepts")
    else:
        print("   → Queries CosmosDB directly for vector search")
        print("   → More scalable approach for production use")
    print()
    
    if args.command == 'setup':
        # Just check environment
        setup_environment()
    
    elif args.command == 'build-kb':
        # Build knowledge base only
        if not setup_environment():
            return
        build_knowledge_base(args.mode)
    
    elif args.command == 'create-embeddings':
        # Create embeddings only
        if not setup_environment():
            return
        create_embeddings(save_path=args.embeddings_path, search_mode=args.mode)
    
    elif args.command == 'chat':
        # Run chatbot
        if not setup_environment():
            return
        run_chatbot(args.embeddings_path, args.mode)
    
    elif args.command == 'test':
        # Run test suite
        if not setup_environment():
            return
        run_tests(args.embeddings_path, args.mode)
    
    elif args.command == 'quick-test':
        # Quick test
        if not setup_environment():
            return
        quick_test(args.embeddings_path, args.mode)
    
    elif args.command == 'full-setup':
        # Complete setup from scratch
        if not setup_environment():
            return
        
        print("🚀 Running full setup...")
        data = build_knowledge_base(args.mode)
        if data is not None:
            result = create_embeddings(data, args.embeddings_path, args.mode)
            if result is not None:  # Only proceed if embeddings were successful
                quick_test(args.embeddings_path, args.mode)
                print(f"\n✅ Full setup complete for {args.mode} mode!")
                print(f"💡 You can now run 'python main.py chat --mode {args.mode}' to start chatting.")
                print(f"💡 To try the other mode, use --mode {'cosmosdb' if args.mode == 'cached' else 'cached'}")
            else:
                print(f"\n❌ Setup failed during embedding creation.")
        else:
            print(f"\n❌ Setup failed during knowledge base creation.")


if __name__ == "__main__":
    main()