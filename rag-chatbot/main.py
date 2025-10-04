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


def build_knowledge_base():
    """Build the knowledge base from scratch"""
    print("🔧 Building knowledge base...")
    
    try:
        kb = KnowledgeBase()
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


def create_embeddings(data=None, save_path="embeddings"):
    """Create embeddings and search index"""
    print("🧠 Creating embeddings and search index...")
    
    embedding_manager = EmbeddingManager()
    
    if data is None:
        # Load knowledge base first
        kb = KnowledgeBase()
        data = kb.build_knowledge_base()
    
    data_with_embeddings = embedding_manager.process_dataframe(data)
    
    # Save embeddings
    embedding_manager.save_embeddings(save_path)
    
    print(f"✅ Embeddings created and saved to {save_path}")
    return embedding_manager


def run_chatbot(embeddings_path="embeddings"):
    """Run the interactive chatbot"""
    print("🤖 Starting RAG Chatbot...")
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    chatbot.load_knowledge_base(embeddings_path)
    
    # Start interactive interface
    interface = ChatInterface(chatbot)
    interface.run()


def run_tests(embeddings_path="embeddings"):
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    # Initialize chatbot
    chatbot = RAGChatBot()
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
    evaluator.save_results(f"test_results_{timestamp}.json")
    
    with open(f"test_report_{timestamp}.txt", 'w') as f:
        f.write(report)
    
    print("✅ Test completed! Results and report saved.")


def quick_test(embeddings_path="embeddings"):
    """Run a quick test with a single query"""
    print("⚡ Quick test...")
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    chatbot.load_knowledge_base(embeddings_path)
    
    # Test query
    test_query = "What is a perceptron?"
    result = chatbot.chat(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Response: {result['response']}")
    print(f"\nSources used:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source['path']} (similarity: {source['similarity_score']:.3f})")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="RAG Chatbot Application")
    parser.add_argument('command', choices=[
        'setup', 'build-kb', 'create-embeddings', 'chat', 'test', 'quick-test', 'full-setup'
    ], help='Command to execute')
    parser.add_argument('--embeddings-path', default='embeddings', 
                       help='Path for saving/loading embeddings (default: embeddings)')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        # Just check environment
        setup_environment()
    
    elif args.command == 'build-kb':
        # Build knowledge base only
        if not setup_environment():
            return
        build_knowledge_base()
    
    elif args.command == 'create-embeddings':
        # Create embeddings only
        if not setup_environment():
            return
        create_embeddings(save_path=args.embeddings_path)
    
    elif args.command == 'chat':
        # Run chatbot
        if not setup_environment():
            return
        run_chatbot(args.embeddings_path)
    
    elif args.command == 'test':
        # Run test suite
        if not setup_environment():
            return
        run_tests(args.embeddings_path)
    
    elif args.command == 'quick-test':
        # Quick test
        if not setup_environment():
            return
        quick_test(args.embeddings_path)
    
    elif args.command == 'full-setup':
        # Complete setup from scratch
        if not setup_environment():
            return
        
        print("🚀 Running full setup...")
        data = build_knowledge_base()
        create_embeddings(data, args.embeddings_path)
        quick_test(args.embeddings_path)
        print("\n✅ Full setup complete! You can now run 'python main.py chat' to start chatting.")


if __name__ == "__main__":
    main()