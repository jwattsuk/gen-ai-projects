"""
RAG Chatbot
Implements a Retrieval Augmented Generation chatbot using OpenAI and vector search
"""

import os
import openai
from typing import List, Dict, Any, Optional
from embeddings_search import EmbeddingManager
from knowledge_base import KnowledgeBase


class RAGChatBot:
    """
    Retrieval Augmented Generation Chatbot
    Combines document retrieval with language model generation
    """
    
    def __init__(self, 
                 model: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 800,
                 top_k_documents: int = 3):
        """
        Initialize the RAG chatbot
        
        Args:
            model: OpenAI model to use for generation (if None, uses AZURE_OPENAI_CHAT_DEPLOYMENT from env)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens in response
            top_k_documents: Number of documents to retrieve for context
        """
        # Use deployment name from environment if not specified
        self.model = model or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo-1106")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k_documents = top_k_documents
        
        self.embedding_manager = None
        self.conversation_history = []
        
        self._setup_openai()
    
    def _setup_openai(self):
        """Setup OpenAI configuration for Azure"""
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = "2023-07-01-preview"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not openai.api_key or not openai.api_base:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables must be set")
        
        # Check if the endpoint format looks correct
        if openai.api_base.endswith('/openai/responses'):
            print("⚠️  Warning: Your AZURE_OPENAI_ENDPOINT ends with '/openai/responses'")
            print("   This might be incorrect. Azure OpenAI endpoints typically look like:")
            print("   https://your-resource.openai.azure.com/")
            print("   You may need to remove '/openai/responses' from your endpoint.")
        
        print(f"🔧 Chat endpoint: {openai.api_base}")
        print(f"🔧 Chat API key: {openai.api_key[:8]}...{openai.api_key[-4:]}")
    
    def load_knowledge_base(self, embeddings_filepath: str = None):
        """
        Load the knowledge base and embeddings
        
        Args:
            embeddings_filepath: Path to saved embeddings (without extension)
        """
        self.embedding_manager = EmbeddingManager()
        
        if embeddings_filepath and os.path.exists(f"{embeddings_filepath}_dataframe.pkl"):
            # Load existing embeddings
            self.embedding_manager.load_embeddings(embeddings_filepath)
            print("Loaded existing embeddings")
        else:
            # Create new embeddings
            print("Creating new knowledge base and embeddings...")
            kb = KnowledgeBase()
            data = kb.build_knowledge_base()
            self.embedding_manager.process_dataframe(data)
            
            if embeddings_filepath:
                self.embedding_manager.save_embeddings(embeddings_filepath)
                print(f"Saved embeddings to {embeddings_filepath}")
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents with metadata
        """
        if self.embedding_manager is None:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")
        
        return self.embedding_manager.search_similar(query, k=self.top_k_documents)
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using OpenAI chat completion
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        # Prepare messages with context
        system_message = {
            "role": "system",
            "content": """You are an AI assistant that helps with AI and machine learning questions. 
            Use the provided context to answer questions accurately. If the context doesn't contain 
            relevant information, say so politely and provide what general knowledge you can."""
        }
        
        user_message = {
            "role": "user",
            "content": f"""Context:\n{context}\n\nQuestion: {query}"""
        }
        
        messages = [system_message] + self.conversation_history + [user_message]
        
        try:
            # For newer models like gpt-5-mini, use minimal parameters
            if 'gpt-5' in self.model.lower():
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            else:
                response = openai.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages
                )
            
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating response: {e}")
            
            if "400" in error_msg and ("max_tokens" in error_msg or "temperature" in error_msg):
                print("🔄 Retrying with minimal parameters...")
                try:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=messages
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
            
            if "404" in error_msg:
                print("🔧 This is likely a configuration issue:")
                print("1. Check your AZURE_OPENAI_ENDPOINT URL format")
                print("2. Verify your chat model deployment name")
                print("3. Ensure the model is deployed in Azure OpenAI Studio")
                print(f"4. Current endpoint: {openai.api_base}")
                print(f"5. Current model: {self.model}")
                
                if openai.api_base and '/openai/responses' in openai.api_base:
                    corrected_url = openai.api_base.replace('/openai/responses', '')
                    print(f"💡 Try updating your endpoint to: {corrected_url}")
            
            return "I apologize, but I encountered an error while generating a response."
    
    def chat(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Main chat function that combines retrieval and generation
        
        Args:
            query: User query
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary containing response, sources, and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_context(query)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Generate response
        response = self.generate_response(query, context)
        
        # Update conversation history (keep last 6 messages for context)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
        
        result = {
            "response": response,
            "query": query,
            "sources": retrieved_docs if include_sources else [],
            "context_used": context
        }
        
        return result
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()


class ChatInterface:
    """Simple command-line interface for the chatbot"""
    
    def __init__(self, chatbot: RAGChatBot):
        """
        Initialize chat interface
        
        Args:
            chatbot: RAGChatBot instance
        """
        self.chatbot = chatbot
    
    def run(self):
        """Run the interactive chat interface"""
        print("RAG Chatbot ready! Type 'quit' to exit, 'clear' to clear history.")
        print("Ask me anything about AI, machine learning, or neural networks!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.chatbot.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                # Get response
                result = self.chatbot.chat(user_input)
                
                print(f"\nBot: {result['response']}\n")
                
                # Optionally show sources
                if result['sources']:
                    print("Sources:")
                    for i, source in enumerate(result['sources']):
                        print(f"  {i+1}. {source['path']} (similarity: {source['similarity_score']:.3f})")
                    print()
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    
    # Load knowledge base
    chatbot.load_knowledge_base("embeddings")
    
    # Test with a single query
    result = chatbot.chat("What is a perceptron?")
    print("Response:", result['response'])
    print("\nSources used:")
    for i, source in enumerate(result['sources']):
        print(f"{i+1}. {source['path']} - Similarity: {source['similarity_score']:.3f}")
    
    # Uncomment to run interactive chat
    # interface = ChatInterface(chatbot)
    # interface.run()