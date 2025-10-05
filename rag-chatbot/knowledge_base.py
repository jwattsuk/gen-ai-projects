"""
Knowledge Base Management
Handles data loading, text chunking, and storage in Cosmos DB
"""

import os
import pandas as pd
from azure.cosmos import CosmosClient
from typing import List, Dict, Any


class TextProcessor:
    """Handles text processing and chunking operations"""
    
    @staticmethod
    def split_text(text: str, max_length: int = 400, min_length: int = 300) -> List[str]:
        """
        Split text into chunks based on word count and length constraints
        
        Args:
            text: Input text to split
            max_length: Maximum character length for each chunk
            min_length: Minimum character length for each chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            current_text = ' '.join(current_chunk)
            
            if len(current_text) >= min_length and len(current_text) <= max_length:
                chunks.append(current_text)
                current_chunk = []

        # If the last chunk didn't reach the minimum length, add it anyway
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class CosmosDBClient:
    """Handles Cosmos DB operations including vector search"""
    
    def __init__(self, database_name: str = 'rag-cosmos-db', container_name: str = 'data'):
        """
        Initialize Cosmos DB client
        
        Args:
            database_name: Name of the Cosmos DB database
            container_name: Name of the container
        """
        self.url = os.getenv('COSMOS_DB_ENDPOINT')
        self.key = os.getenv('COSMOS_DB_KEY')
        
        if not self.url or not self.key:
            raise ValueError("COSMOS_DB_ENDPOINT and COSMOS_DB_KEY environment variables must be set")
        
        # Validate the key format
        self._validate_cosmos_key()
        
        try:
            self.client = CosmosClient(self.url, credential=self.key)
            self.database = self.client.get_database_client(database_name)
            self.container = self.database.get_container_client(container_name)
        except Exception as e:
            if "Incorrect padding" in str(e):
                raise ValueError(
                    f"Invalid COSMOS_DB_KEY format. The key must be a valid base64-encoded string. "
                    f"Please check your .env file and ensure the key is copied correctly from Azure Portal."
                ) from e
            else:
                raise ValueError(f"Failed to connect to Cosmos DB: {e}") from e
    
    def _validate_cosmos_key(self):
        """Validate that the Cosmos DB key is properly formatted"""
        import base64
        import binascii
        
        try:
            # Try to decode the key to validate it's proper base64
            base64.b64decode(self.key)
        except (binascii.Error, ValueError) as e:
            raise ValueError(
                f"Invalid COSMOS_DB_KEY format: {e}\n"
                f"The key should be a base64-encoded string from your Azure Cosmos DB account.\n"
                f"Key length: {len(self.key)} characters\n"
                f"Key preview: {self.key[:20]}...{self.key[-10:] if len(self.key) > 30 else ''}"
            )
    
    def store_document(self, document: Dict[str, Any]) -> None:
        """Store a document in Cosmos DB"""
        try:
            self.container.create_item(body=document)
        except Exception as e:
            print(f"Error storing document: {e}")
    
    def store_document_with_embedding(self, document: Dict[str, Any], embedding: List[float]) -> None:
        """Store a document with its embedding vector in Cosmos DB"""
        try:
            document['embedding'] = embedding
            self.container.create_item(body=document)
        except Exception as e:
            print(f"Error storing document with embedding: {e}")
    
    def query_documents(self, query: str) -> List[Dict[str, Any]]:
        """Query documents from Cosmos DB"""
        try:
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            print(f"Error querying documents: {e}")
            return []
    
    def vector_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using CosmosDB's vector capabilities
        
        Args:
            query_embedding: The embedding vector to search for
            k: Number of top results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Note: This uses a simulated vector search approach since CosmosDB
            # may not have native vector search depending on your setup.
            # For true vector search, you'd need to use CosmosDB's vector indexing
            # and vector search SQL extensions.
            
            # Get all documents with embeddings
            query = "SELECT * FROM c WHERE IS_DEFINED(c.embedding)"
            all_docs = self.query_documents(query)
            
            if not all_docs:
                return []
            
            # Calculate cosine similarity for each document
            import numpy as np
            
            results = []
            query_vector = np.array(query_embedding)
            
            for doc in all_docs:
                if 'embedding' in doc and doc['embedding']:
                    doc_vector = np.array(doc['embedding'])
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(query_vector, doc_vector)
                    query_norm = np.linalg.norm(query_vector)
                    doc_norm = np.linalg.norm(doc_vector)
                    
                    if query_norm != 0 and doc_norm != 0:
                        similarity = dot_product / (query_norm * doc_norm)
                        
                        # Add similarity score to document
                        doc['similarity_score'] = float(similarity)
                        doc['distance'] = float(1 - similarity)  # For compatibility
                        results.append(doc)
            
            # Sort by similarity (highest first) and return top k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []
    
    def get_all_documents_with_embeddings(self) -> List[Dict[str, Any]]:
        """Get all documents that have embeddings stored"""
        query = "SELECT * FROM c WHERE IS_DEFINED(c.embedding)"
        return self.query_documents(query)


class KnowledgeBase:
    """Main class for managing the knowledge base"""
    
    def __init__(self, data_paths: List[str] = None):
        """
        Initialize KnowledgeBase
        
        Args:
            data_paths: List of file paths to load data from
        """
        self.data_paths = data_paths or [
            "data/frameworks.md",
            "data/own_framework.md", 
            "data/perceptron.md"
        ]
        self.text_processor = TextProcessor()
        self.cosmos_client = CosmosDBClient()
        self.df = None
        self.flattened_df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from files into a DataFrame
        
        Returns:
            DataFrame with 'path' and 'text' columns
        """
        df = pd.DataFrame(columns=['path', 'text'])
        
        for path in self.data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                
                # Use pd.concat instead of deprecated append
                new_row = pd.DataFrame({'path': [path], 'text': [file_content]})
                df = pd.concat([df, new_row], ignore_index=True)
                
            except FileNotFoundError:
                print(f"Warning: File {path} not found. Skipping...")
            except Exception as e:
                print(f"Error reading {path}: {e}")
        
        self.df = df
        return df
    
    def process_text(self) -> pd.DataFrame:
        """
        Process text by splitting into chunks
        
        Returns:
            DataFrame with exploded chunks
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        # Create chunks
        splitted_df = self.df.copy()
        splitted_df['chunks'] = splitted_df['text'].apply(
            lambda x: self.text_processor.split_text(x)
        )
        
        # Flatten the chunks into separate rows
        self.flattened_df = splitted_df.explode('chunks').reset_index(drop=True)
        return self.flattened_df
    
    def store_to_cosmos(self, embeddings: List[List[float]] = None) -> None:
        """
        Store processed data to Cosmos DB, optionally with embeddings
        
        Args:
            embeddings: Optional list of embedding vectors corresponding to each chunk
        """
        if self.flattened_df is None:
            raise ValueError("Text must be processed first. Call process_text()")
        
        for idx, row in self.flattened_df.iterrows():
            document = {
                'id': str(idx),
                'path': row['path'],
                'text': row['text'],
                'chunk': row['chunks']
            }
            
            if embeddings and idx < len(embeddings):
                # Store with embedding if provided
                self.cosmos_client.store_document_with_embedding(document, embeddings[idx])
            else:
                # Store without embedding
                self.cosmos_client.store_document(document)
        
        print(f"Stored {len(self.flattened_df)} chunks to Cosmos DB" + 
              (" with embeddings" if embeddings else ""))
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get the processed and flattened DataFrame"""
        return self.flattened_df
    
    def build_knowledge_base(self, with_embeddings: bool = False, 
                           embedding_service = None) -> pd.DataFrame:
        """
        Complete pipeline: load data, process text, and store to Cosmos DB
        
        Args:
            with_embeddings: Whether to generate and store embeddings
            embedding_service: Optional embedding service for generating embeddings
            
        Returns:
            Processed DataFrame
        """
        print("Loading data...")
        self.load_data()
        
        print("Processing text into chunks...")
        self.process_text()
        
        embeddings = None
        if with_embeddings and embedding_service:
            print("Generating embeddings...")
            texts = self.flattened_df['chunks'].tolist()
            embeddings = embedding_service.create_embeddings_batch(texts)
        
        print("Storing to Cosmos DB...")
        self.store_to_cosmos(embeddings)
        
        print("Knowledge base build complete!")
        return self.flattened_df


if __name__ == "__main__":
    # Example usage
    kb = KnowledgeBase()
    processed_data = kb.build_knowledge_base()
    print(f"Processed {len(processed_data)} text chunks")
    print(processed_data.head())