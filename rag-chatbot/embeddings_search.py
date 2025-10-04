"""
Embedding Service and Search Index
Handles text embeddings generation and vector similarity search
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict
from sklearn.neighbors import NearestNeighbors
import openai
from openai import OpenAI


class EmbeddingService:
    """Handles text to embedding conversion using Azure OpenAI"""
    
    def __init__(self, model: str = None):
        """
        Initialize the embedding service
        
        Args:
            model: The embedding model/deployment to use. If None, uses AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT from env
        """
        # Use deployment name from environment if not specified
        self.model = model or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")
        self._setup_openai()
        self._validate_model()
    
    def _setup_openai(self):
        """Setup OpenAI configuration for Azure Embeddings"""
        # Get endpoints and keys
        self.api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")
        self.api_base = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
        
        if not self.api_key or not self.api_base:
            raise ValueError("AZURE_OPENAI_EMBEDDINGS_API_KEY and AZURE_OPENAI_EMBEDDINGS_ENDPOINT environment variables must be set")
        
        # Determine if this is a Cognitive Services endpoint or Azure OpenAI endpoint
        if 'cognitive.microsoft.com' in self.api_base:
            print("🔧 Detected Cognitive Services endpoint")
            self.is_cognitive_services = True
            # For Cognitive Services, we need to construct the full URL
            if not self.api_base.endswith('/'):
                self.api_base += '/'
        else:
            print("🔧 Detected Azure OpenAI endpoint")
            self.is_cognitive_services = False
            # For Azure OpenAI, use the standard configuration
            openai.api_type = "azure"
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            openai.api_version = "2023-07-01-preview"
        
        print(f"🔧 Using embeddings endpoint: {self.api_base}")
        print(f"🔧 Using embeddings API key: {self.api_key[:8]}...{self.api_key[-4:]}")
    
    def _validate_model(self):
        """Validate that the model is available"""
        try:
            # Test with a simple text
            if self.is_cognitive_services:
                test_response = self._create_embedding_cognitive_services("test")
                if test_response:
                    print(f"✅ Embedding model '{self.model}' is working")
                    print(f"   Embedding dimension: {len(test_response)}")
                else:
                    raise ValueError("No response from Cognitive Services")
            else:
                test_response = openai.embeddings.create(
                    input="test",
                    model=self.model
                )
                print(f"✅ Embedding model '{self.model}' is working")
                print(f"   Embedding dimension: {len(test_response.data[0].embedding)}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"❌ Embedding model '{self.model}' not found")
                print("🔧 Possible solutions:")
                print("1. Check your Azure deployment")
                print("2. Verify the deployment name matches your configuration")
                print("3. For Cognitive Services, check the full deployment URL")
                print(f"4. Expected URL format: {self.api_base}openai/deployments/{self.model}/embeddings")
            raise ValueError(f"Embedding model validation failed: {error_msg}")
    
    def _create_embedding_cognitive_services(self, text: str) -> List[float]:
        """Create embedding using Cognitive Services REST API"""
        import requests
        import json
        
        # Construct the full URL for Cognitive Services
        url = f"{self.api_base}openai/deployments/{self.model}/embeddings?api-version=2023-05-15"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        data = {
            "input": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                raise ValueError("Invalid response format")
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            if self.is_cognitive_services:
                return self._create_embedding_cognitive_services(text)
            else:
                response = openai.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing embedding {i+1}/{len(texts)}")
            
            embedding = self.create_embedding(text)
            embeddings.append(embedding)
        
        return embeddings


class SearchIndex:
    """Handles vector similarity search using scikit-learn"""
    
    def __init__(self, n_neighbors: int = 5, algorithm: str = 'ball_tree'):
        """
        Initialize the search index
        
        Args:
            n_neighbors: Number of nearest neighbors to find
            algorithm: Algorithm to use for nearest neighbor search
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.nbrs = None
        self.embeddings = None
        self.metadata = None
    
    def build_index(self, embeddings: List[List[float]], metadata: pd.DataFrame = None):
        """
        Build the search index from embeddings
        
        Args:
            embeddings: List of embedding vectors
            metadata: Optional metadata DataFrame associated with embeddings
        """
        self.embeddings = np.array(embeddings)
        self.metadata = metadata
        
        # Build the index
        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm
        ).fit(self.embeddings)
        
        print(f"Built search index with {len(embeddings)} embeddings")
    
    def search(self, query_embedding: List[float], k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (defaults to n_neighbors)
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.nbrs is None:
            raise ValueError("Index must be built first. Call build_index()")
        
        if k is None:
            k = self.n_neighbors
        
        # Reshape query embedding for sklearn
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        distances, indices = self.nbrs.kneighbors(query_embedding, n_neighbors=k)
        return distances[0], indices[0]
    
    def get_similar_documents(self, query_embedding: List[float], k: int = None) -> List[Dict[str, Any]]:
        """
        Get similar documents with metadata
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document info and similarity scores
        """
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            result = {
                'index': int(idx),
                'distance': float(distance),
                'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity
            }
            
            # Add metadata if available
            if self.metadata is not None and idx < len(self.metadata):
                result.update({
                    'text': self.metadata.iloc[idx].get('chunks', ''),
                    'path': self.metadata.iloc[idx].get('path', ''),
                    'original_text': self.metadata.iloc[idx].get('text', '')
                })
            
            results.append(result)
        
        return results
    
    def save_index(self, filepath: str):
        """Save the search index to file"""
        if self.embeddings is None:
            raise ValueError("No index to save")
        
        index_data = {
            'embeddings': self.embeddings.tolist(),
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm
        }
        
        with open(filepath, 'w') as f:
            json.dump(index_data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load the search index from file"""
        with open(filepath, 'r') as f:
            index_data = json.load(f)
        
        self.embeddings = np.array(index_data['embeddings'])
        self.n_neighbors = index_data['n_neighbors']
        self.algorithm = index_data['algorithm']
        
        # Rebuild the sklearn index
        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm
        ).fit(self.embeddings)
        
        print(f"Index loaded from {filepath}")


class EmbeddingManager:
    """Main class that combines embedding service and search index"""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the embedding manager
        
        Args:
            embedding_model: Model to use for embeddings
        """
        self.embedding_service = EmbeddingService(embedding_model)
        self.search_index = SearchIndex()
        self.df = None
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'chunks') -> pd.DataFrame:
        """
        Process a DataFrame by adding embeddings and building search index
        
        Args:
            df: Input DataFrame with text data
            text_column: Name of the column containing text to embed
            
        Returns:
            DataFrame with embeddings added
        """
        print("Creating embeddings...")
        texts = df[text_column].tolist()
        embeddings = self.embedding_service.create_embeddings_batch(texts)
        
        # Add embeddings to DataFrame
        df_with_embeddings = df.copy()
        df_with_embeddings['embeddings'] = embeddings
        
        print("Building search index...")
        self.search_index.build_index(embeddings, df_with_embeddings)
        
        self.df = df_with_embeddings
        return df_with_embeddings
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents given a text query
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        query_embedding = self.embedding_service.create_embedding(query)
        return self.search_index.get_similar_documents(query_embedding, k)
    
    def save_embeddings(self, filepath: str):
        """Save embeddings and search index"""
        if self.df is not None:
            # Save DataFrame with embeddings
            self.df.to_pickle(f"{filepath}_dataframe.pkl")
            print(f"DataFrame saved to {filepath}_dataframe.pkl")
        
        # Save search index
        self.search_index.save_index(f"{filepath}_index.json")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings and search index"""
        # Load DataFrame
        self.df = pd.read_pickle(f"{filepath}_dataframe.pkl")
        print(f"DataFrame loaded from {filepath}_dataframe.pkl")
        
        # Load search index and set metadata
        self.search_index.load_index(f"{filepath}_index.json")
        self.search_index.metadata = self.df


if __name__ == "__main__":
    # Example usage
    from knowledge_base import KnowledgeBase
    
    # Load knowledge base
    kb = KnowledgeBase()
    data = kb.build_knowledge_base()
    
    # Create embeddings and search index
    embedding_manager = EmbeddingManager()
    data_with_embeddings = embedding_manager.process_dataframe(data)
    
    # Save embeddings
    embedding_manager.save_embeddings("embeddings")
    
    # Test search
    results = embedding_manager.search_similar("what is a perceptron?", k=3)
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Similarity: {result['similarity_score']:.4f}")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Path: {result['path']}")
        print()