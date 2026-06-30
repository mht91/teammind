"""
Vector Database Interface using Qdrant (Fully Compatible)
"""
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from loguru import logger


class VectorStore:
    """Qdrant vector database interface"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "teammind",
                 vector_size: int = 768):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Check if collection exists, create if not
        self._ensure_collection()
        
        logger.info(f"Connected to Qdrant at {host}:{port}")
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_points(self, 
                   vectors: List[np.ndarray],
                   payloads: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add points to the vector store
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            ids: Optional list of point IDs
        
        Returns:
            List of point IDs
        """
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors and payloads must match")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        points = []
        for idx, (vector, payload) in enumerate(zip(vectors, payloads)):
            points.append(
                PointStruct(
                    id=ids[idx],
                    vector=vector.tolist(),
                    payload=payload
                )
            )
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} points to collection")
            return ids
        except Exception as e:
            logger.error(f"Error adding points: {e}")
            raise
    
    def search(self,
               query_vector: np.ndarray,
               filter_conditions: Optional[Dict] = None,
               top_k: int = 10,
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding
            filter_conditions: Metadata filter conditions
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results with payloads
        """
        # Build filter
        filter_obj = None
        if filter_conditions:
            filter_obj = self._build_filter(filter_conditions)
        
        # Try different search methods based on available API
        try:
            # Method 1: Using search method (newer versions)
            if hasattr(self.client, 'search'):
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k,
                    query_filter=filter_obj,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
            # Method 2: Using search_batch method
            elif hasattr(self.client, 'search_batch'):
                search_result = self.client.search_batch(
                    collection_name=self.collection_name,
                    requests=[{
                        'vector': query_vector.tolist(),
                        'limit': top_k,
                        'filter': filter_obj,
                        'score_threshold': score_threshold,
                        'with_payload': True,
                        'with_vectors': False
                    }]
                )
                if search_result:
                    search_result = search_result[0]
                else:
                    search_result = []
            # Method 3: Using search method with different parameters
            else:
                # Fallback: try the raw client search
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    vector=query_vector.tolist(),
                    limit=top_k,
                    filter=filter_obj,
                    score_threshold=score_threshold
                )
            
            # Convert results to expected format
            results = []
            if search_result:
                for result in search_result:
                    results.append({
                        'id': result.id,
                        'score': result.score,
                        'payload': result.payload if hasattr(result, 'payload') else {},
                        'vector': None
                    })
            
            return results
            
        except AttributeError as e:
            logger.error(f"Qdrant search method not found: {e}")
            # Try alternative approach using scroll with vector search
            return self._search_with_scroll(query_vector, filter_conditions, top_k)
        except Exception as e:
            logger.error(f"Error in Qdrant search: {e}")
            raise
    
    def _search_with_scroll(self, query_vector: np.ndarray, 
                           filter_conditions: Optional[Dict],
                           top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback search using scroll and manual similarity calculation
        """
        try:
            # Build filter
            filter_obj = None
            if filter_conditions:
                filter_obj = self._build_filter(filter_conditions)
            
            # Get all points that match filter
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                query_filter=filter_obj,
                limit=100,  # Get more points for better results
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return []
            
            # Calculate similarity manually
            similarities = []
            for point in points:
                if point.vector:
                    # Compute cosine similarity
                    vec = np.array(point.vector)
                    sim = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
                    similarities.append((point, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            results = []
            for point, sim in similarities[:top_k]:
                results.append({
                    'id': point.id,
                    'score': sim,
                    'payload': point.payload if hasattr(point, 'payload') else {},
                    'vector': None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in scroll fallback search: {e}")
            return []
    
    def scroll(self,
               filter_conditions: Optional[Dict] = None,
               limit: int = 10,
               offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scroll through points with optional filter
        
        Args:
            filter_conditions: Metadata filter conditions
            limit: Number of points to return
            offset: Offset for pagination
        
        Returns:
            List of points
        """
        filter_obj = None
        if filter_conditions:
            filter_obj = self._build_filter(filter_conditions)
        
        try:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                query_filter=filter_obj,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in points:
                results.append({
                    'id': point.id,
                    'payload': point.payload if hasattr(point, 'payload') else {},
                    'vector': None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Qdrant scroll: {e}")
            return []
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from conditions"""
        if not conditions:
            return None
        
        must_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, list):
                # Match any value in list - create multiple conditions
                for v in value:
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=v)
                        )
                    )
            elif isinstance(value, dict) and 'range' in value:
                # Range filter
                range_params = value['range']
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gte=range_params.get('gte'),
                            lte=range_params.get('lte')
                        )
                    )
                )
            else:
                # Exact match
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        if not must_conditions:
            return None
            
        return Filter(
            must=must_conditions
        )
    
    def delete_points(self, ids: List[str]):
        """Delete points by ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids)
            )
            logger.info(f"Deleted {len(ids)} points")
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'name': self.collection_name,
                'error': str(e)
            }