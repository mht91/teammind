"""
Simple In-Memory Vector Store - No External Dependencies
"""
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import json
import os
from loguru import logger
import sqlite3


class SimpleVectorStore:
    """Simple vector store using in-memory storage with SQLite persistence"""
    
    def __init__(self, 
                 collection_name: str = "teammind", 
                 vector_size: int = 768,
                 db_path: str = "data/vector_store.db"):
        """
        Initialize the simple vector store
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
            db_path: Path to SQLite database file
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.db_path = db_path
        self.vectors = {}  # id -> vector
        self.payloads = {}  # id -> payload
        self.use_sqlite = False
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Try to use SQLite for persistence
        try:
            self._init_sqlite()
            self.use_sqlite = True
            logger.info(f"✅ Using SQLite vector store at {db_path}")
        except Exception as e:
            logger.warning(f"⚠️ SQLite not available: {e}. Using in-memory only.")
            self.use_sqlite = False
        
        logger.info(f"✅ Initialized SimpleVectorStore: {collection_name}")
    
    def _init_sqlite(self):
        """Initialize SQLite database for persistence"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector TEXT,
                payload TEXT,
                collection TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection ON vectors(collection)')
        self.conn.commit()
        
        # Load existing data from SQLite
        self._load_from_sqlite()
    
    def _load_from_sqlite(self):
        """Load existing vectors from SQLite"""
        try:
            self.cursor.execute(
                'SELECT id, vector, payload FROM vectors WHERE collection = ?',
                (self.collection_name,)
            )
            rows = self.cursor.fetchall()
            
            for row in rows:
                doc_id = row[0]
                vector = np.array(json.loads(row[1]))
                payload = json.loads(row[2])
                self.vectors[doc_id] = vector
                self.payloads[doc_id] = payload
            
            if rows:
                logger.info(f"Loaded {len(rows)} vectors from SQLite")
        except Exception as e:
            logger.error(f"Error loading from SQLite: {e}")
    
    def add_points(self, vectors: List[np.ndarray], payloads: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add points to the store
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            ids: Optional list of point IDs
        
        Returns:
            List of point IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if len(vectors) != len(payloads) or len(vectors) != len(ids):
            raise ValueError("Length of vectors, payloads, and ids must match")
        
        for idx, (vector, payload, doc_id) in enumerate(zip(vectors, payloads, ids)):
            # Store in memory
            self.vectors[doc_id] = vector
            self.payloads[doc_id] = payload
            
            # Store in SQLite if available
            if self.use_sqlite:
                try:
                    vector_json = json.dumps(vector.tolist())
                    payload_json = json.dumps(payload)
                    self.cursor.execute(
                        'INSERT OR REPLACE INTO vectors (id, vector, payload, collection) VALUES (?, ?, ?, ?)',
                        (doc_id, vector_json, payload_json, self.collection_name)
                    )
                except Exception as e:
                    logger.error(f"Error saving to SQLite: {e}")
        
        if self.use_sqlite:
            self.conn.commit()
        
        logger.info(f"✅ Added {len(vectors)} points to {self.collection_name}")
        return ids
    
    def search(self, query_vector: np.ndarray, filter_conditions: Optional[Dict] = None, 
               top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using brute force cosine similarity
        
        Args:
            query_vector: Query embedding
            filter_conditions: Metadata filter conditions
            top_k: Number of results
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results
        """
        if not self.vectors:
            logger.warning("No vectors in store")
            return []
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            logger.warning("Query vector is zero")
            return []
        
        # Compute similarities
        results = []
        for doc_id, vector in self.vectors.items():
            # Calculate cosine similarity
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
            
            # Apply score threshold
            if sim < score_threshold:
                continue
            
            # Apply filters
            payload = self.payloads.get(doc_id, {})
            if filter_conditions:
                match = True
                for key, value in filter_conditions.items():
                    if key in payload:
                        if isinstance(value, list):
                            if payload[key] not in value:
                                match = False
                                break
                        elif payload[key] != value:
                            match = False
                            break
                if not match:
                    continue
            
            results.append({
                'id': doc_id,
                'score': float(sim),
                'payload': payload,
                'vector': None
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def scroll(self, filter_conditions: Optional[Dict] = None, limit: int = 10, 
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
        results = []
        skip = offset or 0
        count = 0
        
        for doc_id, payload in self.payloads.items():
            if filter_conditions:
                match = True
                for key, value in filter_conditions.items():
                    if key in payload:
                        if isinstance(value, list):
                            if payload[key] not in value:
                                match = False
                                break
                        elif payload[key] != value:
                            match = False
                            break
                if not match:
                    continue
            
            if count >= skip:
                results.append({
                    'id': doc_id,
                    'payload': payload,
                    'vector': None
                })
                if len(results) >= limit:
                    break
            count += 1
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get store information
        
        Returns:
            Dictionary with store information
        """
        return {
            'name': self.collection_name,
            'points_count': len(self.vectors),
            'vectors_count': len(self.vectors),
            'status': 'active',
            'storage_type': 'sqlite' if self.use_sqlite else 'in-memory',
            'vector_size': self.vector_size,
            'db_path': self.db_path if self.use_sqlite else None
        }
    
    def delete_points(self, ids: List[str]):
        """
        Delete points by ID
        
        Args:
            ids: List of point IDs to delete
        """
        for doc_id in ids:
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            if doc_id in self.payloads:
                del self.payloads[doc_id]
            
            if self.use_sqlite:
                try:
                    self.cursor.execute('DELETE FROM vectors WHERE id = ?', (doc_id,))
                except Exception as e:
                    logger.error(f"Error deleting from SQLite: {e}")
        
        if self.use_sqlite:
            self.conn.commit()
        
        logger.info(f"✅ Deleted {len(ids)} points from {self.collection_name}")
    
    def clear_all(self):
        """
        Clear all points from the store
        """
        self.vectors.clear()
        self.payloads.clear()
        
        if self.use_sqlite:
            try:
                self.cursor.execute('DELETE FROM vectors WHERE collection = ?', (self.collection_name,))
                self.conn.commit()
                logger.info(f"✅ Cleared all points from {self.collection_name}")
            except Exception as e:
                logger.error(f"Error clearing SQLite: {e}")
    
    def get_all_texts(self) -> List[str]:
        """
        Get all text content from the store for debugging
        
        Returns:
            List of text strings
        """
        texts = []
        for payload in self.payloads.values():
            if 'text' in payload:
                texts.append(payload['text'])
        return texts
    
    def __del__(self):
        """Clean up SQLite connection"""
        if hasattr(self, 'conn'):
            try:
                self.conn.close()
            except:
                pass