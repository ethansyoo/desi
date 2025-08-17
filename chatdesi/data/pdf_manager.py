"""
PDF document management and search functionality.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

from .database import DatabaseManager
from ..config import settings


class EmbeddingModel:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.model.embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze(0).numpy()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or settings.ui.chunk_size
        overlap = overlap or settings.ui.chunk_overlap
        
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks


class PDFManager:
    """Manages PDF document storage, processing, and search."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embedding_model = EmbeddingModel()
        self._collection = None
    
    @property
    def collection(self):
        """Get PDF collection with lazy loading."""
        if self._collection is None:
            self._collection = self.db_manager.get_pdf_collection()
        return self._collection
    
    def compute_text_hash(self, text: str) -> str:
        """Compute hash for text deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def add_pdf_to_db(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Add PDF text to database with chunking and embeddings.
        
        Returns:
            Dict with status and details about the operation
        """
        document_hash = self.compute_text_hash(text)
        
        existing_document = self.collection.find_one({
            "metadata.document_hash": document_hash
        })
        
        if existing_document:
            return {
                "status": "duplicate",
                "message": f"Duplicate document detected: '{filename}' - Skipping",
                "chunks_added": 0
            }
        
        text_chunks = self.embedding_model.chunk_text(text)
        chunks_added = 0
        
        for i, chunk in enumerate(text_chunks):
            embedding = self.embedding_model.embed_text(chunk).tolist()
            
            document = {
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "filename": filename,
                    "document_hash": document_hash,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            }
            
            self.collection.insert_one(document)
            chunks_added += 1
        
        return {
            "status": "success",
            "message": f"Document '{filename}' added successfully",
            "chunks_added": chunks_added
        }
    
    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """
        Finds a matching filename from the database based on a partial identifier in the query.
        """
        all_filenames = self.collection.distinct("metadata.filename")
        normalized_query = query.lower().strip()

        for fname in all_filenames:
            base_fname_match = re.match(r'([a-zA-Z0-9_]+_\d{4}_[a-zA-Z0-9_]+)', fname)
            if base_fname_match:
                base_fname = base_fname_match.group(1).lower().replace('.pdf', '')
                if base_fname in normalized_query:
                    return fname
        return None

    def find_relevant_docs(self, query: str, top_k: int = None, force_vector_search: bool = False) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query, with an option to force a broad vector search.
        """
        top_k = top_k or settings.ui.default_top_k
        
        filename_filter = self._extract_filename_from_query(query)
        
        if filename_filter and not force_vector_search:
            # If a specific document is mentioned, fetch a representative sample
            doc_info = self.collection.find_one(
                {"metadata.filename": filename_filter},
                sort=[("metadata.chunk_index", -1)]
            )
            
            if not doc_info or "total_chunks" not in doc_info.get("metadata", {}):
                 return list(self.collection.find(
                     {"metadata.filename": filename_filter},
                     {"_id": 0, "text": 1, "metadata": 1}
                 ).limit(top_k))

            total_chunks = doc_info['metadata']['total_chunks']
            
            indices = {0}
            if total_chunks > 2:
                indices.add(total_chunks // 2)
            if total_chunks > 1:
                indices.add(total_chunks - 1)
            
            pipeline = [
                {"$match": {
                    "metadata.filename": filename_filter,
                    "metadata.chunk_index": {"$in": list(indices)}
                }},
                {"$sort": {"metadata.chunk_index": 1}},
                {"$project": {"_id": 0, "text": 1, "metadata": 1}}
            ]
            return list(self.collection.aggregate(pipeline))

        # For general queries or fallback searches, use vector search
        query_embedding = self.embedding_model.embed_text(query).tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "similarity": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        return list(self.collection.aggregate(pipeline))

    def rerank_and_fallback(self, query: str, initial_docs: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Re-ranks initial search results and triggers a fallback search if they are not relevant enough.
        """
        if not initial_docs:
            return self.find_relevant_docs(query, force_vector_search=True)

        # If the initial search returned documents by filename, they are likely relevant
        if "similarity" not in initial_docs[0]:
            return initial_docs

        # Calculate the average similarity of the initial results
        avg_similarity = np.mean([doc.get("similarity", 0) for doc in initial_docs])

        # If the average similarity is low, trigger a broader fallback search
        if avg_similarity < similarity_threshold:
            return self.find_relevant_docs(query, force_vector_search=True)
        
        return initial_docs

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents."""
        total_chunks = self.collection.count_documents({})
        unique_docs = len(self.collection.distinct("metadata.document_hash"))

        filenames = self.collection.distinct("metadata.filename")

        return {
            "total_chunks": total_chunks,
            "unique_documents": unique_docs,
            "filenames": filenames
        }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete all chunks of a document by filename."""
        result = self.collection.delete_many({"metadata.filename": filename})

        return {
            "status": "success" if result.deleted_count > 0 else "not_found",
            "chunks_deleted": result.deleted_count,
            "message": f"Deleted {result.deleted_count} chunks for '{filename}'"
        }


class PDFProcessor:
    """Handles PDF file processing and text extraction."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        import fitz
        
        text = ""
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text