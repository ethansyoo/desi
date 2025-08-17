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
        
        # Check for duplicates
        existing_document = self.collection.find_one({
            "metadata.document_hash": document_hash
        })
        
        if existing_document:
            return {
                "status": "duplicate",
                "message": f"Duplicate document detected: '{filename}' - Skipping",
                "chunks_added": 0
            }
        
        # Create chunks and embeddings
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
                    "chunk_index": i
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
        """Extract a potential filename from the user query."""
        # Get all unique filenames from the database
        all_filenames = self.collection.distinct("metadata.filename")
        
        # Find if any of the filenames are mentioned in the query
        for fname in all_filenames:
            if fname in query:
                return fname
        return None

    def find_relevant_docs(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query using a hybrid metadata/vector search.
        """
        top_k = top_k or settings.ui.default_top_k
        query_embedding = self.embedding_model.embed_text(query).tolist()
        
        # Define the core vector search stage
        vector_search_stage = {
            "$vectorSearch": {
                "index": "default",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k
            }
        }
        
        # Check if the query contains a specific filename
        filename_filter_value = self._extract_filename_from_query(query)
        if filename_filter_value:
            # If a filename is found, pre-filter by it.
            # This now uses a correct MQL filter with the '$eq' operator.
            vector_search_stage["$vectorSearch"]["filter"] = {
                "metadata.filename": {
                    "$eq": filename_filter_value
                }
            }

        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "similarity": { "$meta": "vectorSearchScore" }
                }
            }
        ]
        
        relevant_docs = list(self.collection.aggregate(pipeline))
        return relevant_docs
    
    def highlight_keywords(self, text: str, query: str) -> str:
        """Highlight keywords from query in text."""
        terms = re.findall(r'\b\w+\b', query.lower())
        for term in set(terms):
            pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
            text = pattern.sub(
                r'<span style="background-color: #ffff00; font-weight: bold;">\1</span>',
                text
            )
        return text
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents."""
        total_chunks = self.collection.count_documents({})
        unique_docs = len(self.collection.distinct("metadata.document_hash"))
        
        # Get filenames
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
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents from the collection."""
        result = self.collection.delete_many({})
        
        return {
            "status": "success",
            "chunks_deleted": result.deleted_count,
            "message": f"Cleared {result.deleted_count} total chunks"
        }


class PDFProcessor:
    """Handles PDF file processing and text extraction."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        import fitz  # PyMuPDF
        
        text = ""
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    
    @staticmethod 
    def extract_text_from_file(pdf_path: str) -> str:
        """Extract text from PDF file path."""
        import fitz
        
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text