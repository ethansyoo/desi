"""
ADQL query generation and feedback management.
"""

import datetime
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .database import DatabaseManager
from .pdf_manager import EmbeddingModel
from ..config import settings


class ADQLManager:
    """Manages ADQL query generation, execution feedback, and learning."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embedding_model = EmbeddingModel()
        self._collection = None
    
    @property
    def collection(self):
        """Get ADQL collection with lazy loading."""
        if self._collection is None:
            self._collection = self.db_manager.get_adql_collection()
        return self._collection
    
    def log_adql_query(
        self, 
        user_query: str, 
        generated_adql: str, 
        execution_success: bool, 
        tap_result_rows: int
    ) -> str:
        """
        Log an ADQL query attempt to the database.
        
        Returns:
            The inserted document ID
        """
        embedding = self.embedding_model.embed_text(user_query).tolist()
        
        entry = {
            "user_query": user_query,
            "generated_adql": generated_adql,
            "timestamp": datetime.datetime.utcnow(),
            "execution_success": execution_success,
            "tap_result_rows": tap_result_rows,
            "user_feedback": None,
            "retry_count": 0,
            "embedding": embedding
        }
        
        result = self.collection.insert_one(entry)
        return str(result.inserted_id)
    
    def update_feedback(
        self, 
        entry_id: str, 
        feedback: str, 
        retry_count: int = 0
    ) -> bool:
        """
        Update user feedback for an ADQL query.
        
        Args:
            entry_id: Database entry ID
            feedback: "positive" or "negative"
            retry_count: Number of retry attempts
            
        Returns:
            True if update was successful
        """
        from bson import ObjectId
        
        try:
            result = self.collection.update_one(
                {"_id": ObjectId(entry_id)},
                {"$set": {
                    "user_feedback": feedback,
                    "retry_count": retry_count
                }}
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    def get_successful_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent successful queries with positive feedback."""
        return list(self.collection.find({
            "execution_success": True,
            "user_feedback": "positive"
        }).sort("timestamp", -1).limit(limit))
    
    def find_similar_adql_queries(
        self, 
        query_text: str, 
        top_k: int = 5, 
        include_negative: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find similar ADQL queries for few-shot learning.
        
        Returns:
            Dict with "positive" and "negative" example lists
        """
        query_embedding = self.embedding_model.embed_text(query_text)
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # Fetch documents with embeddings and feedback
        filter_criteria = {
            "embedding": {"$exists": True},
            "execution_success": True,
            "user_feedback": {"$in": ["positive", "negative"]}
        }
        
        all_docs = list(self.collection.find(filter_criteria))
        
        if not all_docs:
            return {"positive": [], "negative": []}
        
        # Compute similarity scores
        embeddings = np.array([doc["embedding"] for doc in all_docs])
        similarities = cosine_similarity(query_vector, embeddings).flatten()
        
        # Attach similarity scores
        for i, doc in enumerate(all_docs):
            doc["similarity"] = similarities[i]
        
        # Split by feedback type
        positive = sorted(
            [doc for doc in all_docs if doc["user_feedback"] == "positive"],
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]
        
        negative = []
        if include_negative:
            negative = sorted(
                [doc for doc in all_docs if doc["user_feedback"] == "negative"],
                key=lambda x: x["similarity"],
                reverse=True
            )[:top_k]
        
        return {"positive": positive, "negative": negative}
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about ADQL queries."""
        total_queries = self.collection.count_documents({})
        successful_queries = self.collection.count_documents({"execution_success": True})
        positive_feedback = self.collection.count_documents({"user_feedback": "positive"})
        negative_feedback = self.collection.count_documents({"user_feedback": "negative"})
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "feedback_rate": (positive_feedback + negative_feedback) / total_queries if total_queries > 0 else 0
        }


class ADQLGenerator:
    """Generates ADQL queries from natural language using OpenAI."""
    
    def __init__(self, openai_client, adql_manager: ADQLManager):
        self.client = openai_client
        self.adql_manager = adql_manager
    
    def generate_system_prompt(self, available_columns: str) -> str:
        """Generate the system prompt for ADQL generation."""
        return (
            "You are a helpful assistant that converts natural language queries into ADQL "
            "(Astronomical Data Query Language). Return only the SQL query inside a code block "
            "(```sql ... ```) and nothing else. Avoid explanations, prefaces, or post-processing text. "
            "Follow ADQL format strictly.\n\n"
            "Important rules:\n"
            "- ADQL does NOT support the `LIMIT` clause.\n"
            "- Use BETWEEN or JOIN clauses appropriately.\n"
            "- Ensure the query is executable in a TAP service.\n\n"
            f"Available columns: {available_columns}"
        )
    
    def build_few_shot_examples(self, user_input: str) -> List[Dict[str, str]]:
        """Build few-shot examples from similar queries."""
        messages = []
        
        # Get similar queries for learning
        rl_context = self.adql_manager.find_similar_adql_queries(user_input, top_k=5)
        
        # Add positive examples
        if rl_context["positive"]:
            pos_examples = "\n\n".join([
                f"NL: {doc['user_query']}\nADQL:\n{doc['generated_adql']}" 
                for doc in rl_context["positive"]
            ])
            messages.append({
                "role": "system",
                "content": f"Here are good ADQL examples you should follow:\n\n{pos_examples}"
            })
        
        # Add negative examples
        if rl_context["negative"]:
            neg_examples = "\n\n".join([
                f"NL: {doc['user_query']}\nIncorrect ADQL:\n{doc['generated_adql']}" 
                for doc in rl_context["negative"]
            ])
            messages.append({
                "role": "system",
                "content": f"Here are incorrect ADQL examples to avoid:\n\n{neg_examples}"
            })
        
        return messages
    
    def generate_adql_query(
        self, 
        user_input: str, 
        available_columns: str,
        conversation_history: List[Dict[str, str]] = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """
        Generate ADQL query from natural language.
        
        Args:
            user_input: Natural language query
            available_columns: Available database columns
            conversation_history: Previous conversation context
            temperature: OpenAI temperature parameter
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated ADQL query or None if failed
        """
        temperature = temperature or settings.model.default_temperature
        max_tokens = max_tokens or settings.model.default_token_limit
        
        # Build messages
        messages = [
            {"role": "system", "content": self.generate_system_prompt(available_columns)}
        ]
        
        # Add few-shot examples
        messages.extend(self.build_few_shot_examples(user_input))
        
        # Add conversation history
        if conversation_history:
            # Limit history to prevent token overflow
            max_history_tokens = 3000
            token_count = 0
            
            for entry in reversed(conversation_history):
                est_tokens = len(entry["content"]) // 4
                if token_count + est_tokens > max_history_tokens:
                    break
                messages.insert(-1, entry)  # Insert before user query
                token_count += est_tokens
        
        # Add current user query
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=settings.model.openai_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Extract SQL from ```sql block
            match = re.search(r"```sql\s*(.*?)\s*```", full_response, re.DOTALL)
            adql_query = match.group(1).strip() if match else full_response.strip()
            
            return adql_query
            
        except Exception as e:
            print(f"Error generating ADQL query: {e}")
            return None


class RenderUtilities:
    """Utilities for rendering mathematical content in Streamlit."""
    
    @staticmethod
    def render_openai_with_math(text: str):
        """
        Detects LaTeX-like equations and renders them nicely in Streamlit.
        Converts raw inline or block LaTeX into rendered form.
        """
        try:
            import streamlit as st
        except ImportError:
            print(text)  # Fallback for testing
            return
        
        # Convert things like [ ... ] or bare equations into $$ ... $$ blocks
        def convert_blocks(txt):
            txt = re.sub(r'\[\s*(\\?.*?)\s*\]', r'$$\1$$', txt)
            return txt
        
        text = convert_blocks(text)
        
        # Split by double newlines or blocks
        paragraphs = re.split(r"\n{2,}", text)
        
        for p in paragraphs:
            if "$$" in p:
                # For block equations
                matches = re.findall(r"\$\$(.*?)\$\$", p, flags=re.DOTALL)
                for match in matches:
                    st.latex(match.strip())
            else:
                # Inline equations are supported inside markdown
                st.markdown(p, unsafe_allow_html=True)