#!/usr/bin/env python
"""
File: retrieval_module.py
Purpose:
  - Provides a simple retrieval-augmented generation (RAG) interface for storing
    and retrieving context relevant to user queries.
  - Uses OpenAI embeddings for semantic search.
  - Illustrates how to chunk text, generate embeddings, and query them.

Dependencies:
  - openai>=1.0.0
  - psycopg2 (or another DB library) if storing vectors in a relational DB
  - numpy, pandas (optional for convenience)
  - scikit-learn or FAISS or other vector store library (optional if using local in-memory index)

Notes:
  - In production, you may prefer an external vector database (e.g. Pinecone, Weaviate)
    or a specialized library (e.g. FAISS, Milvus, Chroma).
  - This module demonstrates the structure of a RAG pipeline, but you’ll need
    to adapt it to your own environment and DB specifics.

Usage Example:
  from retrieval_module import RetrievalModule

  # Initialize with a DB connection or in-memory store
  retrieval = RetrievalModule()

  # Insert documents
  doc_id = retrieval.add_document("Here is some text about cats and dogs...")

  # Query for best matching documents
  results = retrieval.query("What is the difference between cats and dogs?")
  # 'results' will be a list of (doc_id, similarity_score)
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Union

# If you're using a DB for storage:
# import psycopg2  # or any other DB library

# If you're using an in-memory approach, you can store vectors in a Python list
# or use FAISS or scikit-learn's NearestNeighbors for indexing.

from openai import OpenAI


class RetrievalModule:
    """
    A demonstration of a retrieval-augmented generation pipeline component.
    - Splits documents into chunks
    - Creates embeddings
    - Stores embeddings in an in-memory store (or a DB in production)
    - Queries embeddings to retrieve relevant chunks
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        db_connection_params: Optional[dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        :param model_name: Name of the OpenAI embedding model (e.g., "text-embedding-ada-002").
        :param db_connection_params: If using a DB, pass connection details here (host, user, etc.).
        :param logger: Optional logger instance.
        """
        self.model_name = model_name
        self.logger = logger or logging.getLogger("RetrievalModule")
        self.logger.setLevel(logging.INFO)

        # Attempt to get OPENAI_API_KEY from environment if not set
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            self.logger.warning(
                "No OPENAI_API_KEY found in environment. Calls to OpenAI may fail."
            )

        # Create an OpenAI client instance with the stored API key
        self.client = OpenAI(api_key=self.api_key)

        # If storing in DB, connect here:
        # self.db_conn = psycopg2.connect(**db_connection_params) if db_connection_params else None
        # self._create_tables_if_needed()

        # For demonstration, we’ll store embeddings in an in-memory list
        # Format: (doc_id, chunk_text, vector)
        self.in_memory_store = []

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Splits text into smaller chunks to embed individually.
        Adjust chunk_size as needed for your domain.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk_words = words[start : start + chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append(chunk_str)
            start += chunk_size
        return chunks

    def _embed_text(self, text_list: List[str]) -> List[np.ndarray]:
        """
        Calls the OpenAI embeddings.create(...) method to embed each text chunk.
        Returns a list of numpy arrays.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text_list
            )
            # Each item in response.data is an embedding result
            embeddings = [
                np.array(item["embedding"], dtype=np.float32) for item in response.data
            ]
            return embeddings
        except Exception as e:
            self.logger.error(f"Error calling OpenAI embeddings: {e}", exc_info=True)
            # Return a zero vector fallback
            return [np.zeros(1536, dtype=np.float32) for _ in text_list]

    def add_document(self, text: str) -> str:
        """
        Splits a document into chunks, embeds them, and stores them in memory (or DB).
        Returns a doc_id (in a real system, generate a unique ID).
        """
        # For demonstration, generate a naive doc_id:
        doc_id = f"doc_{len(self.in_memory_store)}"
        chunks = self._chunk_text(text)

        # Embed all chunks
        embeddings = self._embed_text(chunks)

        # Store them
        for chunk, emb in zip(chunks, embeddings):
            self.in_memory_store.append((doc_id, chunk, emb))

        self.logger.info(f"Inserted document {doc_id} with {len(chunks)} chunks.")
        return doc_id

    def query(self, query_str: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Embeds the query, retrieves top_k most similar chunks from the in-memory store.
        Returns a list of tuples (doc_id, similarity_score).
        """
        # 1) Embed the query
        query_emb_list = self._embed_text([query_str])
        if not query_emb_list:
            return []
        query_emb = query_emb_list[0]

        # 2) Compute similarity to each stored chunk
        scores = []
        for (doc_id, chunk_text, chunk_emb) in self.in_memory_store:
            # Dot product or cosine similarity
            dot = np.dot(query_emb, chunk_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_c = np.linalg.norm(chunk_emb)
            cosine_sim = dot / (norm_q * norm_c + 1e-8)
            scores.append((doc_id, cosine_sim))

        # 3) Sort by descending similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        return top_results

    # OPTIONAL: If you prefer to store embeddings in a DB:
    # def _create_tables_if_needed(self):
    #     create_query = """
    #     CREATE TABLE IF NOT EXISTS documents (
    #         id SERIAL PRIMARY KEY,
    #         doc_id TEXT,
    #         chunk_text TEXT,
    #         embedding VECTOR(1536)  -- Postgres 14+ with pgvector extension
    #     );
    #     """
    #     with self.db_conn.cursor() as cur:
    #         cur.execute(create_query)
    #         self.db_conn.commit()

    # def add_document_db(self, text: str) -> str:
    #     doc_id = ...
    #     ...
    #     return doc_id

    # def query_db(self, query_str: str, top_k: int = 3) -> List[Tuple[str, float]]:
    #     ...
    #     return [...]

    def close(self):
        """
        Closes any DB connections if used. For in-memory usage, do nothing.
        """
        # if self.db_conn:
        #     self.db_conn.close()
        pass


def demo_usage():
    """
    A small demo of how one might use the RetrievalModule in code.
    Not executed by default. You can run `python retrieval_module.py` to test.
    """
    logging.basicConfig(level=logging.INFO)
    retrieval = RetrievalModule(model_name="text-embedding-ada-002")

    doc_text = (
        "Cats are small, carnivorous mammals that are often kept as pets. "
        "They are known for their independence and agility. Dogs, on the other hand, "
        "are domesticated descendants of wolves, valued for their loyalty and trainability."
    )
    doc_id = retrieval.add_document(doc_text)
    print(f"Inserted doc_id: {doc_id}")

    query_str = "What's the difference between cats and dogs?"
    results = retrieval.query(query_str)
    print("Top results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    demo_usage()