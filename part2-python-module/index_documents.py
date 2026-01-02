"""
Document Indexing Script with Multiple Chunking Strategies
Uses Google Gemini embeddings and PostgreSQL with pgvector
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core.schema import TextNode

# Database imports
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise ValueError("Missing GEMINI_API_KEY or POSTGRES_URL in .env file")


def setup_database():
    """Create database table with pgvector extension"""
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding vector(768),
            filename TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database table created successfully")


def load_document(file_path: str) -> List[Document]:
    """Load document from file (PDF or DOCX)"""
    print(f"📄 Loading document: {file_path}")
    
    # Use SimpleDirectoryReader to load the file
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    print(f"✅ Loaded {len(documents)} document(s)")
    return documents


def chunk_fixed_size(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[TextNode]:
    """Strategy 1: Fixed-size chunking with overlap"""
    print(f"🔪 Chunking with fixed-size strategy (size={chunk_size}, overlap={chunk_overlap})")
    
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ Created {len(nodes)} chunks")
    return nodes


def chunk_sentence_based(documents: List[Document]) -> List[TextNode]:
    """Strategy 2: Sentence-based chunking"""
    print("🔪 Chunking with sentence-based strategy")
    
    splitter = SentenceSplitter(
        chunk_size=1024,  # Larger chunks to fit more sentences
        chunk_overlap=100,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ Created {len(nodes)} chunks")
    return nodes


def chunk_semantic(documents: List[Document], embed_model) -> List[TextNode]:
    """Strategy 3: Semantic/paragraph-based chunking"""
    print("🔪 Chunking with semantic strategy")
    
    # Using SemanticSplitter which groups semantically similar sentences
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ Created {len(nodes)} chunks")
    return nodes


def create_embeddings(nodes: List[TextNode], embed_model) -> List[tuple]:
    """Create embeddings for chunks using Gemini"""
    print(f"🧠 Creating embeddings for {len(nodes)} chunks...")
    
    chunks_with_embeddings = []
    
    for i, node in enumerate(nodes):
        # Get embedding from Gemini
        embedding = embed_model.get_text_embedding(node.get_content())
        
        chunks_with_embeddings.append((
            node.get_content(),
            embedding
        ))
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(nodes)} chunks")
    
    print(f"✅ Created {len(chunks_with_embeddings)} embeddings")
    return chunks_with_embeddings


def save_to_postgres(chunks_with_embeddings: List[tuple], filename: str, strategy: str):
    """Save chunks and embeddings to PostgreSQL"""
    print(f"💾 Saving {len(chunks_with_embeddings)} chunks to PostgreSQL...")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    
    # Register adapter for numpy arrays
    def adapt_array(arr):
        return AsIs(str(arr.tolist()))
    
    register_adapter(np.ndarray, adapt_array)
    
    for chunk_text, embedding in chunks_with_embeddings:
        # Convert embedding to PostgreSQL vector format
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        cur.execute(
            """
            INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
            VALUES (%s, %s, %s, %s)
            """,
            (chunk_text, embedding_str, filename, strategy)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data saved successfully to PostgreSQL")


def main():
    parser = argparse.ArgumentParser(description="Index documents with different chunking strategies")
    parser.add_argument("file_path", help="Path to PDF or DOCX file")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "semantic"],
        default="fixed",
        help="Chunking strategy to use (default: fixed)"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File not found: {args.file_path}")
        return
    
    filename = Path(args.file_path).name
    
    print("\n" + "="*60)
    print(f"📚 Document Indexing Pipeline")
    print(f"File: {filename}")
    print(f"Strategy: {args.strategy}")
    print("="*60 + "\n")
    
    # Setup
    setup_database()
    
    # Initialize Gemini embeddings
    print("🔧 Initializing Gemini embeddings...")
    embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=GEMINI_API_KEY
    )
    print("✅ Gemini embeddings initialized\n")
    
    # Load document
    documents = load_document(args.file_path)
    
    # Chunk based on strategy
    if args.strategy == "fixed":
        nodes = chunk_fixed_size(documents)
    elif args.strategy == "sentence":
        nodes = chunk_sentence_based(documents)
    elif args.strategy == "semantic":
        nodes = chunk_semantic(documents, embed_model)
    
    # Create embeddings
    chunks_with_embeddings = create_embeddings(nodes, embed_model)
    
    # Save to database
    save_to_postgres(chunks_with_embeddings, filename, args.strategy)
    
    print("\n" + "="*60)
    print("✅ INDEXING COMPLETE!")
    print(f"Total chunks indexed: {len(chunks_with_embeddings)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
