# Document Indexing with Vector Embeddings

Python module for indexing PDF/DOCX documents using multiple chunking strategies and Google Gemini embeddings, stored in PostgreSQL with pgvector.

## Features

- Load PDF and DOCX files
- Three chunking strategies:
  - **Fixed-size**: 512 characters with 50-char overlap
  - **Sentence-based**: Respects sentence boundaries
  - **Semantic**: Groups semantically related sentences
- Google Gemini embeddings (768-dimensional vectors)
- PostgreSQL storage with pgvector

## Requirements

- Python 3.8+
- Docker (for PostgreSQL)
- Google Gemini API key

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd part2-python-module
```

### 2. Install Dependencies
```bash
pip install llama-index llama-index-embeddings-google llama-index-vector-stores-postgres psycopg2-binary python-dotenv pypdf python-docx sqlalchemy pgvector
```

### 3. Start PostgreSQL with pgvector
```bash
docker run --name jeen-postgres \
  -e POSTGRES_PASSWORD=password123 \
  -e POSTGRES_DB=jeen_documents \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### 4. Configure Environment

Create `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
POSTGRES_URL=postgresql://postgres:password123@localhost:5432/jeen_documents
```

## Usage

### Basic Usage
```bash
python index_documents.py <file_path> --strategy <strategy>
```

### Examples
```bash
# Fixed-size chunking (default)
python index_documents.py document.pdf --strategy fixed

# Sentence-based chunking
python index_documents.py report.docx --strategy sentence

# Semantic chunking
python index_documents.py article.pdf --strategy semantic
```

## Database Schema
```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    filename TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Chunking Strategies Explained

### Fixed-size
- Splits text into 512-character chunks
- 50-character overlap between chunks
- Fast and predictable

### Sentence-based
- Respects sentence boundaries
- Larger chunks (up to 1024 chars)
- Better context preservation

### Semantic
- Groups semantically related sentences
- Uses embeddings to detect topic changes
- Most intelligent but slower

## Query Database
```bash
# View all chunks
docker exec -it jeen-postgres psql -U postgres -d jeen_documents \
  -c "SELECT id, filename, split_strategy, LEFT(chunk_text, 50) FROM document_chunks;"

# Count chunks by strategy
docker exec -it jeen-postgres psql -U postgres -d jeen_documents \
  -c "SELECT split_strategy, COUNT(*) FROM document_chunks GROUP BY split_strategy;"
```

## Cleanup
```bash
# Stop and remove container
docker stop jeen-postgres
docker rm jeen-postgres
```

## Author

Iegor Kovalov - Jeen.ai Home Assignment
