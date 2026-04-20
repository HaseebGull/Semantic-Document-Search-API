## RAG DOCUMENT QA API

Backend API service for a Retrieval-Augmented Generation (RAG) system.
It enables document upload, semantic search using FAISS, and AI-powered question answering using OpenAI.


## Features

```
Document upload (.txt files)
Automatic text chunking
OpenAI embeddings generation
FAISS vector database (local storage)
Semantic search using embeddings
LLM-based question answering (GPT)
Top source tracking for responses
Document listing API
Document deletion with index rebuild
```

## Tech Stack

```
Framework: Flask
Language: Python
Embeddings: OpenAI text-embedding-3-small
LLM: OpenAI GPT-3.5 / GPT-4
Vector DB: FAISS (local)
Data processing: NumPy
File handling: Werkzeug
Environment: python-dotenv
```

## Clone repository:

```
git clone https://github.com/HaseebGull/Semantic-Document-Search-API.git
pip install -r requirements.txt
```

## Running the Application

```
python app.py
```

## API Endpoints

```
POST /upload              Upload and index TXT document
POST /documents           Add documents via JSON
POST /query               Ask question (RAG pipeline)
GET  /documents           List all documents
DELETE /documents/<id>    Delete document from index
```

## API Base URL

```
http://127.0.0.1:5000
```

API Endpoints

```
POST /upload              Upload and index TXT document
POST /query               Ask question (RAG pipeline)
GET  /documents           List all documents
DELETE /documents/<id>    Delete document from index
```

## Upload Document

```
POST /upload

Form-data:

file: .txt file
```

## Query Example

```
POST /query
{
  "question": "What is Artificial Intelligence?"
}
```

## How It Works

```
Upload document
→ Split into chunks
→ Generate embeddings
→ Store in FAISS index
→ Convert query to embedding
→ Semantic search (top-k retrieval)
→ Send context to LLM
→ Generate final answer
```

## Future Improvements

```
Add PDF/DOCX support
Add JWT authentication
Add streaming responses
Add hybrid search (BM25 + vector)
Dockerize application
Deploy to cloud (AWS / Render)
Add async background processing
```

## Author

```
Haseeb Gull
Python Backend Engineer
```
