from flask import Flask, request, jsonify
import uuid
from dotenv import load_dotenv

from vector_store import VectorStore
from embedding_service import get_embeddings, get_embedding
from chunking import chunk_text
from llm_service import generate_answer
import os
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

vector_store = VectorStore()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------------------
# ADD DOCUMENTS
# ---------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']

    if file.filename == '':
        return {"error": "Empty filename"}, 400

    if not file.filename.endswith(".txt"):
        return {"error": "Only .txt files are supported"}, 400

    # secure filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    # -----------------------
    # READ FILE CONTENT
    # -----------------------
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) > 1_000_000:
        return {"error": "File too large"}, 400

    os.remove(filepath)
    # -----------------------
    # PROCESS DOCUMENT
    # -----------------------
    doc_id = str(uuid.uuid4())
    title = filename

    chunks = chunk_text(content)

    embeddings = get_embeddings(chunks)

    for chunk, embedding in zip(chunks, embeddings):
        vector_store.add(embedding, {
            "doc_id": doc_id,
            "title": title,
            "content": chunk
        })

    vector_store.save()

    return {
        "message": "File uploaded and indexed",
        "doc_id": doc_id,
        "chunks": len(chunks)
    }

# ---------------------------
# QUERY
# ---------------------------
@app.route('/query', methods=['POST'])
def query():
    question = request.json.get("question")

    query_embedding = get_embedding(question)

    results = vector_store.search(query_embedding, k=5)

    if not results:
        return jsonify({"answer": "No relevant documents found.", "sources": []})

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    filtered = [r for r in results if r["score"] > 0.5]
    results = filtered if filtered else results[:2]

    top_source = results[0]

    answer = generate_answer(results, question)

    return jsonify({
        "answer": answer,
        "top_source": top_source,
        "sources": results
    })


# ---------------------------
# LIST DOCUMENTS
# ---------------------------
@app.route('/documents', methods=['GET'])
def list_documents():
    docs = {}

    for meta in vector_store.metadata:
        docs[meta["doc_id"]] = meta["title"]

    return jsonify(docs)


# ---------------------------
# DELETE DOCUMENT
# ---------------------------
@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    import numpy as np
    import faiss

    new_metadata = []
    vectors = []

    for i, meta in enumerate(vector_store.metadata):
        if meta["doc_id"] != doc_id:
            new_metadata.append(meta)
            vectors.append(vector_store.index.reconstruct(i))

    # rebuild index
    dim = vector_store.index.d
    new_index = faiss.IndexFlatIP(dim)

    if vectors:
        new_index.add(np.array(vectors).astype("float32"))

    vector_store.index = new_index
    vector_store.metadata = new_metadata
    vector_store.save()

    return jsonify({"message": "Document deleted"})


if __name__ == '__main__':
    app.run(debug=True)