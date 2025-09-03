import os
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from utils.read_process_data import load_and_preprocess_pdfs
from utils.chunk_embeddings import chunk_and_embed_documents
from utils.vector_database import QdrantDB
from utils.retrieval_qa import RetrievalQA

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

# Initialize Qdrant vector database
collection_name = "indian_legal_docs"
vdb = QdrantDB(
    url=os.getenv("QDRANT_URL"),  
    api_key=os.getenv("QDRANT_API_KEY"),  
    collection_name=collection_name
)

# Check if collection exists and has documents
if not vdb.collection_exists() or vdb.get_collection_info()["points_count"] == 0:
    print("Building Qdrant vector collection from PDFs...")
    t0 = time.time()

    # 1) Load and preprocess PDFs
    docs = load_and_preprocess_pdfs(DATA_DIR)
    print(f"- Loaded and preprocessed pages: {len(docs)}")

    # 2) Chunk documents and create embeddings
    print("- Chunking documents and creating embeddings...")
    chunk_and_embed_documents(
        docs=docs,
        vdb=vdb,
        chunk_size=3000,
        chunk_overlap=300,
        separators=["\n\n", "\n", "Section ", "SECTION ", "Sec. ", "CHAPTER ", "Chapter ", " "],
        batch_size=16  # smaller batch for API calls
    )
    
    print(f"Collection build completed in {time.time() - t0:.1f}s.")
else:
    print("Using existing Qdrant collection...")
    info = vdb.get_collection_info()
    print(f"- Collection has {info['points_count']} vectors")

# Initialize QA system
qa = RetrievalQA(vdb)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/health")
def health():
    try:
        info = vdb.get_collection_info()
        return jsonify({
            "status": "OK", 
            "collection_points": info.get("points_count", 0)
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route("/logs")
def logs():
    try:
        info = vdb.get_collection_info()
        return jsonify({
            "logs": [
                "System started successfully",
                f"Qdrant collection loaded with {info.get('points_count', 0)} vectors",
                "Legal AI chatbot ready for queries"
            ]
        })
    except Exception as e:
        return jsonify({"logs": [f"Error getting collection info: {str(e)}"]})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("question") or "").strip()
    if not query:
        return jsonify({"error": "Empty question"}), 400

    try:
        answer, hits = qa.ask(query, top_k=6, max_tokens=1200)

        # Clean sources: keep only PDF file name, not full path
        cleaned_hits = []
        for h in hits:
            payload = h.get("payload", {})
            if "source" in payload:
                payload["source"] = os.path.basename(payload["source"])
            cleaned_hits.append({
                "rank": h.get("rank"),
                "score": h.get("score"),
                "text": h.get("text"),
                "meta": payload
            })

        return jsonify({"answer": answer, "sources": cleaned_hits})
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"error": "An error occurred while processing your question."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)