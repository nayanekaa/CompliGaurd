import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_engine import get_engine


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = Flask(__name__)
CORS(app)

# initialize engine once
engine = get_engine(persist_directory=DB_DIR)


@app.get("/health")
def health():
	return jsonify({"status": "ok"}), 200


@app.post("/ingest")
def ingest():
	if 'file' not in request.files:
		return jsonify({"ok": False, "error": "no file provided"}), 400

	uploaded = request.files['file']
	filename = uploaded.filename or 'upload'
	data = uploaded.read()

	# save file for auditing
	saved_path = os.path.join(UPLOAD_DIR, filename)
	with open(saved_path, 'wb') as f:
		f.write(data)

	res = engine.ingest_file(data, filename, metadata={"path": saved_path})
	if not res.get('ok'):
		return jsonify(res), 400

	return jsonify({"ok": True, "ingested": res.get('ingested', 0)}), 200


@app.post("/chat")
def chat():
	try:
		payload = request.get_json(force=True)
	except Exception:
		return jsonify({"error": "invalid_json"}), 400

	question = payload.get('question') if isinstance(payload, dict) else None
	if not question or not isinstance(question, str) or not question.strip():
		return jsonify({"error": "question is required"}), 400

	result = engine.generate_answer(question=question, top_k=3)

	response = {
		"answer": result.get('answer', ''),
		"citations": result.get('citations', []),
		"confidence": result.get('confidence', 'Low')
	}

	return jsonify(response), 200


if __name__ == '__main__':
	# Run dev server on port 8000 to match frontend API_BASE_URL
	app.run(host='0.0.0.0', port=8000, debug=True)
