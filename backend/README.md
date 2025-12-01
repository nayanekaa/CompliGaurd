# Backend (CompliGuard) — Quick Start

This small Flask-based backend provides a demo RAG (retrieval-augmented generation) service used by the UI.

Features
- /health - simple health check
- /ingest - POST a file (pdf/docx/txt) to ingest into the vector store
- /chat - POST { question } and get a structured response with citations

Requirements
- Python 3.9+
- Install dependencies:

```
pip install -r requirements.txt
```

Run (development)

```
python main.py
```

Default: server listens on port 8000.

Gemini (optional)
------------------

You can optionally enable generation via Google's Generative Language / Gemini model for higher-quality answers.

1. Set your API key in the environment before starting the server:

```
setx GEMINI_API_KEY "YOUR_KEY_HERE"
```

2. Optionally set the model you want to call (default is `text-bison-001`):

```
setx GEMINI_MODEL "text-bison-001"
```

When enabled the RAG engine will call the Generative Language API with the retrieved passages and expect a structured response (see the frontend's `geminiService.ts` for input/output expectations). If the API is not available or returns an error, the engine will fall back to local snippet-based synthesis.

Examples

Health check:

```
curl http://localhost:8000/health
```

Upload a file:

```
curl -F "file=@/path/to/policy.pdf" http://localhost:8000/ingest
```

Chat (post a question):

```
curl -X POST -H "Content-Type: application/json" -d '{"question":"What is our sick leave policy?"}' http://localhost:8000/chat
```

Notes
- The RAG engine uses sentence-transformers and Chroma for a lightweight demo. The answer synthesis is intentionally simple; in production you'd call a stronger LLM for final generation while using the retrieved passages as context.
