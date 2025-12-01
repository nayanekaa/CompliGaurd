import os
import io
import math
import importlib
from types import ModuleType
from typing import List, Dict, Any, Optional
import requests
import os


class RAGEngine:
	"""A lightweight Retrieval-Augmented Generation (RAG) helper using
	sentence-transformers for embeddings and Chroma as the vector store.

	This implementation is intentionally simple and deterministic — it
	performs document ingestion (pdf/docx/txt), creates chunks, embeds them,
	and uses nearest neighbor retrieval to produce a short answer and citations.
	"""

	DEFAULT_CHUNK_SIZE = 800
	DEFAULT_CHUNK_OVERLAP = 100

	def __init__(self, persist_directory: str = "./chroma_db"):
		self.persist_directory = persist_directory
		os.makedirs(os.path.dirname(self.persist_directory) or '.', exist_ok=True)

		# Start Chroma client (file persistence if directory provided)
		chroma = self._import_module_optional('chromadb')
		if chroma is None:
			raise RuntimeError('chromadb package is required for the RAG engine. Install chromadb')

		# Different chromadb builds expose different APIs. Prefer PersistentClient if present
		if hasattr(chroma, 'PersistentClient'):
			# newer chromadb has PersistentClient when using local persistence
			self.client = getattr(chroma, 'PersistentClient')(path=self.persist_directory)
		else:
			# fallback: try Client + Settings
			cfg = self._import_module_optional('chromadb.config')
			Settings = getattr(cfg, 'Settings', None) if cfg is not None else None
			if Settings is not None and hasattr(chroma, 'Client'):
				self.client = getattr(chroma, 'Client')(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory))
			elif hasattr(chroma, 'Client'):
				self.client = getattr(chroma, 'Client')()
			else:
				raise RuntimeError('chromadb client APIs not found in installed chromadb package')
		# Collection name
		self.collection = self.client.get_or_create_collection(name="compliguard_docs")

		# Embedding model (lazy import so missing packages don't fail static checks)
		st_mod = self._import_module_optional('sentence_transformers')
		if st_mod is None:
			raise RuntimeError('sentence-transformers package is required for embedding generation')
		SentenceTransformer = getattr(st_mod, 'SentenceTransformer', None)
		if SentenceTransformer is None:
			raise RuntimeError('sentence_transformers.SentenceTransformer not found in sentence-transformers package')
		self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

	# ---- Ingest helpers ----
	def _extract_text_from_pdf(self, stream: io.BytesIO) -> str:
		pypdf2 = self._import_module_optional('PyPDF2')
		if pypdf2 is None:
			raise RuntimeError('PyPDF2 is required to extract PDF text')
		PdfReader = getattr(pypdf2, 'PdfReader', None)
		if PdfReader is None:
			raise RuntimeError('PyPDF2.PdfReader not available')

		reader = PdfReader(stream)
		texts = []
		for page in reader.pages:
			try:
				texts.append(page.extract_text() or "")
			except Exception:
				texts.append("")

		return "\n\n".join(texts)

	def _extract_text_from_docx(self, stream: io.BytesIO) -> str:
		docx_mod = self._import_module_optional('docx')
		if docx_mod is None:
			raise RuntimeError('python-docx is required to extract DOCX text')
		Document = getattr(docx_mod, 'Document', None)
		if Document is None:
			raise RuntimeError('docx.Document not available from python-docx package')

		doc = Document(stream)
		texts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
		return "\n\n".join(texts)

	def _extract_text_from_bytes(self, data: bytes, filename: str) -> str:
		lower = filename.lower()
		stream = io.BytesIO(data)
		if lower.endswith(".pdf"):
			return self._extract_text_from_pdf(stream)
		elif lower.endswith(".docx") or lower.endswith(".doc"):
			return self._extract_text_from_docx(stream)
		else:
			# Attempt to decode as utf-8 text
			try:
				return data.decode("utf-8")
			except Exception:
				# As a fallback, return empty string
				return ""

	def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
		if chunk_size is None:
			chunk_size = self.DEFAULT_CHUNK_SIZE
		if overlap is None:
			overlap = self.DEFAULT_CHUNK_OVERLAP

		if not text:
			return []

		text = text.replace("\r\n", "\n").strip()
		chunks: List[str] = []
		start = 0
		length = len(text)
		while start < length:
			end = min(start + chunk_size, length)
			chunk = text[start:end].strip()
			# avoid tiny chunks
			if len(chunk) >= 20:
				chunks.append(chunk)
			if end == length:
				break
			start = max(0, end - overlap)

		return chunks

	def ingest_file(self, data: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Ingest a single file into the vector store.

		Args:
			data: raw bytes of the file upload
			filename: name used as the source label
			metadata: optional metadata to attach to the records

		Returns:
			dict with counts and status
		"""
		metadata = metadata or {}

		raw_text = self._extract_text_from_bytes(data, filename)
		if not raw_text:
			return {"ok": False, "reason": "empty_or_unsupported_file", "ingested": 0}

		chunks = self._chunk_text(raw_text)
		if not chunks:
			return {"ok": False, "reason": "no_chunks_generated", "ingested": 0}

		# Prepare documents and metadatas
		ids = []
		documents = []
		metadatas = []
		for idx, chunk in enumerate(chunks):
			ids.append(f"{filename}::{idx}")
			documents.append(chunk)
			md = {"source": filename, "chunk_index": idx}
			md.update(metadata)
			metadatas.append(md)

		# compute embeddings
		embeddings = self.embed_model.encode(documents, show_progress_bar=False)

		# add to chroma collection
		emb_payload = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
		self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=emb_payload)

		# persist
		try:
			self.client.persist()
		except Exception:
			# persist may not be available in all host configs
			pass

		return {"ok": True, "ingested": len(documents)}

	# ---- Retrieval / Answering ----
	def query(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
		"""Return the top_k nearest chunks for the question."""
		if not question or (hasattr(self.collection, 'count') and self.collection.count() == 0):
			return []

		q_emb = self.embed_model.encode([question], show_progress_bar=False)
		query_results = self.collection.query(query_embeddings=q_emb.tolist() if hasattr(q_emb, 'tolist') else q_emb, n_results=top_k, include=["metadatas", "documents", "distances"])

		# Query returns lists in structure - normalize to dicts
		results = []
		for i in range(len(query_results.get("ids", [[]])[0])):
			res = {
				"id": query_results.get("ids", [[]])[0][i],
				"distance": query_results.get("distances", [[]])[0][i],
				"document": query_results.get("documents", [[]])[0][i],
				"metadata": query_results.get("metadatas", [[]])[0][i],
			}
			results.append(res)

		return results

	def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
		"""Create a simple structured response given a question and the top retrieved passages.

		This DOES NOT call an external LLM — it synthesizes a short answer directly from retrieved passages
		and returns citations and a confidence level based on the distances.
		"""
		if not question:
			return {"answer": "", "citations": [], "confidence": "Low"}

		hits = self.query(question, top_k=top_k)
		if not hits:
			return {"answer": "I cannot find this information in the current policy documents.", "citations": [], "confidence": "Low"}

		# Build answer as concatenated text of the top passages (safely trimmed)
		snippets = []
		citations = []
		distances = []

		for hit in hits:
			doc = hit.get("document", "")
			excerpt = doc.strip()[:400]
			snippets.append(excerpt)
			distances.append(hit.get("distance", 1.0))
			md = hit.get("metadata", {}) or {}
			citations.append({
				"source": md.get("source", "unknown"),
				"page": md.get("page", None),
				"excerpt": excerpt,
			})

		# If a Gemini/GenAI API key is available, prefer calling the LLM to synthesize
		# a more fluent, auditable response using the retrieved passages as context.
		gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
		gemini_model = os.environ.get('GEMINI_MODEL', 'text-bison-001')
		if gemini_api_key:
			try:
				llm_result = self._call_gemini(question, snippets, gemini_api_key, gemini_model)
				if llm_result:
					# Expect LLM to return structured fields { answer, confidence, citations }
					return llm_result
			except Exception:
				# On failure, fall back to local synthesis
				pass

		# Simple fallback synthesis — pick the first 1-2 snippets as the 'answer'
		joined = " \n\n ".join(snippets[:2])
		answer = f"Based on the policy text: {joined}"

		# Confidence based on mean distance: lower distance -> higher similarity -> higher confidence
		mean_dist = sum(distances) / len(distances) if distances else 1.0
		# Distances in Chroma with sentence-transformers are usually in range ~0.0..1.0
		if mean_dist < 0.25:
			confidence = "High"
		elif mean_dist < 0.45:
			confidence = "Medium"
		else:
			confidence = "Low"

		return {"answer": answer, "citations": citations, "confidence": confidence}

	def _call_gemini(self, question: str, snippets: List[str], api_key: str, model: str) -> Optional[Dict[str, Any]]:
		"""Call Google Generative Language API (text generation) to synthesize a structured response.

		We construct a system prompt requiring the LLM to ONLY use the provided context and to
		return an auditable, delimited response with sections: __ANSWER__, __CONFIDENCE__, __EVIDENCE__, __ACTION__

		The function supports using the API key as a simple key query parameter (works for API keys in many GCP setups).
		"""
		endpoint = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate"

		context_text = "\n\n---BEGIN CONTEXT---\n\n" + "\n\n".join(snippets[:6]) + "\n\n---END CONTEXT---\n\n"

		system_instruction = (
			"You are CompliGuard, an enterprise compliance assistant. Answer the question ONLY using the provided CONTEXT. "
			"Format the response exactly with sections: __ANSWER__, __CONFIDENCE__, __EVIDENCE__, __ACTION__. "
			"If the information is not in the context, say 'I cannot find this information in the current policy documents.' and set confidence to 'Low'. "
			"Provide verbatim excerpts for evidence entries."
		)

		prompt = f"SYSTEM INSTRUCTION:\n{system_instruction}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\n"

		payload = {
			"prompt": {"text": prompt},
			"temperature": 0.05,
			"maxOutputTokens": 512,
		}

		params = {"key": api_key}

		resp = requests.post(endpoint, params=params, json=payload, timeout=20.0)
		if not resp.ok:
			raise RuntimeError(f"Gemini API call failed: {resp.status_code} {resp.text}")

		data = resp.json()

		# Response shape may vary; try common candidate fields
		text_out = None
		if isinstance(data, dict):
			# v1beta2 typically contains 'candidates' with 'output' or 'content'
			if 'candidates' in data and len(data['candidates']) > 0:
				cand = data['candidates'][0]
				text_out = cand.get('output') or cand.get('content') or cand.get('text')
			# Some variants use 'output' top-level
			if text_out is None:
				text_out = data.get('output') or data.get('content')

		if not text_out:
			return None

		# Parse LLM response for the structured sections
		parsed = {
			'answer': '',
			'citations': [],
			'confidence': 'Low'
		}

		try:
			ans_match = text_out.split('__CONFIDENCE__')[0].split('__ANSWER__')[-1].strip()
			parsed['answer'] = ans_match

			# Confidence
			if '__CONFIDENCE__' in text_out:
				conf_blk = text_out.split('__CONFIDENCE__')[1]
				conf_val = conf_blk.split('__EVIDENCE__')[0].strip().split('\n')[0].strip()
				if conf_val in ('High', 'Medium', 'Low'):
					parsed['confidence'] = conf_val

			# Evidence
			if '__EVIDENCE__' in text_out:
				ev_blk = text_out.split('__EVIDENCE__')[1]
				if '__ACTION__' in ev_blk:
					ev_blk = ev_blk.split('__ACTION__')[0]
				lines = [l.strip() for l in ev_blk.split('\n') if l.strip()]
				citations = []
				for line in lines:
					parts = line.split('|')
					if len(parts) >= 3:
						citations.append({'source': parts[0].strip(), 'page': parts[1].strip(), 'excerpt': '|'.join(parts[2:]).strip()})
					else:
						citations.append({'source': 'unknown', 'excerpt': line})
				parsed['citations'] = citations

			return parsed
		except Exception:
			return None

	def _import_module_optional(self, name: str) -> Optional[ModuleType]:
		"""Attempt to import a module by name and return it, or None on failure.

		Using importlib avoids static import-time checks that some static analyzers flag as missing.
		"""
		try:
			return importlib.import_module(name)
		except Exception:
			return None


# Instantiate a module-level helper for convenience when the app imports this module
_ENGINE: Optional[RAGEngine] = None


def get_engine(persist_directory: str = "./chroma_db") -> RAGEngine:
	global _ENGINE
	if _ENGINE is None:
		_ENGINE = RAGEngine(persist_directory=persist_directory)
	return _ENGINE

