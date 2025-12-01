/**
 * API Service
 * 
 * This file handles communication with the Python/FastAPI backend.
 * In the demo environment, we use the `geminiService.ts` to mock this behavior directly in the browser,
 * but in a production deployment, the calls below would be used.
 */

const API_BASE_URL = 'http://localhost:8000';

export interface BackendResponse {
  answer: string;
  citations: any[];
  confidence: string;
}

export const apiService = {
  /**
   * Health Check
   */
  checkHealth: async (): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      return res.ok;
    } catch (e) {
      return false; // Backend likely offline in demo mode
    }
  },

  /**
   * Send a question to the Python RAG backend
   */
  chat: async (question: string): Promise<BackendResponse> => {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    
    if (!res.ok) throw new Error('Backend request failed');
    return res.json();
  },

  /**
   * Upload a policy document for ingestion
   */
  uploadPolicy: async (file: File): Promise<void> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE_URL}/ingest`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error('Upload failed');
  }
};
