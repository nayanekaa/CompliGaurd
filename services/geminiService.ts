import { GoogleGenAI } from "@google/genai";
import { MOCK_KNOWLEDGE_BASE } from "../constants";
import { Citation } from "../types";

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

interface StructuredResult {
  text: string;
  structuredResponse: {
    answer: string;
    citations: Citation[];
    confidence: 'High' | 'Medium' | 'Low';
    suggestedAction: string;
  };
}

export const generateComplianceResponse = async (query: string): Promise<StructuredResult> => {
  if (!process.env.API_KEY) {
    throw new Error("API Key is missing. Please select an API Key.");
  }

  // System instruction for strict enterprise compliance behavior
  const systemInstruction = `
    You are 'CompliGuard', an auditable, enterprise-grade compliance assistant.
    
    CORE DIRECTIVE:
    Answer the user's question using ONLY the provided Context. 
    You must provide evidence for every claim.
    
    RESPONSE FORMAT:
    You must format your response using the following sections separated by exact delimiters:
    
    __ANSWER__
    (A concise, direct answer to the question, approx 2-4 sentences.)
    
    __CONFIDENCE__
    (One word: High, Medium, or Low. Base this on how directly the text answers the question.)
    
    __EVIDENCE__
    (List of citations. Each citation on a new line in this format: Source Name | Page Number | Exact Excerpt)
    
    __ACTION__
    (A short recommended action, e.g., "Submit request via HR Portal", "Review Policy", or "Escalate to InfoSec".)
    
    RULES:
    1. If the answer is not in the context, your Answer should be "I cannot find this information in the current policy documents." and Confidence should be "Low".
    2. Do not invent information.
    3. Ensure excerpts are verbatim from the text.
  `;

  const prompt = `
    CONTEXT:
    ${MOCK_KNOWLEDGE_BASE}

    USER QUESTION:
    ${query}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        systemInstruction: systemInstruction,
        temperature: 0.1, // Low temp for factual accuracy
      }
    });

    const fullText = response.text || "";
    return parseGeminiResponse(fullText);

  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
};

function parseGeminiResponse(text: string): StructuredResult {
  const result: StructuredResult = {
    text: text, // Fallback
    structuredResponse: {
      answer: '',
      citations: [],
      confidence: 'Medium',
      suggestedAction: ''
    }
  };

  try {
    const answerMatch = text.match(/__ANSWER__\s*([\s\S]*?)(?=__CONFIDENCE__|$)/);
    const confidenceMatch = text.match(/__CONFIDENCE__\s*([\s\S]*?)(?=__EVIDENCE__|$)/);
    const evidenceMatch = text.match(/__EVIDENCE__\s*([\s\S]*?)(?=__ACTION__|$)/);
    const actionMatch = text.match(/__ACTION__\s*([\s\S]*?)$/);

    if (answerMatch) result.structuredResponse.answer = answerMatch[1].trim();
    if (confidenceMatch) {
      const conf = confidenceMatch[1].trim();
      if (['High', 'Medium', 'Low'].includes(conf)) {
        result.structuredResponse.confidence = conf as 'High' | 'Medium' | 'Low';
      }
    }
    
    if (evidenceMatch) {
      const evidenceBlock = evidenceMatch[1].trim();
      const lines = evidenceBlock.split('\n').filter(l => l.trim().length > 0);
      result.structuredResponse.citations = lines.map(line => {
        // Expected format: Source | Page | Excerpt
        const parts = line.split('|');
        if (parts.length >= 3) {
          return {
            source: parts[0].trim(),
            page: parts[1].trim(),
            excerpt: parts.slice(2).join('|').trim() // Join back if excerpt contained |
          };
        }
        return { source: 'Unknown', excerpt: line, page: 'N/A' };
      });
    }

    if (actionMatch) result.structuredResponse.suggestedAction = actionMatch[1].trim();

    // Fallback if parsing failed completely but we have text
    if (!result.structuredResponse.answer && text) {
      result.structuredResponse.answer = text;
    }

  } catch (e) {
    console.error("Parsing error", e);
    result.structuredResponse.answer = text;
  }

  return result;
}
