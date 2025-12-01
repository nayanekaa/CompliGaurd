export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string; // The raw text for display or fallback
  timestamp: Date;
  // Structured data for RAG responses
  structuredResponse?: {
    answer: string;
    citations: Citation[];
    confidence: 'High' | 'Medium' | 'Low';
    suggestedAction?: string;
  };
  isThinking?: boolean;
}

export interface Citation {
  source: string;
  page?: string;
  excerpt: string;
  url?: string;
}

export interface PolicyDocument {
  id: string;
  title: string;
  category: 'HR' | 'InfoSec' | 'Legal' | 'Finance' | 'Ops';
  lastUpdated: string;
  status: 'Active' | 'Review' | 'Draft' | 'Archived';
  owner: string;
  coverage: number;
  version: string;
  nextReviewDate: string;
}

export interface Regulation {
  id: string;
  name: string; // e.g., GDPR, ISO 27001
  section: string;
  description: string;
  mappedControlId: string | null;
  status: 'Compliant' | 'Gap' | 'Partial';
  evidenceCount: number;
}

export enum AppView {
  DASHBOARD = 'DASHBOARD',
  CHAT = 'CHAT',
  POLICIES = 'POLICIES',
  MAPPING = 'MAPPING',
}
