/**
 * Socrates AI API Client
 * 
 * Handles all communication with the FastAPI backend.
 * Set VITE_API_URL in your environment to point to the deployed backend.
 */

const API_ORIGIN = (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:8000';
const API_BASE_URL = `${API_ORIGIN}/api`;

export interface UploadResponse {
  session_id: string;
  section_count: number;
  message: string;
}

export interface QuestionResponse {
  question: string;
  knowledge_anchor: string;
  question_number: number;
}

export interface ClaimResult {
  claim: string;
  verdict: string;
  score: number;
  evidence: Array<{
    evidence_text: string;
    section_heading: string;
    page_number: number;
  }>;
  reasoning: string;
}

export interface EvaluationResponse {
  score: number;
  verdict: 'correct' | 'partial' | 'incorrect';
  claims: ClaimResult[];
  followup_question: string;
  cognitive_level: number;
  question_number: number;
}

export interface SessionStatus {
  session_id: string;
  document_loaded: boolean;
  section_count: number;
  question_count: number;
  current_score: number | null;
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, errorData.detail || 'Request failed');
  }
  return response.json();
}

/**
 * Upload a PDF document for examination.
 */
export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<UploadResponse>(response);
}

/**
 * Get the next examination question.
 */
export async function getQuestion(sessionId: string): Promise<QuestionResponse> {
  const response = await fetch(`${API_BASE_URL}/question/${sessionId}`);
  return handleResponse<QuestionResponse>(response);
}

/**
 * Submit an answer for evaluation.
 */
export async function submitAnswer(sessionId: string, answer: string): Promise<EvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      answer: answer,
    }),
  });

  return handleResponse<EvaluationResponse>(response);
}

/**
 * Get session status.
 */
export async function getSessionStatus(sessionId: string): Promise<SessionStatus> {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
  return handleResponse<SessionStatus>(response);
}

/**
 * End an examination session.
 */
export async function endSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to end session');
  }
}

/**
 * Check if the backend is available.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_ORIGIN}/`, {
      method: 'GET',
    });
    return response.ok;
  } catch {
    return false;
  }
}

export { ApiError };
