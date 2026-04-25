"""
Socrates AI - FastAPI Backend Server

Provides REST API endpoints for the adaptive viva examination system.
This server handles PDF upload, question generation, answer evaluation,
and adaptive difficulty adjustment.

Usage:
    python -m src.server
    
The server runs on http://localhost:8000 by default.
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import examination functions from engine
from .engine import (
    debug_ingest_pdf,
    extract_knowledge_anchors,
    generate_examiner_question,
    extract_atomic_claims,
    retrieve_evidence_for_claim,
    judge_claim_with_evidence,
    aggregate_claim_judgments,
    generate_followup_question,
    select_cognitive_level,
)

# Load environment variables (graceful handling if dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def extract_knowledge_anchors_stub(chunks: list) -> list:
    """
    Stubbed version of knowledge anchor extraction.
    Extracts key concepts from section headings and text without LLM.
    """
    anchors = []
    seen = set()
    
    # Extract from section headings
    for chunk in chunks:
        heading = chunk.get("section_heading", "")
        if heading and heading not in seen and len(heading) > 5:
            # Convert heading to anchor concept
            anchor = f"the role of {heading.lower()} in the proposed approach"
            if len(anchor) < 100:
                anchors.append(anchor)
                seen.add(heading)
        
        if len(anchors) >= 5:
            break
    
    # If not enough from headings, extract from text keywords
    if len(anchors) < 5:
        keywords = ["method", "approach", "model", "algorithm", "technique", 
                    "framework", "system", "architecture", "mechanism"]
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            for kw in keywords:
                if kw in text and kw not in seen:
                    anchor = f"how the {kw} addresses the research problem"
                    anchors.append(anchor)
                    seen.add(kw)
                    if len(anchors) >= 5:
                        break
            if len(anchors) >= 5:
                break
    
    # Fallback anchors
    if not anchors:
        anchors = [
            "the main contribution of this research",
            "the methodology used in the paper",
            "the experimental design and evaluation",
            "the limitations discussed by the authors",
            "the implications for future work"
        ]
    
    return anchors[:5]


# FastAPI app
app = FastAPI(
    title="Socrates AI API",
    description="Adaptive Viva Voce Examiner for Research Papers",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# In-memory session storage (replace with Redis/DB for production)
sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response
class UploadResponse(BaseModel):
    session_id: str
    section_count: int
    message: str


class QuestionResponse(BaseModel):
    question: str
    knowledge_anchor: str
    question_number: int


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class EvaluationResponse(BaseModel):
    score: int
    verdict: str
    claims: List[Dict[str, Any]]
    followup_question: str
    cognitive_level: int
    question_number: int


class SessionStatusResponse(BaseModel):
    session_id: str
    document_loaded: bool
    section_count: int
    question_count: int
    current_score: Optional[int]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Socrates AI API"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document for examination.
    
    Returns a session ID and document metadata.
    """
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Create session
    session_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest the PDF
        chunks = debug_ingest_pdf(tmp_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to extract content from PDF")
        
        # Extract knowledge anchors for examination using OpenAI
        try:
            anchors = extract_knowledge_anchors(chunks)
        except Exception as llm_error:
            print(f"LLM anchor extraction failed: {llm_error}, using stub")
            anchors = extract_knowledge_anchors_stub(chunks)
        
        if not anchors:
            # Fallback: create anchors from section headings
            anchors = [c.get("section_heading", "the main concepts") for c in chunks[:5] if c.get("section_heading")]
            if not anchors:
                anchors = ["the methodology proposed in this paper"]
        
        # Store session data
        sessions[session_id] = {
            "file_name": file.filename,
            "chunks": chunks,
            "anchors": anchors,
            "current_anchor_index": 0,
            "question_count": 0,
            "scores": [],
            "tmp_path": tmp_path,
        }
        
        # Count unique sections
        section_set = set()
        for chunk in chunks:
            if chunk.get("section_heading"):
                section_set.add(chunk["section_heading"])
        section_count = len(section_set) if section_set else len(chunks)
        
        return UploadResponse(
            session_id=session_id,
            section_count=section_count,
            message="Document ingested successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/api/question/{session_id}", response_model=QuestionResponse)
async def get_question(session_id: str):
    """
    Get the next examination question for a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    anchors = session.get("anchors", [])
    
    if not anchors:
        raise HTTPException(status_code=400, detail="No knowledge anchors extracted from document")
    
    # Get current anchor
    anchor_index = session.get("current_anchor_index", 0)
    if anchor_index >= len(anchors):
        anchor_index = 0  # Cycle back to start
    
    anchor = anchors[anchor_index]
    
    # Generate question from anchor
    question = generate_examiner_question(anchor)
    
    # Update session
    session["question_count"] += 1
    session["current_question"] = question
    session["current_anchor"] = anchor
    
    return QuestionResponse(
        question=question,
        knowledge_anchor=anchor,
        question_number=session["question_count"]
    )


@app.post("/api/answer", response_model=EvaluationResponse)
async def submit_answer(request: AnswerRequest):
    """
    Submit an answer for evaluation.
    
    Returns score, verdict, and follow-up question.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    chunks = session.get("chunks", [])
    
    if not request.answer or not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    
    # Step 1: Extract atomic claims from answer
    claims = extract_atomic_claims(request.answer)
    
    if not claims:
        claims = [request.answer.strip()]  # Use whole answer as single claim
    
    # Step 2: Judge each claim against evidence
    claim_results = []
    for claim in claims:
        # Retrieve evidence for this claim
        evidence = retrieve_evidence_for_claim(claim, chunks)
        
        # Judge claim with evidence
        judgment = judge_claim_with_evidence(claim, evidence)
        
        claim_results.append({
            "claim": claim,
            "verdict": judgment.get("verdict", "INCORRECT"),
            "score": judgment.get("score", 1),
            "evidence": evidence,
            "reasoning": judgment.get("reasoning", "")
        })
    
    # Step 3: Aggregate judgments
    aggregation = aggregate_claim_judgments(claim_results)
    final_score = aggregation.get("final_score", 1)
    final_verdict = aggregation.get("final_verdict", "INCORRECT")
    
    # Step 4: Select cognitive level based on performance
    cognitive_level = select_cognitive_level(final_score)
    
    # Step 5: Generate follow-up question
    followup = generate_followup_question(claim_results, cognitive_level)
    
    # Update session
    session["scores"].append(final_score)
    session["current_anchor_index"] = (session.get("current_anchor_index", 0) + 1) % len(session.get("anchors", [1]))
    session["current_question"] = followup
    session["question_count"] += 1
    session["last_claim_results"] = claim_results
    
    # Convert verdict to frontend format
    verdict_map = {
        "CORRECT": "correct",
        "PARTIALLY_CORRECT": "partial",
        "INCORRECT": "incorrect"
    }
    
    return EvaluationResponse(
        score=final_score,
        verdict=verdict_map.get(final_verdict, "incorrect"),
        claims=claim_results,
        followup_question=followup,
        cognitive_level=cognitive_level,
        question_number=session["question_count"]
    )


@app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get the current status of an examination session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    scores = session.get("scores", [])
    
    return SessionStatusResponse(
        session_id=session_id,
        document_loaded=True,
        section_count=len(set(c.get("section_heading", "") for c in session.get("chunks", []))),
        question_count=session.get("question_count", 0),
        current_score=scores[-1] if scores else None
    )


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """
    End an examination session and cleanup resources.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions.pop(session_id)
    
    # Cleanup temp file
    tmp_path = session.get("tmp_path")
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return {"message": "Session ended", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
