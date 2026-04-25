"""
Socrates AI - Core Source Package

This package contains the main examination engine and API server.
"""

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
    run_examiner_dry_run,
)

__all__ = [
    "debug_ingest_pdf",
    "extract_knowledge_anchors",
    "generate_examiner_question",
    "extract_atomic_claims",
    "retrieve_evidence_for_claim",
    "judge_claim_with_evidence",
    "aggregate_claim_judgments",
    "generate_followup_question",
    "select_cognitive_level",
    "run_examiner_dry_run",
]
