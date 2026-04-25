"""Test script for retrieve_evidence_for_claim and judge_claim_with_evidence functions.

For local testing only - tests the evidence retrieval and claim judging pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import retrieve_evidence_for_claim, judge_claim_with_evidence, debug_ingest_pdf

# Load actual chunks from PDF
chunks = debug_ingest_pdf(r'C:\Users\dhrub\OneDrive\Desktop\SocratesAI\test.pdf')

claim = "Self-attention removes recurrence"
evidence = retrieve_evidence_for_claim(claim, chunks)

print("=" * 60)
print(f"CLAIM: {claim}")
print("=" * 60)
print()

print("EVIDENCE:")
for e in evidence:
    print(f"[Page {e['page_number']}] ({e['section_heading']})")
    print(e['evidence_text'][:200])
    print()

print("=" * 60)
result = judge_claim_with_evidence(claim, evidence)
print(f"VERDICT: {result['verdict']}")
print(f"SCORE: {result['score']}/10")
print("=" * 60)
