"""Test script for aggregate_claim_judgments function.

For local testing only - tests the scoring aggregation logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import aggregate_claim_judgments

claim_results = [
    {"verdict": "CORRECT", "score": 9},
    {"verdict": "PARTIALLY_CORRECT", "score": 6},
    {"verdict": "CORRECT", "score": 8},
]

result = aggregate_claim_judgments(claim_results)

print("Final Verdict:", result["final_verdict"])
print("Final Score:", result["final_score"])
