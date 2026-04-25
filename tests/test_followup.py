"""Test script for generate_followup_question function.

For local testing only - tests the adaptive follow-up question generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import generate_followup_question

# Test 1: Mixed results with INCORRECT claim
print("Test 1: INCORRECT claim present")
print("-" * 40)
claim_results_1 = [
    {"claim": "Self-attention removes recurrence", "verdict": "CORRECT"},
    {"claim": "The model uses CNNs for encoding", "verdict": "INCORRECT"},
    {"claim": "Training is parallelizable", "verdict": "PARTIALLY_CORRECT"},
]
question_1 = generate_followup_question(claim_results_1)
print(f"Question: {question_1}")
print()

# Test 2: Only PARTIALLY_CORRECT claims
print("Test 2: PARTIALLY_CORRECT claims")
print("-" * 40)
claim_results_2 = [
    {"claim": "Attention allows focusing on relevant parts", "verdict": "PARTIALLY_CORRECT"},
    {"claim": "The encoder processes input sequences", "verdict": "CORRECT"},
]
question_2 = generate_followup_question(claim_results_2)
print(f"Question: {question_2}")
print()

# Test 3: All CORRECT claims
print("Test 3: All CORRECT claims")
print("-" * 40)
claim_results_3 = [
    {"claim": "Self-attention removes recurrence", "verdict": "CORRECT"},
    {"claim": "Multi-head attention enables parallel computation", "verdict": "CORRECT"},
]
question_3 = generate_followup_question(claim_results_3)
print(f"Question: {question_3}")
