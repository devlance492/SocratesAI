"""Test script for voice input (Speech-to-Text).

For local testing only - tests microphone input and transcription.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import transcribe_voice_answer

print("=" * 60)
print("VOICE INPUT TEST")
print("=" * 60)
print()
print("Speak into your microphone when prompted.")
print("You have 15 seconds to complete your answer.")
print()

text = transcribe_voice_answer(timeout=5, phrase_time_limit=15)

print()
print("-" * 60)
if text:
    print(f"TRANSCRIBED TEXT: {text}")
else:
    print("No text was transcribed.")
print("-" * 60)
