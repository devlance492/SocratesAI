"""
Test script for OpenAI API connection.
For local testing only - do not use in production.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set in environment or .env file")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.chatanywhere.tech"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Reply with only the word OK"}
    ],
    max_tokens=10
)

print("Model reply:", response.choices[0].message.content)
