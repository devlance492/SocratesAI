"""
Socrates AI - Terminal Demo

This script runs an interactive viva examination demonstration in the terminal.
It loads a PDF, generates adaptive questions, and evaluates student responses.

Usage:
    python run_examiner_dry_run.py
    
Requirements:
    - A test PDF file at the default path (or modify src/engine.py)
    - OPENAI_API_KEY set in .env or environment (for LLM-powered questions)
    
Without an API key, the system falls back to rule-based question templates.
"""

from src.engine import run_examiner_dry_run

if __name__ == "__main__":
    run_examiner_dry_run()
