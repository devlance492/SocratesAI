# Socrates AI

**An Adaptive Viva Voce Examiner for Research Papers**

Socrates AI is an intelligent examination system that conducts adaptive oral assessments (viva voce) based on research documents. The system dynamically adjusts question difficulty based on student performance, providing a personalized examination experience that mirrors human examiner behavior.

## Core Concept

Traditional automated assessment tools use fixed question sets. Socrates AI implements **adaptive difficulty adjustment** through:

- **Claim-level verification**: Student answers are decomposed into atomic claims, each verified against source evidence
- **Dynamic cognitive leveling**: Question difficulty scales from basic understanding (Level 3) to synthesis and critique (Level 6) based on response quality
- **Evidence-grounded feedback**: All evaluations cite specific passages from the source document

## How It Works

1. **Document Ingestion**: Upload a research paper (PDF). The system extracts and chunks the content with section awareness.

2. **Knowledge Anchor Extraction**: Key concepts and research contributions are identified as examination topics.

3. **Question Generation**: The examiner poses questions calibrated to the current cognitive level, using either LLM-powered generation or rule-based templates.

4. **Answer Evaluation**: Student responses are parsed into atomic claims. Each claim is matched against document evidence and scored.

5. **Adaptive Follow-up**: Based on claim verdicts, the system adjusts difficulty and generates targeted follow-up questions addressing gaps or advancing the discourse.

6. **Scoring Aggregation**: Individual claim scores are combined into a final assessment with detailed feedback.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your OpenAI API key to .env (optional - enables LLM questions)

# 3. Run the terminal demo
python run_examiner_dry_run.py
```

## Project Structure

```
SocratesAI/
├── src/                     # Core source code
│   ├── engine.py            # Examination engine (ingestion, scoring, questions)
│   └── server.py            # FastAPI backend server
├── tests/                   # Component test scripts
├── frontend/                # React web interface (optional)
├── run_examiner_dry_run.py  # Terminal demo entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md
```

## What Makes It Innovative

- **Adaptive difficulty**: Unlike static question banks, difficulty responds to demonstrated understanding
- **Evidence-grounded**: Every verdict cites specific document passages, ensuring transparency
- **Claim decomposition**: Partial credit through atomic claim analysis rather than holistic scoring
- **Cognitive level targeting**: Questions align with Bloom's taxonomy levels 3-6

## Requirements

- Python 3.10+
- OpenAI API key (optional, for LLM-powered question generation)
- PDF document for examination

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | Enables LLM-powered question generation |
| `SOCRATES_DEBUG` | Optional | Set to `1` for verbose debug output |

Without an API key, the system uses rule-based question templates.

## License

This project was developed for academic research and demonstration purposes.
