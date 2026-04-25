"""
Socrates AI - Core Examination Engine

This module contains the core pipeline functions for the adaptive viva
examination system including:
- PDF ingestion and chunking
- Knowledge anchor extraction
- Examiner question generation (LLM-powered)
- Atomic claim extraction from student answers
- Evidence retrieval using semantic search
- Claim verification and scoring
- Adaptive difficulty adjustment

DEBUG_MODE controls verbose output (set via environment variable).
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for production

# Debug mode: Set SOCRATES_DEBUG=1 in environment for verbose output
DEBUG_MODE = os.getenv("SOCRATES_DEBUG", "0") == "1"


def _debug_print(*args, **kwargs):
    """Print only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)


# Approximate characters per token (for English text)
CHARS_PER_TOKEN = 4
MIN_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 600
TARGET_CHUNK_TOKENS = 500


# Patterns to IGNORE as section headers
IGNORE_PATTERNS = [
    # Page headers/footers with journal info
    re.compile(r"^IEEE\s+", re.IGNORECASE),
    re.compile(r"^\d+\s+IEEE\s+", re.IGNORECASE),
    re.compile(r"VOL\.\s*\d+", re.IGNORECASE),
    re.compile(r"^[A-Z]+\s+et\s+al\.:", re.IGNORECASE),  # "LI et al.:"
    
    # Copyright/license text
    re.compile(r"authorized\s+licensed\s+use", re.IGNORECASE),
    re.compile(r"downloaded\s+on", re.IGNORECASE),
    re.compile(r"restrictions\s+apply", re.IGNORECASE),
    re.compile(r"©|copyright", re.IGNORECASE),
    
    # Figure/Table captions
    re.compile(r"^fig\.?\s*\d+", re.IGNORECASE),
    re.compile(r"^figure\s*\d+", re.IGNORECASE),
    re.compile(r"^table\s*\d+", re.IGNORECASE),
    re.compile(r"^tab\.?\s*\d+", re.IGNORECASE),
    
    # Email patterns
    re.compile(r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    
    # Single letters or symbols (like dropped caps "P" for "Printed")
    re.compile(r"^\*{0,2}[A-Z]\*{0,2}$"),
    
    # URLs
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"orcid\.org", re.IGNORECASE),
    
    # Manuscript/submission info
    re.compile(r"^manuscript\s+received", re.IGNORECASE),
    re.compile(r"^date\s+of\s+publication", re.IGNORECASE),
    
    # Abstract (handled separately)
    re.compile(r"^\*{0,2}abstract\*{0,2}$", re.IGNORECASE),
]

# Valid section header patterns for academic papers
SECTION_PATTERNS = [
    # Pattern: "1 Introduction" or "1. Introduction"
    re.compile(r"^(\d+)\.?\s+([A-Z][A-Za-z\s\-&]+)$"),
    
    # Pattern: "3.1 Encoder" or "3.1. Encoder"
    re.compile(r"^(\d+\.\d+)\.?\s+([A-Z][A-Za-z\s\-&]+)$"),
    
    # Pattern: "3.1.2 Sub-subsection"
    re.compile(r"^(\d+\.\d+\.\d+)\.?\s+([A-Z][A-Za-z\s\-&]+)$"),
    
    # Pattern: "**1** **Introduction**" or "**1** Introduction"
    re.compile(r"^\*{2}(\d+(?:\.\d+)*)\*{2}\s*\*{0,2}([A-Z][A-Za-z\s\-&\*]+?)\*{0,2}$"),
    
    # Pattern: "**3.2** **Attention**"
    re.compile(r"^\*{2}(\d+\.\d+)\*{2}\s*\*{0,2}([A-Z][A-Za-z\s\-&\*]+?)\*{0,2}$"),
    
    # Pattern: Roman numerals "I. INTRODUCTION" or "II. BACKGROUND"
    re.compile(r"^([IVX]+)\.?\s+([A-Z][A-Za-z\s\-&]+)$"),
    
    # Pattern: "A. Subsection" (letter-based subsections)
    re.compile(r"^([A-Z])\.?\s+([A-Z][A-Za-z\s\-&]+)$"),
]

# Markdown header pattern
MARKDOWN_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _clean_section_title(title: str) -> str:
    """Remove markdown formatting from section title."""
    # Remove bold markers
    title = title.replace("**", "").replace("*", "")
    # Clean up extra whitespace
    title = " ".join(title.split())
    return title.strip()


def _is_valid_section_header(text: str) -> bool:
    """Check if text should be ignored as a section header."""
    text = text.strip()
    
    # Too short or too long
    if len(text) < 3 or len(text) > 100:
        return False
    
    # Check against ignore patterns
    for pattern in IGNORE_PATTERNS:
        if pattern.search(text):
            return False
    
    return True


def _extract_section_header(line: str) -> Optional[str]:
    """
    Try to extract a section header from a line.
    Returns the formatted section string or None.
    """
    line = line.strip()
    
    if not line or not _is_valid_section_header(line):
        return None
    
    # Try each section pattern
    for pattern in SECTION_PATTERNS:
        match = pattern.match(line)
        if match:
            number = match.group(1)
            title = _clean_section_title(match.group(2))
            
            # Additional validation: title should be meaningful
            if len(title) < 2:
                continue
            if title.upper() == title and len(title) > 20:
                # Likely a header line, not a section
                continue
                
            return f"{number} {title}"
    
    return None


def _find_sections_in_text(markdown_text: str) -> List[Tuple[int, str]]:
    """
    Find all section headers in text and their positions.
    Returns list of (position, section_heading).
    """
    sections: List[Tuple[int, str]] = []
    
    # First, check for markdown headers
    for match in MARKDOWN_HEADER_PATTERN.finditer(markdown_text):
        header_text = match.group(2).strip()
        # Try to extract as academic section
        section = _extract_section_header(header_text)
        if section:
            sections.append((match.start(), section))
        elif _is_valid_section_header(header_text) and len(header_text) > 5:
            # Use the markdown header as-is if it looks valid
            sections.append((match.start(), _clean_section_title(header_text)))
    
    # Then, scan for standalone section headers (line by line)
    lines = markdown_text.split('\n')
    current_pos = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip if this position was already captured as markdown header
        already_found = any(abs(pos - current_pos) < len(line) + 5 for pos, _ in sections)
        
        if not already_found:
            section = _extract_section_header(line_stripped)
            if section:
                sections.append((current_pos, section))
        
        current_pos += len(line) + 1  # +1 for newline
    
    # Sort by position
    sections.sort(key=lambda x: x[0])
    
    return sections


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text."""
    return len(text) // CHARS_PER_TOKEN


def _chunk_section_text(text: str) -> List[str]:
    """
    Chunk text within a single section (400-600 tokens).
    
    Args:
        text: Section text to chunk.
        
    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    estimated_tokens = _estimate_tokens(text)

    # If text is small enough, return as single chunk
    if estimated_tokens <= MAX_CHUNK_TOKENS:
        return [text.strip()]

    chunks: List[str] = []
    
    # Split by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    current_chunk: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = _estimate_tokens(paragraph)

        # If single paragraph exceeds max, split by sentences
        if para_tokens > MAX_CHUNK_TOKENS:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph by sentences
            sentence_chunks = _split_by_sentences(paragraph)
            chunks.extend(sentence_chunks)
            continue

        # Check if adding this paragraph exceeds target
        if current_tokens + para_tokens > TARGET_CHUNK_TOKENS:
            # Save current chunk if it meets minimum size
            if current_tokens >= MIN_CHUNK_TOKENS:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

        current_chunk.append(paragraph)
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk:
        final_chunk = "\n\n".join(current_chunk)
        if final_chunk.strip():
            # If last chunk is too small, merge with previous if possible
            if _estimate_tokens(final_chunk) < MIN_CHUNK_TOKENS and chunks:
                prev_chunk = chunks.pop()
                combined = prev_chunk + "\n\n" + final_chunk
                # Only merge if combined doesn't exceed max
                if _estimate_tokens(combined) <= MAX_CHUNK_TOKENS:
                    chunks.append(combined)
                else:
                    chunks.append(prev_chunk)
                    chunks.append(final_chunk)
            else:
                chunks.append(final_chunk)

    return chunks


def _split_by_sentences(paragraph: str) -> List[str]:
    """Split a large paragraph into sentence-based chunks."""
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(paragraph)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_tokens = _estimate_tokens(sentence)

        if current_tokens + sent_tokens > TARGET_CHUNK_TOKENS and current_chunk:
            if current_tokens >= MIN_CHUNK_TOKENS:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def debug_ingest_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Debug function to test PDF ingestion without embedding or storage.
    Chunks text into 400-600 token pieces, never crossing section boundaries.
    Detects academic paper section headers (e.g., "1 Introduction", "3.2 Attention").

    Args:
        pdf_path: Path to a locally saved PDF file.

    Returns:
        List of dictionaries with keys: text, page_number, section_heading
    """
    import pymupdf4llm

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract markdown with page chunks
    pages_data = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    # Process all pages and track sections across page boundaries
    all_sections: List[Tuple[str | None, str, int]] = []  # (heading, text, page_number)
    current_section: Optional[str] = None

    for page_data in pages_data:
        page_number = page_data.get("metadata", {}).get("page", 1)
        markdown_text = page_data.get("text", "")

        if not markdown_text.strip():
            continue

        # Find all section headers in this page
        section_positions = _find_sections_in_text(markdown_text)

        if not section_positions:
            # No new sections on this page, entire page belongs to current section
            all_sections.append((current_section, markdown_text.strip(), page_number))
            continue

        # Process text between sections
        last_end = 0
        
        for pos, section_heading in section_positions:
            # Text before this section header
            if pos > last_end:
                text_before = markdown_text[last_end:pos].strip()
                if text_before and len(text_before) > 20:
                    all_sections.append((current_section, text_before, page_number))
            
            # Update current section
            current_section = section_heading
            
            # Find where section header ends (next line)
            header_end = markdown_text.find('\n', pos)
            if header_end == -1:
                header_end = len(markdown_text)
            last_end = header_end

        # Remaining text after last section header
        remaining = markdown_text[last_end:].strip()
        if remaining and len(remaining) > 20:
            all_sections.append((current_section, remaining, page_number))

    # Chunk each section independently
    results: List[Dict[str, Any]] = []

    for section_heading, section_text, page_number in all_sections:
        text_chunks = _chunk_section_text(section_text)

        for chunk_text in text_chunks:
            results.append({
                "text": chunk_text,
                "page_number": page_number,
                "section_heading": section_heading,
            })

    return results


def print_debug_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Print chunks for manual human inspection (only in DEBUG_MODE).

    Args:
        chunks: List of chunk dictionaries from debug_ingest_pdf.
    """
    if not DEBUG_MODE:
        return
        
    total = len(chunks)
    display_count = min(10, total)

    _debug_print()
    _debug_print("=" * 70)
    _debug_print(f"TOTAL CHUNKS: {total}")
    _debug_print(f"DISPLAYING: First {display_count} chunks")
    _debug_print("=" * 70)

    for i, chunk in enumerate(chunks[:10]):
        _debug_print()
        _debug_print(f"{'─' * 70}")
        _debug_print(f"CHUNK {i + 1} of {total}")
        _debug_print(f"{'─' * 70}")
        _debug_print(f"  Section Heading : {chunk['section_heading'] or '(None)'}")
        _debug_print(f"  Page Number     : {chunk['page_number']}")
        _debug_print(f"{'─' * 70}")
        
        text = chunk["text"]
        preview = text[:500] + "..." if len(text) > 500 else text
        _debug_print(preview)
        _debug_print()

    if total > 10:
        _debug_print("=" * 70)
        _debug_print(f"... {total - 10} more chunks not displayed")
        _debug_print("=" * 70)


def check_section_integrity(chunks: List[Dict[str, Any]]) -> None:
    """
    Check section integrity and print diagnostics (only in DEBUG_MODE).

    Args:
        chunks: List of chunk dictionaries from debug_ingest_pdf.
    """
    if not DEBUG_MODE:
        return
        
    total = len(chunks)

    if total == 0:
        _debug_print("[WARN] No chunks to analyze.")
        return

    # Count chunks with missing/empty section_heading
    missing_section_count = 0
    section_counts: Dict[str, int] = {}

    for chunk in chunks:
        heading = chunk.get("section_heading")
        
        if not heading or not heading.strip():
            missing_section_count += 1
            section_key = "(No Section)"
        else:
            section_key = heading.strip()

        section_counts[section_key] = section_counts.get(section_key, 0) + 1

    # Calculate percentages
    missing_pct = (missing_section_count / total) * 100

    _debug_print()
    _debug_print("=" * 70)
    _debug_print("SECTION INTEGRITY CHECK")
    _debug_print("=" * 70)
    _debug_print(f"Total chunks: {total}")
    _debug_print(f"Chunks without section heading: {missing_section_count} ({missing_pct:.1f}%)")
    _debug_print()

    # Print section distribution
    _debug_print("Section distribution:")
    _debug_print("-" * 50)
    
    sorted_sections = sorted(section_counts.items(), key=lambda x: -x[1])
    for section, count in sorted_sections:
        pct = (count / total) * 100
        bar = "█" * int(pct // 2)
        _debug_print(f"  {section[:40]:<40} : {count:>4} ({pct:>5.1f}%) {bar}")

    _debug_print()
    _debug_print("-" * 70)
    _debug_print("DIAGNOSTICS:")
    _debug_print("-" * 70)

    warnings_found = False

    # Warning: more than 30% chunks have no section
    if missing_pct > 30:
        _debug_print(f"[WARN] {missing_pct:.1f}% of chunks have no section heading (threshold: 30%)")
        _debug_print("       This may indicate poor document structure or header detection issues.")
        warnings_found = True

    # Warning: one section dominates more than 60%
    for section, count in sorted_sections:
        pct = (count / total) * 100
        if pct > 60:
            _debug_print(f"[WARN] Section '{section[:50]}' contains {pct:.1f}% of all chunks (threshold: 60%)")
            _debug_print("       This may indicate unbalanced content or missed section headers.")
            warnings_found = True

    if not warnings_found:
        _debug_print("[OK] No issues detected.")

    _debug_print("=" * 70)


# ==============================================================================
# OPENAI LLM CONFIGURATION
# ==============================================================================

# Feature flag: Set to True to use LLM for question generation
# Set to False to use rule-based templates only
USE_LLM_QUESTIONS = True

EXAMINER_PERSONA = """You are a strict university viva examiner. You do not help, hint, or explain.

Language constraints:
- Use formal academic tone only
- Avoid polite or encouraging phrases
- Avoid hedging words like: may, might, seems, possibly
- Do not teach or explain concepts"""


# Cognitive level descriptions for prompts
COGNITIVE_LEVEL_PROMPTS = {
    3: "Clarify misconceptions - ask a basic understanding question",
    4: "Analyze and compare - ask the student to analyze or compare concepts",
    5: "Critique assumptions - ask about limitations, assumptions, or weaknesses",
    6: "Propose alternatives - ask to generalize, extend, or propose modifications",
}


def _get_openai_client():
    """Initialize OpenAI client with API key from environment."""
    import os
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.tech"
    )
    return client


def call_llm_question(prompt: str, max_tokens: int = 60) -> str:
    """
    Call LLM to generate an examiner question.
    
    Centralized LLM call for question generation only.
    Uses strict parameters for academic examiner tone.
    
    Args:
        prompt: The question generation prompt
        max_tokens: Maximum tokens for response (default 60)
    
    Returns:
        Generated question string, or empty string on failure
    """
    try:
        client = _get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": EXAMINER_PERSONA},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        
        question = response.choices[0].message.content.strip()
        
        # Ensure it ends with question mark
        if question and not question.endswith('?'):
            question += '?'
        
        return question
        
    except Exception as e:
        print(f"[LLM ERROR] Question generation failed: {e}")
        return ""  # Return empty to trigger fallback


def _call_llm(prompt: str, system_context: str = "") -> str:
    """
    Call OpenAI with strict examiner parameters.
    
    Args:
        prompt: The user prompt
        system_context: Additional context to prepend (after persona)
    
    Returns:
        Truncated response (max 6 lines)
    """
    client = _get_openai_client()
    
    # Combine persona with system context
    system_message = EXAMINER_PERSONA
    if system_context:
        system_message += f"\n\n{system_context}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        # Try fallback model
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as fallback_error:
            raise RuntimeError(f"OpenAI API call failed: {e}. Fallback also failed: {fallback_error}")
    
    # SAFETY GUARD: Truncate to max 6 lines
    lines = [line for line in response_text.split('\n') if line.strip()]
    truncated = '\n'.join(lines[:6])
    
    return truncated


def extract_knowledge_anchors(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract 5 Knowledge Anchors from research paper chunks using Gemini.
    
    Knowledge Anchors are deep conceptual ideas suitable for graduate-level
    viva questions. They focus on methodology, motivation, design choices,
    trade-offs, and implications rather than surface-level facts.

    Args:
        chunks: List of chunk dictionaries with keys: text, page_number, section_heading

    Returns:
        List of exactly 5 knowledge anchor strings.
    """
    # Prepare context from chunks - group by section for better structure
    sections_text: Dict[str, List[str]] = {}
    for chunk in chunks:
        section = chunk.get("section_heading") or "Introduction"
        if section not in sections_text:
            sections_text[section] = []
        sections_text[section].append(chunk["text"][:1500])  # Limit per chunk
    
    # Build structured paper summary
    paper_content = ""
    for section, texts in sections_text.items():
        paper_content += f"\n\n=== {section} ===\n"
        paper_content += "\n".join(texts[:3])  # Max 3 chunks per section
    
    # Truncate if too long
    if len(paper_content) > 30000:
        paper_content = paper_content[:30000]

    system_context = """Task: Extract knowledge anchors for viva examination.

Content rules:
- Anchors must be deep conceptual ideas (why/how/trade-offs)
- Do NOT restate section titles
- Do NOT summarize the paper
- Focus on methodology, design choices, limitations, implications"""

    prompt = f"""Analyze this research paper. Output EXACTLY 5 knowledge anchors.

FORMAT RULES (strict):
- Output ONLY a numbered list (1-5)
- Each item must be ONE sentence
- Maximum 25 words per item
- No markdown, no headings, no explanations

PAPER CONTENT:
{paper_content}

Output:"""

    response = _call_llm(prompt, system_context)
    
    # Parse the response - extract numbered items
    anchors: List[str] = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match lines starting with 1. 2. 3. 4. 5.
        for i in range(1, 6):
            prefixes = [f"{i}.", f"{i})", f"{i}:"]
            for prefix in prefixes:
                if line.startswith(prefix):
                    anchor = line[len(prefix):].strip()
                    if anchor and len(anchors) < 5:
                        # Enforce max 25 words
                        words = anchor.split()
                        if len(words) > 25:
                            anchor = ' '.join(words[:25])
                        anchors.append(anchor)
                    break
    
    return anchors[:5]


def generate_examiner_question(knowledge_anchor: str, cognitive_level: int = 4) -> str:
    """
    Generate a strict viva-style question from a knowledge anchor.
    
    If USE_LLM_QUESTIONS is True, uses OpenAI gpt-4.1-mini.
    Otherwise, uses deterministic rule-based templates.
    
    Args:
        knowledge_anchor: A knowledge anchor string describing a deep concept
        cognitive_level: Bloom's taxonomy level (3-6) for question difficulty
    
    Returns:
        A single viva examination question
    """
    # Try LLM generation if enabled
    if USE_LLM_QUESTIONS:
        level_desc = COGNITIVE_LEVEL_PROMPTS.get(cognitive_level, COGNITIVE_LEVEL_PROMPTS[4])
        
        prompt = f"""Generate ONE viva examination question about this concept:
"{knowledge_anchor}"

Cognitive level: {level_desc}

Rules:
- One sentence only
- No hints or answers
- No praise or encouragement
- Formal academic tone
- Must probe understanding, not recall
- Reference the concept directly

Output the question only, no preamble."""

        llm_question = call_llm_question(prompt, max_tokens=60)
        if llm_question:
            return llm_question
        # Fall through to rule-based on LLM failure
    
    # Rule-based fallback (original logic)
    anchor_clean = knowledge_anchor.strip().rstrip('.')
    anchor_lower = anchor_clean.lower()
    
    # Remove leading question words to get neutral concept phrase
    question_prefixes = [
        ("why ", 4),
        ("how ", 4),
        ("what ", 5),
        ("when ", 5),
        ("where ", 6),
        ("which ", 6),
    ]
    
    concept = anchor_clean
    for prefix, length in question_prefixes:
        if anchor_lower.startswith(prefix):
            concept = anchor_clean[length:]  # Remove prefix, preserve original case
            break
    
    # Remove common article prefixes for cleaner templates
    concept_lower = concept.lower()
    for prefix in ["the ", "this ", "a ", "an "]:
        if concept_lower.startswith(prefix):
            concept = concept[len(prefix):]
            break
    
    # Truncate if too long for template
    if len(concept) > 100:
        concept = concept[:100].rsplit(' ', 1)[0]
    
    # Question templates - WHY/HOW focused, no definitions
    templates = [
        f"Why is {concept} significant to the methodology proposed in this paper?",
        f"How does {concept} influence the design choices made by the authors?",
        f"What trade-offs arise from {concept} as discussed in the paper?",
        f"Why did the authors prioritize {concept} over alternative approaches?",
        f"How would the proposed method be affected if {concept} were modified?",
        f"What assumptions underlie {concept} and how might they limit generalizability?",
        f"How does {concept} address limitations present in prior work?",
        f"Why should {concept} be considered a contribution rather than an incremental change?",
        f"What evidence in the paper supports the effectiveness of {concept}?",
        f"How does {concept} interact with other components of the proposed system?",
    ]
    
    # Select template based on anchor content for consistency
    # Use hash for deterministic selection
    template_index = hash(knowledge_anchor) % len(templates)
    question = templates[template_index]
    
    return question


def generate_examiner_question_llm(anchor: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a strict examiner question using Gemini LLM.
    
    This is the LLM-based version. Use generate_examiner_question() for
    the template-based stubbed version.
    
    Args:
        anchor: A knowledge anchor string
        context_chunks: Relevant chunks for context
    
    Returns:
        A single viva examination question
    """
    # Build context from chunks
    context = "\n".join([c["text"][:500] for c in context_chunks[:3]])
    
    system_context = """Task: Generate a viva examination question.

Question rules:
- Bloom taxonomy level 3-6 ONLY (apply, analyze, evaluate, create)
- NEVER ask "what is" or definition questions
- Ask ONE question only
- No hints, no follow-up explanations
- Question must probe deep understanding"""

    prompt = f"""Knowledge anchor: {anchor}

Paper context:
{context}

Generate ONE examination question. Output the question only, no preamble."""

    response = _call_llm(prompt, system_context)
    
    # Extract just the question (first non-empty line)
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        question = lines[0]
        # Ensure it ends with ?
        if not question.endswith('?'):
            question += '?'
        return question
    
    return "Explain the core methodological contribution of this work."


def extract_atomic_claims(answer_text: str) -> List[str]:
    """
    Extract atomic factual claims from an answer text.
    
    Uses rule-based heuristics. Will be replaced with LLM version later.
    
    Args:
        answer_text: The student's answer to a viva question
    
    Returns:
        List of atomic factual claims (strings)
    """
    if not answer_text or not answer_text.strip():
        return []
    
    # Step 1: Remove leading filler/opinion phrases
    filler_prefixes = [
        "i think that ", "i think ", "i believe that ", "i believe ",
        "in my opinion, ", "in my opinion ", "basically, ", "basically ",
        "we can say that ", "we can say ", "it seems that ", "it seems like ",
        "i would say that ", "i would say ", "from my understanding, ",
        "from my understanding ", "as far as i know, ", "as far as i know ",
        "i guess ", "i suppose ", "essentially, ", "essentially ",
        "well, ", "so, ", "um, ", "uh, ", "like, ",
    ]
    
    text = answer_text.strip()
    text_lower = text.lower()
    
    for filler in filler_prefixes:
        if text_lower.startswith(filler):
            text = text[len(filler):]
            text_lower = text.lower()
    
    # Capitalize first letter after removing filler
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Step 2: Split into sentences first (on . ! ?)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)
    
    claims: List[str] = []
    
    for sentence in sentences:
        sentence = sentence.strip().rstrip('.')
        if not sentence or len(sentence) < 10:
            continue
        
        # Step 3: Process compound sentences with causal connectors
        extracted = _extract_claims_from_sentence(sentence)
        claims.extend(extracted)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_claims = []
    for claim in claims:
        claim_normalized = claim.lower()
        if claim_normalized not in seen:
            seen.add(claim_normalized)
            unique_claims.append(claim)
    
    return unique_claims


def _extract_claims_from_sentence(sentence: str) -> List[str]:
    """
    Extract atomic claims from a single sentence by splitting on causal connectors.
    Reconstructs fragments into complete standalone claims.
    """
    claims = []
    
    # Pattern: "X, which Y and Z" -> ["X", "This Y", "This Z"]
    # Pattern: "X because Y" -> ["X", "Y"]
    
    # First, handle ", which" splits
    if ", which " in sentence.lower():
        parts = re.split(r',\s+which\s+', sentence, flags=re.IGNORECASE)
        if len(parts) >= 2:
            # First part is a complete claim
            first_claim = parts[0].strip()
            claims.append(_capitalize(first_claim))
            
            # Extract the action/result from first claim to use as subject
            # Heuristic: get the predicate (verb + object) from first claim
            implied_subject = _extract_implied_subject(first_claim)
            
            # Process the "which" clause(s)
            for which_part in parts[1:]:
                # Split on " and " within the which-clause
                and_parts = re.split(r'\s+and\s+', which_part, flags=re.IGNORECASE)
                
                for i, and_part in enumerate(and_parts):
                    and_part = and_part.strip().rstrip('.,;:')
                    if len(and_part) < 5:
                        continue
                    
                    # Reconstruct as complete sentence
                    if i == 0:
                        # First part after "which": use implied subject
                        claim = f"{implied_subject} {and_part}"
                    else:
                        # Subsequent "and" parts: extract new subject from previous
                        prev_subject = _extract_result_subject(and_parts[i-1])
                        claim = f"{prev_subject} {and_part}"
                    
                    claims.append(_capitalize(claim))
            
            return claims
    
    # Handle ", that " splits (similar to which)
    if ", that " in sentence.lower():
        parts = re.split(r',\s+that\s+', sentence, flags=re.IGNORECASE)
        if len(parts) >= 2:
            claims.append(_capitalize(parts[0].strip()))
            implied_subject = _extract_implied_subject(parts[0])
            for that_part in parts[1:]:
                that_part = that_part.strip().rstrip('.,;:')
                if len(that_part) >= 5:
                    claims.append(_capitalize(f"{implied_subject} {that_part}"))
            return claims
    
    # Handle " because " splits
    if " because " in sentence.lower():
        parts = re.split(r'\s+because\s+', sentence, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip().rstrip('.,;:')
            if len(part) >= 10:
                claims.append(_capitalize(part))
        return claims
    
    # Handle " therefore " / " thus " / " hence " splits
    for connector in [" therefore ", " thus ", " hence ", " so "]:
        if connector in sentence.lower():
            parts = re.split(r'\s+(?:therefore|thus|hence|so)\s+', sentence, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip().rstrip('.,;:')
                if len(part) >= 10:
                    claims.append(_capitalize(part))
            return claims
    
    # Handle simple " and " in main clause (not after which/that)
    if " and " in sentence.lower():
        # Check if it's joining predicates with same subject
        and_parts = re.split(r'\s+and\s+', sentence, flags=re.IGNORECASE)
        if len(and_parts) >= 2:
            # First part should have subject
            first_part = and_parts[0].strip()
            claims.append(_capitalize(first_part))
            
            # Extract subject from first part
            subject = _extract_subject(first_part)
            
            for part in and_parts[1:]:
                part = part.strip().rstrip('.,;:')
                if len(part) < 5:
                    continue
                # Check if part already has a subject (contains a noun before verb)
                if _has_subject(part):
                    claims.append(_capitalize(part))
                else:
                    # Add subject from first clause
                    claims.append(_capitalize(f"{subject} {part}"))
            return claims
    
    # No connectors found, return as single claim
    sentence = sentence.strip().rstrip('.,;:')
    if len(sentence) >= 10:
        claims.append(_capitalize(sentence))
    
    return claims


def _capitalize(text: str) -> str:
    """Capitalize first letter of text."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def _extract_implied_subject(clause: str) -> str:
    """
    Extract an implied subject from a clause for use in dependent clauses.
    Heuristic: Convert the action to a gerund form.
    
    "Self-attention removes recurrence" -> "Removing recurrence"
    "The model uses attention" -> "Using attention"
    """
    clause = clause.strip()
    words = clause.split()
    
    if len(words) < 2:
        return "This"
    
    # Find the verb (heuristic: first word after subject that could be a verb)
    # Common verb patterns
    verb_idx = -1
    for i, word in enumerate(words):
        word_lower = word.lower()
        # Skip common subjects
        if i == 0 and word_lower in ["the", "a", "an", "this", "that", "it"]:
            continue
        # Look for verbs (simple heuristic: 3+ letter words after first word)
        if i > 0 and len(word) >= 3 and word_lower not in ["the", "a", "an", "and", "or"]:
            # Check if it could be a verb (ends in s, es, ed, or common verbs)
            if (word_lower.endswith('s') or word_lower.endswith('es') or 
                word_lower.endswith('ed') or word_lower in 
                ["is", "are", "was", "were", "has", "have", "had", "does", "do",
                 "uses", "allows", "enables", "removes", "reduces", "provides",
                 "creates", "makes", "gives", "takes", "shows", "means"]):
                verb_idx = i
                break
    
    if verb_idx > 0 and verb_idx < len(words):
        verb = words[verb_idx].lower()
        rest = " ".join(words[verb_idx + 1:])
        
        # Convert to gerund
        gerund = _to_gerund(verb)
        
        if rest:
            return f"{gerund} {rest}"
        return gerund
    
    return "This"


def _to_gerund(verb: str) -> str:
    """Convert a verb to its gerund form (adding -ing)."""
    verb = verb.lower().rstrip('s')
    
    # Handle common irregular cases
    if verb.endswith('e') and not verb.endswith('ee'):
        return verb[:-1] + "ing"
    if verb.endswith('ie'):
        return verb[:-2] + "ying"
    if len(verb) >= 3 and verb[-1] not in 'aeiou' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
        # CVC pattern - double final consonant
        return verb + verb[-1] + "ing"
    
    return verb + "ing"


def _extract_result_subject(clause: str) -> str:
    """
    Extract the result/object of a clause to use as subject for next clause.
    
    "allows parallel computation" -> "Parallel computation"
    """
    clause = clause.strip()
    words = clause.split()
    
    if len(words) < 2:
        return "This"
    
    # Take the last noun phrase (heuristic: last 2-3 words)
    if len(words) >= 2:
        result = " ".join(words[-2:])
        return _capitalize(result)
    
    return "This"


def _extract_subject(clause: str) -> str:
    """
    Extract the subject from a clause.
    Heuristic: Take words before the first verb.
    """
    clause = clause.strip()
    words = clause.split()
    
    if len(words) < 2:
        return "It"
    
    # Find first verb
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in ["is", "are", "was", "were", "has", "have", "had",
                          "does", "do", "uses", "allows", "enables", "removes",
                          "reduces", "provides", "creates", "makes", "can", "will",
                          "would", "could", "should", "may", "might"]:
            if i > 0:
                return " ".join(words[:i])
            break
    
    # Default: first word or two
    return words[0] if len(words) == 1 else " ".join(words[:2])


def _has_subject(clause: str) -> bool:
    """
    Check if a clause appears to have its own subject.
    Heuristic: Starts with a noun/pronoun or determiner + noun.
    """
    clause = clause.strip().lower()
    words = clause.split()
    
    if not words:
        return False
    
    first = words[0]
    
    # Starts with determiner
    if first in ["the", "a", "an", "this", "that", "these", "those", "my", "your", "its"]:
        return True
    
    # Starts with pronoun
    if first in ["it", "they", "he", "she", "we", "i", "you"]:
        return True
    
    # Starts with capitalized word (likely proper noun or sentence start)
    if words[0] and words[0][0].isupper():
        return True
    
    return False


def retrieve_evidence_for_claim(
    claim: str,
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence snippets from chunks that support or refute a claim.
    
    Uses keyword overlap scoring with section preference weighting.
    This is a temporary Reverse-RAG stub - will be replaced with embeddings later.
    
    Args:
        claim: One atomic factual claim (string)
        chunks: List of dicts with keys: text, section_heading, page_number
    
    Returns:
        List of evidence dicts (top 2-3), each containing:
        - evidence_text: Relevant text snippet
        - section_heading: Section where evidence was found
        - page_number: Page number of the evidence
    """
    if not claim or not chunks:
        return []
    
    # Step 1: Extract keywords from claim
    claim_keywords = _extract_keywords(claim)
    
    if not claim_keywords:
        return []
    
    # Step 2: Score each chunk
    scored_chunks: List[Tuple[float, int, Dict[str, Any]]] = []
    
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        section = chunk.get("section_heading", "")
        page = chunk.get("page_number", 0)
        
        if not text:
            continue
        
        # Calculate keyword overlap score
        overlap_score = _calculate_keyword_overlap(claim_keywords, text)
        
        if overlap_score == 0:
            continue
        
        # Apply section preference boost
        section_boost = _get_section_boost(section)
        final_score = overlap_score * section_boost
        
        scored_chunks.append((final_score, idx, {
            "text": text,
            "section_heading": section,
            "page_number": page,
        }))
    
    # Step 3: Sort by score (descending), then by index (for determinism)
    scored_chunks.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 4: Return top 2-3 chunks
    top_k = min(3, len(scored_chunks))
    
    # Only include chunks with meaningful scores
    results: List[Dict[str, Any]] = []
    for score, idx, chunk_data in scored_chunks[:top_k]:
        if score < 0.1:  # Minimum relevance threshold
            break
        
        # Extract the most relevant snippet (around keyword matches)
        snippet = _extract_relevant_snippet(claim_keywords, chunk_data["text"])
        
        results.append({
            "evidence_text": snippet,
            "section_heading": chunk_data["section_heading"],
            "page_number": chunk_data["page_number"],
        })
    
    return results


def _extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from text.
    Removes stopwords and short words.
    """
    # Stopwords to ignore
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "also",
        "now", "and", "but", "or", "if", "because", "until", "while",
        "although", "though", "whether", "this", "that", "these", "those",
        "which", "what", "who", "whom", "whose", "it", "its", "they", "them",
        "their", "we", "us", "our", "you", "your", "he", "him", "his", "she",
        "her", "i", "me", "my", "about", "over", "any", "both", "up", "down",
    }
    
    # Tokenize: extract alphanumeric words
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
    
    # Filter stopwords and short words
    keywords = [w for w in words if w not in stopwords and len(w) >= 3]
    
    return keywords


def _calculate_keyword_overlap(keywords: List[str], text: str) -> float:
    """
    Calculate keyword overlap score between keywords and text.
    Returns a score between 0.0 and 1.0.
    """
    if not keywords:
        return 0.0
    
    text_lower = text.lower()
    text_words = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text_lower))
    
    # Count keyword matches
    matches = sum(1 for kw in keywords if kw in text_words)
    
    # Normalize by number of keywords
    overlap_ratio = matches / len(keywords)
    
    # Bonus for exact phrase matches (consecutive keywords)
    if len(keywords) >= 2:
        phrase = " ".join(keywords[:3])
        if phrase in text_lower:
            overlap_ratio += 0.3
    
    return min(1.0, overlap_ratio)


def _get_section_boost(section_heading: str) -> float:
    """
    Return a boost multiplier based on section relevance.
    Technical sections get higher boosts.
    """
    if not section_heading:
        return 1.0
    
    section_lower = section_heading.lower()
    
    # High relevance sections
    high_relevance = ["method", "architecture", "model", "approach", "algorithm",
                      "implementation", "design", "mechanism", "attention",
                      "transformer", "encoder", "decoder", "training"]
    for term in high_relevance:
        if term in section_lower:
            return 1.5
    
    # Medium relevance sections
    medium_relevance = ["experiment", "result", "evaluation", "analysis",
                        "performance", "ablation", "comparison", "discussion"]
    for term in medium_relevance:
        if term in section_lower:
            return 1.3
    
    # Standard sections
    standard = ["introduction", "background", "related", "overview", "motivation"]
    for term in standard:
        if term in section_lower:
            return 1.1
    
    # Lower relevance sections
    low_relevance = ["conclusion", "future", "acknowledgment", "reference",
                     "appendix", "abstract"]
    for term in low_relevance:
        if term in section_lower:
            return 0.8
    
    return 1.0


def _extract_relevant_snippet(keywords: List[str], text: str, max_length: int = 500) -> str:
    """
    Extract the most relevant snippet from text based on keyword density.
    Returns a substring centered around the highest keyword concentration.
    """
    if len(text) <= max_length:
        return text.strip()
    
    text_lower = text.lower()
    
    # Find positions of keyword matches
    positions: List[int] = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        for match in re.finditer(pattern, text_lower):
            positions.append(match.start())
    
    if not positions:
        # No keywords found, return beginning of text
        return text[:max_length].strip() + "..."
    
    # Find the window with highest keyword density
    positions.sort()
    
    best_start = 0
    best_count = 0
    
    window_size = max_length
    
    for i, pos in enumerate(positions):
        # Count keywords in window starting near this position
        window_start = max(0, pos - 50)  # Start slightly before keyword
        window_end = window_start + window_size
        
        count = sum(1 for p in positions if window_start <= p < window_end)
        
        if count > best_count:
            best_count = count
            best_start = window_start
    
    # Extract snippet
    snippet_start = best_start
    snippet_end = min(len(text), snippet_start + max_length)
    
    # Adjust to word boundaries
    if snippet_start > 0:
        # Find start of word
        while snippet_start > 0 and text[snippet_start - 1].isalnum():
            snippet_start -= 1
    
    if snippet_end < len(text):
        # Find end of word
        while snippet_end < len(text) and text[snippet_end].isalnum():
            snippet_end += 1
    
    snippet = text[snippet_start:snippet_end].strip()
    
    # Add ellipsis if truncated
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    
    return snippet


def judge_claim_with_evidence(
    claim: str,
    evidence_snippets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Judge a single atomic claim against evidence snippets using keyword matching.
    
    This is a temporary heuristic-based stub - will be replaced with LLM later.
    
    Args:
        claim: One atomic factual claim (string)
        evidence_snippets: List of dicts with 'evidence_text' key
    
    Returns:
        Dict with:
        - verdict: "CORRECT" / "PARTIALLY_CORRECT" / "INCORRECT"
        - score: int from 1 to 10
    """
    if not claim:
        return {"verdict": "INCORRECT", "score": 1}
    
    if not evidence_snippets:
        return {"verdict": "INCORRECT", "score": 2}
    
    # Step 1: Extract key terms from claim
    claim_keywords = _extract_keywords(claim)
    
    if not claim_keywords:
        return {"verdict": "INCORRECT", "score": 2}
    
    # Step 2: Combine all evidence text
    combined_evidence = " ".join(
        e.get("evidence_text", "") for e in evidence_snippets
    ).lower()
    
    evidence_words = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', combined_evidence))
    
    # Step 3: Check keyword presence in evidence
    matched_keywords = [kw for kw in claim_keywords if kw in evidence_words]
    match_ratio = len(matched_keywords) / len(claim_keywords) if claim_keywords else 0
    
    # Step 4: Check for contradiction indicators
    contradiction_found = _check_for_contradiction(claim, combined_evidence)
    
    # Step 5: Determine verdict and score
    if contradiction_found:
        verdict = "INCORRECT"
        score = 2
    elif match_ratio >= 0.7:
        verdict = "CORRECT"
        # Score based on match quality: 8-10
        score = 8 + int(match_ratio * 2)  # 8, 9, or 10
        score = min(10, score)
    elif match_ratio >= 0.4:
        verdict = "PARTIALLY_CORRECT"
        # Score based on match quality: 4-7
        score = 4 + int(match_ratio * 6)  # Maps 0.4-0.7 to roughly 6-8, capped at 7
        score = min(7, max(4, score))
    else:
        verdict = "INCORRECT"
        # Score based on match quality: 1-3
        score = 1 + int(match_ratio * 5)  # 1, 2, or 3
        score = min(3, max(1, score))
    
    return {"verdict": verdict, "score": score}


def _check_for_contradiction(claim: str, evidence: str) -> bool:
    """
    Check if evidence contains indicators that contradict the claim.
    Simple heuristic: look for negation patterns near claim keywords.
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()
    
    # Extract main verb/action from claim
    claim_keywords = _extract_keywords(claim)
    
    if not claim_keywords:
        return False
    
    # Negation patterns
    negation_words = ["not", "no", "never", "cannot", "don't", "doesn't",
                      "didn't", "won't", "wouldn't", "isn't", "aren't",
                      "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                      "without", "lack", "lacks", "lacking", "fails",
                      "unable", "impossible", "incorrect", "wrong", "false"]
    
    # Check if negation appears near claim keywords in evidence
    for kw in claim_keywords[:3]:  # Check first few keywords
        # Find keyword position in evidence
        kw_pattern = r'\b' + re.escape(kw) + r'\b'
        for match in re.finditer(kw_pattern, evidence_lower):
            pos = match.start()
            # Check 50 chars before and after for negation
            window_start = max(0, pos - 50)
            window_end = min(len(evidence_lower), pos + len(kw) + 50)
            window = evidence_lower[window_start:window_end]
            
            for neg in negation_words:
                if re.search(r'\b' + neg + r'\b', window):
                    return True
    
    return False


def aggregate_claim_judgments(
    claim_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate multiple claim judgments into a final verdict and score.
    
    This is a temporary heuristic-based stub - will be replaced with LLM later.
    
    Args:
        claim_results: List of dicts, each containing:
            - verdict: "CORRECT" / "PARTIALLY_CORRECT" / "INCORRECT"
            - score: int from 1 to 10
    
    Returns:
        Dict with:
        - final_verdict: "CORRECT" / "PARTIALLY_CORRECT" / "INCORRECT"
        - final_score: int from 1 to 10
    """
    if not claim_results:
        return {"final_verdict": "INCORRECT", "final_score": 1}
    
    # Count verdicts
    correct_count = 0
    partial_count = 0
    incorrect_count = 0
    
    for result in claim_results:
        verdict = result.get("verdict", "INCORRECT").upper()
        if verdict == "CORRECT":
            correct_count += 1
        elif verdict == "PARTIALLY_CORRECT":
            partial_count += 1
        else:
            incorrect_count += 1
    
    total_claims = len(claim_results)
    
    # Determine final verdict based on composition
    if incorrect_count == 0 and partial_count == 0:
        # All claims are CORRECT
        final_verdict = "CORRECT"
    elif incorrect_count == 0:
        # Mix of CORRECT and PARTIALLY_CORRECT only
        final_verdict = "PARTIALLY_CORRECT"
    elif incorrect_count == total_claims:
        # All claims are INCORRECT
        final_verdict = "INCORRECT"
    elif incorrect_count >= total_claims / 2:
        # Majority INCORRECT → INCORRECT
        final_verdict = "INCORRECT"
    else:
        # Some INCORRECT but not majority → PARTIALLY_CORRECT
        final_verdict = "PARTIALLY_CORRECT"
    
    # Compute weighted average score
    # Weights: CORRECT=1.0, PARTIALLY_CORRECT=0.7, INCORRECT=0.3
    weighted_sum = 0.0
    weight_total = 0.0
    
    for result in claim_results:
        verdict = result.get("verdict", "INCORRECT").upper()
        score = result.get("score", 1)
        
        if verdict == "CORRECT":
            weight = 1.0
        elif verdict == "PARTIALLY_CORRECT":
            weight = 0.7
        else:
            weight = 0.3
        
        weighted_sum += score * weight
        weight_total += weight
    
    # Calculate final score
    if weight_total > 0:
        final_score = round(weighted_sum / weight_total)
    else:
        final_score = 1
    
    # Clamp score to 1-10 range
    final_score = max(1, min(10, final_score))
    
    return {"final_verdict": final_verdict, "final_score": final_score}


def generate_followup_question(
    claim_results: List[Dict[str, Any]],
    cognitive_level: int = 0
) -> str:
    """
    Generate a follow-up examiner question based on claim judgment results.
    
    If USE_LLM_QUESTIONS is True, uses OpenAI gpt-4.1-mini.
    Otherwise, uses deterministic rule-based templates.
    
    Args:
        claim_results: List of dicts, each containing:
            - claim: The atomic claim (string)
            - verdict: "CORRECT" / "PARTIALLY_CORRECT" / "INCORRECT"
        cognitive_level: Adaptive difficulty level (3-6)
            - 0: Auto-select based on verdicts (legacy behavior)
            - 3: Clarify misconceptions
            - 4: Analyze / compare
            - 5: Critique assumptions
            - 6: Propose alternatives / generalize
    
    Returns:
        A single follow-up question string in strict examiner tone.
    """
    if not claim_results:
        return "Explain the core mechanism described in the paper."
    
    # Categorize claims by verdict
    incorrect_claims: List[str] = []
    partial_claims: List[str] = []
    correct_claims: List[str] = []
    
    for result in claim_results:
        claim = result.get("claim", "")
        verdict = result.get("verdict", "INCORRECT").upper()
        
        if not claim:
            continue
        
        if verdict == "INCORRECT":
            incorrect_claims.append(claim)
        elif verdict == "PARTIALLY_CORRECT":
            partial_claims.append(claim)
        else:
            correct_claims.append(claim)
    
    # Get target claim for question generation (weakest first)
    if incorrect_claims:
        target_claim = incorrect_claims[0]
        target_verdict = "INCORRECT"
    elif partial_claims:
        target_claim = partial_claims[0]
        target_verdict = "PARTIALLY_CORRECT"
    elif correct_claims:
        target_claim = correct_claims[0]
        target_verdict = "CORRECT"
    else:
        return "Explain the fundamental contribution of this paper."
    
    # Try LLM generation if enabled
    if USE_LLM_QUESTIONS and cognitive_level >= 3:
        level_desc = COGNITIVE_LEVEL_PROMPTS.get(cognitive_level, COGNITIVE_LEVEL_PROMPTS[4])
        
        prompt = f"""Generate ONE follow-up viva question.

Student's claim (marked {target_verdict}):
"{target_claim}"

Cognitive level: {level_desc}

Rules:
- One sentence only
- Target the weakness in the claim
- Increase depth by one level
- No hints, no answers, no praise
- Formal academic tone
- Do not repeat the student's words

Output the question only."""

        llm_question = call_llm_question(prompt, max_tokens=50)
        if llm_question:
            return llm_question
        # Fall through to rule-based on LLM failure
    
    # Rule-based fallback (original logic)
    # Route based on cognitive level
    if cognitive_level == 3:
        return _generate_level3_question(target_claim)
    elif cognitive_level == 4:
        return _generate_level4_question(target_claim)
    elif cognitive_level == 5:
        return _generate_level5_question(target_claim)
    elif cognitive_level == 6:
        return _generate_level6_question(target_claim)
    
    # Legacy behavior (cognitive_level == 0): route by verdict
    if incorrect_claims:
        return _generate_correction_question(target_claim)
    elif partial_claims:
        return _generate_clarification_question(target_claim)
    else:
        return _generate_extension_question(target_claim)


def select_cognitive_level(final_score: int) -> int:
    """
    Select cognitive difficulty level based on student performance.
    
    Implements adaptive difficulty policy:
    - Score ≤ 3:  Level 3 (Clarify misconceptions)
    - Score 4-6:  Level 4 (Analyze / compare)
    - Score 7-8:  Level 5 (Critique assumptions)
    - Score 9-10: Level 6 (Propose alternatives / generalize)
    
    Args:
        final_score: The aggregated final score (1-10)
    
    Returns:
        Cognitive level (3, 4, 5, or 6)
    """
    if final_score <= 3:
        return 3
    elif final_score <= 6:
        return 4
    elif final_score <= 8:
        return 5
    else:
        return 6


def get_cognitive_level_label(level: int) -> str:
    """
    Get human-readable label for cognitive level.
    
    Args:
        level: Cognitive level (3-6)
    
    Returns:
        Label string
    """
    labels = {
        3: "Clarify Misconceptions",
        4: "Analyze / Compare",
        5: "Critique Assumptions",
        6: "Propose Alternatives / Generalize",
    }
    return labels.get(level, "Unknown")


def _generate_level3_question(claim: str) -> str:
    """
    Level 3: Clarify misconceptions.
    Basic correction and understanding questions.
    """
    core_concept = _extract_core_concept(claim)
    
    # Check if claim is non-substantive (e.g., "I don't know")
    claim_lower = claim.lower()
    non_substantive = ["don't know", "do not know", "not sure", "no idea", "sorry"]
    is_non_substantive = any(phrase in claim_lower for phrase in non_substantive)
    
    if not core_concept or is_non_substantive:
        return "What is the main mechanism proposed in the paper? Start with the basics."
    
    templates = [
        f"What does the paper actually say about {core_concept}?",
        f"Can you restate what {core_concept} means according to the paper?",
        f"What is the correct understanding of {core_concept}?",
    ]
    
    # Deterministic selection based on claim length
    idx = len(claim) % len(templates)
    return templates[idx]


def _generate_level4_question(claim: str) -> str:
    """
    Level 4: Analyze / compare.
    Questions that require analysis or comparison.
    """
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "How does the proposed approach compare to existing methods?"
    
    templates = [
        f"How does {core_concept} compare to the baseline approach?",
        f"What are the key differences between {core_concept} and traditional methods?",
        f"Analyze the relationship between {core_concept} and model performance.",
        f"What role does {core_concept} play in the overall architecture?",
    ]
    
    idx = len(claim) % len(templates)
    return templates[idx]


def _generate_level5_question(claim: str) -> str:
    """
    Level 5: Critique assumptions.
    Questions about limitations, assumptions, and critical evaluation.
    """
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "What assumptions does the paper make, and are they justified?"
    
    templates = [
        f"What assumptions underlie {core_concept}, and could they be violated?",
        f"What are the limitations of {core_concept} as presented in the paper?",
        f"Under what conditions might {core_concept} fail or underperform?",
        f"What criticisms could be raised against the design of {core_concept}?",
    ]
    
    idx = len(claim) % len(templates)
    return templates[idx]


def _generate_level6_question(claim: str) -> str:
    """
    Level 6: Propose alternatives / generalize.
    Synthesis, transfer, and generalization questions.
    """
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "How might this approach be extended to other domains or problems?"
    
    templates = [
        f"How could {core_concept} be modified to address its limitations?",
        f"What alternative approaches to {core_concept} might achieve similar goals?",
        f"How might {core_concept} generalize to other domains or modalities?",
        f"If you were to improve {core_concept}, what changes would you propose?",
    ]
    
    idx = len(claim) % len(templates)
    return templates[idx]


def _generate_correction_question(claim: str) -> str:
    """
    Generate a WHY/HOW question targeting an incorrect claim.
    Forces the student to reconsider their understanding.
    """
    # Extract core concept from claim (not just keywords)
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "Your previous statement was incorrect. Justify your reasoning."
    
    # Select question template based on claim structure
    claim_lower = claim.lower()
    
    # Check for causal claims
    if any(word in claim_lower for word in ["allows", "enables", "causes", "leads", "results"]):
        return f"Why does {core_concept} have that effect? Justify with evidence from the paper."
    
    # Check for comparison claims
    if any(word in claim_lower for word in ["better", "faster", "more", "less", "unlike"]):
        return f"What specific evidence supports your comparison regarding {core_concept}?"
    
    # Check for mechanism claims
    if any(word in claim_lower for word in ["uses", "computes", "processes", "applies"]):
        return f"How exactly does {core_concept} work according to the paper?"
    
    # Default correction question
    return f"Reconsider your statement about {core_concept}. What does the paper actually say?"


def _generate_clarification_question(claim: str) -> str:
    """
    Generate a clarification or depth question for partially correct claims.
    Probes for missing details or precision.
    """
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "Be more precise. What specific details are you omitting?"
    
    claim_lower = claim.lower()
    
    # Ask for specifics on mechanism
    if any(word in claim_lower for word in ["attention", "layer", "encoder", "decoder"]):
        return f"What are the specific components of {core_concept} as described in the paper?"
    
    # Ask for numerical or quantitative details
    if any(word in claim_lower for word in ["reduces", "improves", "increases", "faster"]):
        return f"By what factor or metric does {core_concept} achieve this improvement?"
    
    # Ask for missing conditions or constraints
    if any(word in claim_lower for word in ["allows", "enables", "can"]):
        return f"Under what conditions does {core_concept} enable this capability?"
    
    # Default clarification
    return f"Elaborate on {core_concept}. What details are you leaving out?"


def _generate_extension_question(claim: str) -> str:
    """
    Generate a deeper extension question for correct claims.
    Probes for implications, trade-offs, or connections.
    """
    core_concept = _extract_core_concept(claim)
    
    if not core_concept:
        return "What are the implications of this approach? Discuss trade-offs."
    
    claim_lower = claim.lower()
    
    # Ask about trade-offs
    if any(word in claim_lower for word in ["removes", "eliminates", "replaces"]):
        return f"What trade-offs arise from {core_concept}? What is lost or gained?"
    
    # Ask about implications
    if any(word in claim_lower for word in ["allows", "enables", "improves"]):
        return f"What are the downstream implications of {core_concept} for model design?"
    
    # Ask about limitations
    if any(word in claim_lower for word in ["attention", "transformer", "model"]):
        return f"What limitations of {core_concept} does the paper acknowledge?"
    
    # Ask about connections to other work
    if any(word in claim_lower for word in ["novel", "new", "proposed", "introduces"]):
        return f"How does {core_concept} relate to prior approaches in the literature?"
    
    # Default extension
    return f"What would happen if {core_concept} were modified or removed? Analyze the impact."


def _extract_core_concept(claim: str) -> str:
    """
    Extract the core concept phrase from a claim.
    Returns a meaningful phrase (not just a single keyword).
    
    Examples:
    - "Removing recurrence allows parallel computation" -> "removing recurrence"
    - "Self-attention removes recurrence" -> "self-attention removing recurrence"
    - "Parallel computation reduces training time" -> "parallel computation"
    """
    if not claim:
        return ""
    
    claim = claim.strip()
    words = claim.split()
    
    if len(words) < 2:
        return claim.lower()
    
    claim_lower = claim.lower()
    
    # Pattern 1: "X removes/eliminates/replaces Y" -> "X removing Y" or "removing Y"
    for verb in ["removes", "eliminates", "replaces", "dispenses"]:
        if verb in claim_lower:
            parts = claim_lower.split(verb)
            if len(parts) >= 2:
                subject = parts[0].strip()
                obj = parts[1].strip().rstrip('.,;:')
                # Convert to gerund form
                gerund = verb.rstrip('s') + "ing"
                if subject and len(subject.split()) <= 3:
                    return f"{subject} {gerund} {obj}"
                return f"{gerund} {obj}"
    
    # Pattern 2: "X allows/enables Y" -> "X" (the enabler)
    for verb in ["allows", "enables", "permits", "facilitates"]:
        if verb in claim_lower:
            parts = claim_lower.split(verb)
            if len(parts) >= 1 and parts[0].strip():
                subject = parts[0].strip()
                if len(subject.split()) <= 4:
                    return subject
    
    # Pattern 3: "X reduces/improves/increases Y" -> "X"
    for verb in ["reduces", "improves", "increases", "decreases", "enhances"]:
        if verb in claim_lower:
            parts = claim_lower.split(verb)
            if len(parts) >= 1 and parts[0].strip():
                subject = parts[0].strip()
                if len(subject.split()) <= 4:
                    return subject
    
    # Pattern 4: "X computes/processes Y" -> "X computing Y"
    for verb in ["computes", "processes", "calculates", "generates"]:
        if verb in claim_lower:
            parts = claim_lower.split(verb)
            if len(parts) >= 2:
                subject = parts[0].strip()
                obj = parts[1].strip().rstrip('.,;:')
                # Take first few words of object
                obj_words = obj.split()[:3]
                obj_short = " ".join(obj_words)
                if subject and len(subject.split()) <= 3:
                    return f"{subject}"
                return obj_short
    
    # Pattern 5: Gerund at start "Removing X..." -> "removing X"
    if words[0].lower().endswith('ing'):
        # Take the gerund and next 1-2 words
        concept_words = words[:3] if len(words) >= 3 else words
        return " ".join(concept_words).lower().rstrip('.,;:')
    
    # Default: Take first 2-3 meaningful words (skip articles)
    skip_words = {"the", "a", "an", "this", "that"}
    meaningful = [w for w in words if w.lower() not in skip_words]
    if meaningful:
        concept_words = meaningful[:3]
        return " ".join(concept_words).lower().rstrip('.,;:')
    
    return words[0].lower()


# ============================================================================
# VOICE INPUT MODULE (Standalone)
# ============================================================================

def speak_examiner_text(text: str) -> None:
    """
    Speak text using text-to-speech in a neutral academic tone.
    
    Uses pyttsx3 for offline TTS. Also prints the text to console.
    Silently ignores any TTS errors to avoid blocking the flow.
    
    Args:
        text: The text to speak
    """
    # Always print to console
    print(f"  [EXAMINER]: {text}")
    
    # Attempt TTS
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Configure for neutral academic tone
        engine.setProperty('rate', 150)  # Moderate speed
        engine.setProperty('volume', 0.9)  # Slightly below max
        
        # Try to select a neutral voice if available
        voices = engine.getProperty('voices')
        if voices:
            # Prefer a neutral/default voice (usually first)
            engine.setProperty('voice', voices[0].id)
        
        engine.say(text)
        engine.runAndWait()
        
    except ImportError:
        # pyttsx3 not installed - silent fail
        pass
    except Exception:
        # Any TTS error - silent fail
        pass


def transcribe_judge_answer(timeout: int = 5, phrase_time_limit: int = 5) -> str:
    """
    Capture audio from the default microphone and transcribe to text.
    
    Uses Google's free speech recognition API via speech_recognition library.
    Strictly limited to avoid blocking waits.
    
    Args:
        timeout: Maximum seconds to wait for speech to start (default 5)
        phrase_time_limit: Maximum seconds for a single phrase (default 5)
    
    Returns:
        Transcribed text as a string, or empty string on failure.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return ""
    
    recognizer = sr.Recognizer()
    
    # Tight energy threshold for faster detection
    recognizer.energy_threshold = 400
    recognizer.dynamic_energy_threshold = False  # Disable for determinism
    recognizer.pause_threshold = 0.8  # Shorter pause detection
    
    try:
        with sr.Microphone() as source:
            # Quick ambient noise adjustment (0.5 seconds max)
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("[VOICE] Listening... (5 seconds max)")
            
            try:
                audio = recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                return ""
            
    except OSError:
        return ""
    except Exception:
        return ""
    
    # Transcribe using Google's free API
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""
    except Exception:
        return ""


def get_judge_answer(prompt: str = "Your answer: ", min_length: int = 10) -> str:
    """
    Get the student's answer via voice or typed input.
    
    First attempts voice transcription (max 5 seconds).
    If empty or too short, falls back to typed input immediately.
    
    Args:
        prompt: The prompt to show for typed input fallback
        min_length: Minimum character length to accept voice input
    
    Returns:
        The student's answer as a string (never empty if user types something).
    """
    print()
    print("[INPUT] Speak your answer now (5 seconds)...")
    
    # Try voice input first (strict 5-second limit)
    voice_text = transcribe_judge_answer(timeout=5, phrase_time_limit=5)
    
    # Check if voice input was successful
    if voice_text and len(voice_text.strip()) >= min_length:
        print(f"[INPUT] Heard: \"{voice_text}\"")
        return voice_text.strip()
    
    # Fallback to typed input
    print()
    print("[INPUT] No speech detected. Please answer, or press Enter to skip.")
    try:
        typed_text = input(prompt)
        return typed_text.strip()
    except EOFError:
        return ""
    except Exception:
        return ""


def is_clarification_request(answer: str) -> bool:
    """
    Check if the answer is a request to repeat or clarify the question.
    
    Uses deterministic string matching.
    
    Args:
        answer: The student's answer text
    
    Returns:
        True if the answer is a clarification/repeat request, False otherwise.
    """
    if not answer:
        return False
    
    answer_lower = answer.lower().strip()
    
    # Phrases that indicate a repeat/clarification request
    repeat_phrases = [
        "repeat",
        "say again",
        "didn't hear",
        "did not hear",
        "can you repeat",
        "could you repeat",
        "please repeat",
        "rephrase",
        "say that again",
        "what was the question",
        "what's the question",
        "pardon",
        "sorry what",
        "come again",
        "one more time",
        "again please",
        "i didn't catch",
        "i did not catch",
        "can't hear",
        "cannot hear",
    ]
    
    for phrase in repeat_phrases:
        if phrase in answer_lower:
            return True
    
    # Also check for very short "what?" or "huh?" responses
    short_confusion = ["what", "huh", "sorry", "excuse me", "pardon"]
    if len(answer_lower) < 15:
        for phrase in short_confusion:
            if answer_lower.strip("?.,! ") == phrase:
                return True
    
    return False


def is_explanation_request(text: str) -> bool:
    """
    Check if the text is a request for explanation of the verdict.
    
    Uses deterministic string matching.
    
    Args:
        text: The user's input text
    
    Returns:
        True if the text is an explanation request, False otherwise.
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Phrases that indicate an explanation request
    explanation_phrases = [
        "why",
        "explain",
        "on what basis",
        "where in the paper",
        "show evidence",
        "what evidence",
        "how come",
        "reasoning",
        "justify",
        "based on what",
        "prove it",
        "show me",
        "which part",
        "what section",
        "page number",
        "source",
    ]
    
    for phrase in explanation_phrases:
        if phrase in text_lower:
            return True
    
    return False


def _print_verdict_explanation(claim_results: List[Dict[str, Any]]) -> None:
    """
    Print explanation of the verdict using stored evidence.
    
    Reuses evidence already retrieved during judging (no re-retrieval).
    
    Args:
        claim_results: List of claim judgment results with stored evidence
    """
    if not claim_results:
        print("No claims were evaluated.")
        return
    
    # Find incorrect and partial claims
    incorrect_claims = [r for r in claim_results if r.get("verdict") == "INCORRECT"]
    partial_claims = [r for r in claim_results if r.get("verdict") == "PARTIALLY_CORRECT"]
    correct_claims = [r for r in claim_results if r.get("verdict") == "CORRECT"]
    
    # Report on incorrect claims
    if incorrect_claims:
        print("INCORRECT CLAIMS:")
        for r in incorrect_claims:
            claim = r.get("claim", "")
            print(f"  - \"{claim}\"")
            
            # Use stored evidence (no re-retrieval)
            evidence = r.get("evidence", [])
            if evidence:
                e = evidence[0]
                section = e.get("section_heading", "Unknown")
                page = e.get("page_number", "?")
                snippet = e.get("evidence_text", "")[:150]
                print(f"    [Section: {section}, Page: {page}]")
                print(f"    Evidence: \"{snippet}...\"")
            else:
                print("    [No matching evidence found in paper]")
        print()
    
    # Report on partial claims
    if partial_claims:
        print("PARTIALLY CORRECT CLAIMS:")
        for r in partial_claims:
            claim = r.get("claim", "")
            print(f"  - \"{claim}\"")
            print("    (Missing precision or supporting details)")
            
            # Use stored evidence
            evidence = r.get("evidence", [])
            if evidence:
                e = evidence[0]
                section = e.get("section_heading", "Unknown")
                page = e.get("page_number", "?")
                print(f"    [Section: {section}, Page: {page}]")
        print()
    
    # If all correct
    if not incorrect_claims and not partial_claims and correct_claims:
        print("All claims were verified against the paper.")
        r = correct_claims[0]
        evidence = r.get("evidence", [])
        if evidence:
            e = evidence[0]
            section = e.get("section_heading", "Unknown")
            page = e.get("page_number", "?")
            print(f"  [Supporting: Section {section}, Page {page}]")


def _get_explanation_summary(claim_results: List[Dict[str, Any]]) -> str:
    """
    Generate a concise spoken summary of the verdict explanation.
    
    Returns a single sentence suitable for TTS.
    
    Args:
        claim_results: List of claim judgment results
    
    Returns:
        A concise summary sentence.
    """
    if not claim_results:
        return "No claims were evaluated."
    
    incorrect_count = sum(1 for r in claim_results if r.get("verdict") == "INCORRECT")
    partial_count = sum(1 for r in claim_results if r.get("verdict") == "PARTIALLY_CORRECT")
    correct_count = sum(1 for r in claim_results if r.get("verdict") == "CORRECT")
    total = len(claim_results)
    
    if incorrect_count == total:
        return "The answer was marked incorrect because no claims were supported by the paper."
    elif incorrect_count > 0 and partial_count > 0:
        return f"The answer contained {incorrect_count} incorrect and {partial_count} partially correct claims."
    elif incorrect_count > 0:
        return f"The answer was marked incorrect due to {incorrect_count} unsupported claims."
    elif partial_count == total:
        return "The answer was partially correct but lacked precision or supporting details."
    elif partial_count > 0:
        return f"The answer had {correct_count} correct and {partial_count} partially correct claims."
    else:
        return "All claims in the answer were verified against the paper."


def generate_verdict_explanation(
    claim_results: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]]
) -> str:
    """
    Generate a brief explanation of the verdict with supporting evidence.
    DEPRECATED: Use _print_verdict_explanation instead.
    
    Args:
        claim_results: List of claim judgment results
        chunks: The document chunks for evidence retrieval
    
    Returns:
        A concise explanation string.
    """
    # Legacy function - redirect to new implementation
    _print_verdict_explanation(claim_results)
    return ""  # Already printed


def _legacy_generate_verdict_explanation(
    claim_results: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]]
) -> str:
    """
    Legacy: Generate explanation by re-retrieving evidence.
    Kept for backwards compatibility.
    """
    if not claim_results:
        return "No claims were evaluated."
    
    # Find incorrect and partial claims
    incorrect_claims = [r for r in claim_results if r.get("verdict") == "INCORRECT"]
    partial_claims = [r for r in claim_results if r.get("verdict") == "PARTIALLY_CORRECT"]
    
    lines: List[str] = []
    
    # Report on incorrect claims
    if incorrect_claims:
        lines.append("INCORRECT CLAIMS:")
        for r in incorrect_claims[:2]:  # Limit to 2
            claim = r.get("claim", "")
            lines.append(f"  - \"{claim}\"")
            
            # Get evidence for this claim
            evidence = retrieve_evidence_for_claim(claim, chunks)
            if evidence:
                e = evidence[0]
                section = e.get("section_heading", "Unknown")
                page = e.get("page_number", "?")
                snippet = e.get("evidence_text", "")[:150]
                lines.append(f"    [Section: {section}, Page: {page}]")
                lines.append(f"    Evidence: \"{snippet}...\"")
        lines.append("")
    
    # Report on partial claims
    if partial_claims:
        lines.append("PARTIALLY CORRECT CLAIMS:")
        for r in partial_claims[:2]:  # Limit to 2
            claim = r.get("claim", "")
            lines.append(f"  - \"{claim}\"")
            lines.append("    (Missing precision or supporting details)")
            
            # Get evidence for this claim
            evidence = retrieve_evidence_for_claim(claim, chunks)
            if evidence:
                e = evidence[0]
                section = e.get("section_heading", "Unknown")
                page = e.get("page_number", "?")
                lines.append(f"    [Section: {section}, Page: {page}]")
        lines.append("")
    
    # If all correct
    if not incorrect_claims and not partial_claims:
        lines.append("All claims were verified against the paper.")
        # Show one supporting evidence
        if claim_results:
            claim = claim_results[0].get("claim", "")
            evidence = retrieve_evidence_for_claim(claim, chunks)
            if evidence:
                e = evidence[0]
                section = e.get("section_heading", "Unknown")
                page = e.get("page_number", "?")
                lines.append(f"  [Supporting: Section {section}, Page {page}]")
    
    return "\n".join(lines)


def judge_answer(
    question: str,
    student_answer: str,
    evidence_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Strictly judge a student's answer based on paper evidence.
    
    Args:
        question: The examination question
        student_answer: The student's response
        evidence_chunks: Chunks from the paper as evidence
    
    
    Returns:
        Dict with keys: verdict (str), score (int)
    """
    # Build evidence from chunks
    evidence = "\n".join([c["text"][:600] for c in evidence_chunks[:4]])
    
    system_context = """Task: Evaluate a viva answer using binary-first evaluation.

Evaluation rules:
- Base evaluation ONLY on the provided paper evidence
- Ignore fluency, confidence, or tone of the answer
- If ANY factual claim contradicts evidence → verdict is INCORRECT
- Do NOT explain concepts
- Do NOT soften feedback
- Do NOT add commentary beyond verdict and score"""

    prompt = f"""QUESTION: {question}

STUDENT ANSWER: {student_answer}

PAPER EVIDENCE:
{evidence}

Evaluate the answer. Output EXACTLY in this format:
Verdict: CORRECT / PARTIALLY_CORRECT / INCORRECT
Score: <integer from 1 to 10>"""

    response = _call_llm(prompt, system_context)
    
    # Parse verdict and score
    verdict = "INCORRECT"
    score = 1
    
    lines = response.upper().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict_text = line.replace("VERDICT:", "").strip()
            if "CORRECT" in verdict_text and "PARTIALLY" not in verdict_text and "INCORRECT" not in verdict_text:
                verdict = "CORRECT"
            elif "PARTIALLY" in verdict_text:
                verdict = "PARTIALLY_CORRECT"
            else:
                verdict = "INCORRECT"
        elif line.startswith("SCORE:"):
            try:
                score_text = line.replace("SCORE:", "").strip()
                # Extract just the number
                score_num = ''.join(c for c in score_text if c.isdigit())
                if score_num:
                    score = max(1, min(10, int(score_num)))
            except:
                score = 1
    
    return {
        "verdict": verdict,
        "score": score
    }


def print_knowledge_anchors(anchors: List[str]) -> None:
    """Print knowledge anchors in a formatted way."""
    print()
    print("=" * 70)
    print("KNOWLEDGE ANCHORS")
    print("=" * 70)
    print()
    
    for i, anchor in enumerate(anchors, 1):
        print(f"  {i}. {anchor}")
        print()
    
    print("=" * 70)


def run_examiner_dry_run() -> None:
    """
    Simulate a complete examiner interaction using stubbed functions.
    
    This dry run demonstrates the full viva flow:
    1. Select knowledge anchor → Generate question
    2. Process sample answer → Extract atomic claims
    3. Retrieve evidence → Judge each claim
    4. Aggregate results → Generate follow-up
    
    No LLMs, no voice, no user input. Fully deterministic.
    """
    # ========================================
    # HARDCODED PDF PATH
    # ========================================
    pdf_path = r"C:\Users\dhrub\OneDrive\Desktop\SocratesAI\test.pdf"
    
    # ========================================
    # STEP 0: Load PDF chunks
    # ========================================
    print()
    print("=" * 70)
    print("SOCRATES AI - EXAMINER DRY RUN")
    print("=" * 70)
    print()
    
    print("[Loading PDF...]")
    try:
        chunks = debug_ingest_pdf(pdf_path)
        print(f"[Loaded {len(chunks)} chunks from PDF]")
    except Exception as e:
        print(f"[ERROR] Could not load PDF: {e}")
        return
    
    print()
    
    # ========================================
    # STEP 1: Select Knowledge Anchor (hardcoded)
    # ========================================
    knowledge_anchor = "Self-attention mechanism and its role in replacing recurrence"
    
    print("-" * 70)
    print("STEP 1: KNOWLEDGE ANCHOR")
    print("-" * 70)
    print(f"  {knowledge_anchor}")
    print()
    
    # ========================================
    # STEP 2: Generate Examiner Question (stubbed)
    # ========================================
    examiner_question = generate_examiner_question(knowledge_anchor)
    
    print("-" * 70)
    print("STEP 2: EXAMINER QUESTION")
    print("-" * 70)
    speak_examiner_text(examiner_question)
    print()
    
    # ========================================
    # STEP 3: Get Student Answer (with repeat handling)
    # ========================================
    print("-" * 70)
    print("STEP 3: YOUR ANSWER")
    print("-" * 70)
    
    max_repeats = 3  # Prevent infinite loops
    repeat_count = 0
    sample_answer = ""
    
    while repeat_count < max_repeats:
        sample_answer = get_judge_answer(prompt="  > ")
        
        # Check if this is a clarification/repeat request
        if sample_answer and is_clarification_request(sample_answer):
            repeat_count += 1
            print()
            print(f"  [Repeating question ({repeat_count}/{max_repeats})...]")
            print()
            speak_examiner_text(examiner_question)
            print()
            continue
        
        # Got a real answer (or empty), exit loop
        break
    
    if not sample_answer:
        print("  [No answer provided. Using default sample answer.]")
        sample_answer = (
            "Self-attention removes recurrence, which allows parallel computation "
            "and reduces training time. The attention mechanism computes weighted "
            "sums of all input positions, enabling the model to capture long-range "
            "dependencies without sequential processing."
        )
    
    print()
    print(f"  Answer: {sample_answer}")
    print()
    
    # ========================================
    # STEP 4: Extract Atomic Claims
    # ========================================
    claims = extract_atomic_claims(sample_answer)
    
    print("-" * 70)
    print("STEP 4: EXTRACTED CLAIMS")
    print("-" * 70)
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim}")
    print()
    
    # ========================================
    # STEP 5: Retrieve Evidence & Judge Each Claim
    # ========================================
    print("-" * 70)
    print("STEP 5: CLAIM VERDICTS")
    print("-" * 70)
    
    claim_results: List[Dict[str, Any]] = []
    
    for i, claim in enumerate(claims, 1):
        # Retrieve evidence for this claim
        evidence = retrieve_evidence_for_claim(claim, chunks)
        
        # Judge the claim against evidence
        judgment = judge_claim_with_evidence(claim, evidence)
        
        # Store result with claim text AND evidence for explainability
        result = {
            "claim": claim,
            "verdict": judgment["verdict"],
            "score": judgment["score"],
            "evidence": evidence  # Store for later explanation
        }
        claim_results.append(result)
        
        # Print result
        verdict = judgment["verdict"]
        score = judgment["score"]
        print(f"  {i}. [{verdict}] (Score: {score}/10)")
        print(f"     \"{claim}\"")
        print()
    
    # ========================================
    # STEP 6: Aggregate Final Verdict
    # ========================================
    aggregated = aggregate_claim_judgments(claim_results)
    final_verdict = aggregated["final_verdict"]
    final_score = aggregated["final_score"]
    
    print("-" * 70)
    print("STEP 6: FINAL RESULT")
    print("-" * 70)
    print(f"  Final Verdict: {final_verdict}")
    print(f"  Final Score:   {final_score}/10")
    print()
    
    # ========================================
    # STEP 6a: Select Cognitive Level (Adaptive Difficulty)
    # ========================================
    cognitive_level = select_cognitive_level(final_score)
    cognitive_label = get_cognitive_level_label(cognitive_level)
    print(f"  Cognitive Level Selected: Level {cognitive_level} ({cognitive_label})")
    print()
    
    # ========================================
    # STEP 6b: Explainability on Demand (post-verdict)
    # ========================================
    print("[EXPLAINABILITY] You may ask for an explanation now. Speak clearly (5 seconds)...")
    print()
    
    # Try voice input first (5-second limit)
    explanation_request = transcribe_judge_answer(timeout=5, phrase_time_limit=5)
    
    # If voice is empty, allow typed input
    if not explanation_request:
        print("[EXPLAINABILITY] No voice detected. Type 'why' or 'explain', or press Enter to continue.")
        try:
            explanation_request = input("  > ").strip()
        except (EOFError, Exception):
            explanation_request = ""
    else:
        print(f"[EXPLAINABILITY] Heard: \"{explanation_request}\"")
    
    # Check for explanation triggers
    if explanation_request and is_explanation_request(explanation_request):
        print()
        print("-" * 70)
        print("EXPLANATION OF VERDICT")
        print("-" * 70)
        # Speak a concise summary (do NOT read evidence aloud)
        summary = _get_explanation_summary(claim_results)
        speak_examiner_text(summary)
        print()
        # Print detailed explanation to console
        _print_verdict_explanation(claim_results)
        print()
    
    # ========================================
    # STEP 7: Generate Follow-up Question (Adaptive)
    # ========================================
    followup = generate_followup_question(claim_results, cognitive_level=cognitive_level)
    
    print("-" * 70)
    print("STEP 7: FOLLOW-UP QUESTION")
    print("-" * 70)
    speak_examiner_text(followup)
    print()
    
    print("=" * 70)
    print("DRY RUN COMPLETE")
    print("=" * 70)


def run_ingestion_debug() -> None:
    """
    Manual testing function for PDF ingestion debugging.
    This function will be deleted later.
    """
    # ========================================
    # HARDCODED PDF PATH - CHANGE AS NEEDED
    # ========================================
    pdf_path = r"C:\Users\dhrub\OneDrive\Desktop\SocratesAI\test.pdf"
    # ========================================

    print()
    print("=" * 70)
    print("RUNNING INGESTION DEBUG")
    print(f"PDF: {pdf_path}")
    print("=" * 70)

    try:
        chunks = debug_ingest_pdf(pdf_path)
        print_debug_chunks(chunks)
        check_section_integrity(chunks)
        
        # Extract knowledge anchors
        print("\nExtracting Knowledge Anchors (requires OPENAI_API_KEY)...")
        try:
            anchors = extract_knowledge_anchors(chunks)
            print_knowledge_anchors(anchors)
        except Exception as e:
            print(f"[WARN] Could not extract knowledge anchors: {e}")
            
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    run_examiner_dry_run()
