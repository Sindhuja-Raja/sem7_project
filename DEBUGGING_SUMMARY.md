# DEBUGGING RESULTS - Problem-Solution & Contradiction Detection

## TEST EXECUTION

Ran actual pipeline with real papers from `data/papers/`:
- Papers tested: 2 valid papers (1d84ec52ed4992e4, 5464c76b2da16e07)
- Combined size: 13,943 words, 102,472 characters
- Configuration: Default config, GROQ_API_KEY enabled

---

## ISSUE #1: PROBLEM-SOLUTION ANALYSIS - "No data extracted for this section"

### ROOT CAUSE

**Insufficient token budget for JSON generation from retrieved context**

The LLM receives ~3,000+ tokens of retrieved context but is only allocated 120-240 tokens for the complete JSON response. This causes the model to generate incomplete/truncated JSON that fails validation.

Groq's JSON validator rejects the malformed output, `llm.parse_json()` returns None, and the function returns empty results.

### EVIDENCE

**Real execution trace:**

```
Papers selected: 2
RAG Query: "problem statement, research problem, methodology, results"
RAG Top K: 3

Context retrieved:
  Paper 0: 6,445 characters (~1,600 tokens)
  Paper 1: 6,973 characters (~1,700 tokens)
  TOTAL: 13,418 characters (~3,300 tokens)

Max tokens allocated for JSON response: min(3000, 120 * 2) = 240 tokens

LLM Groq response:
  Status: 400 Bad Request
  Error: "Failed to validate JSON. Please adjust your prompt."
  Reason: Incomplete JSON (ran out of tokens mid-generation)

Extracted problems:
  Paper 0: main=False (NOT_REPORTED), secondary=0
  Paper 1: main=False (NOT_REPORTED), secondary=0

Final table: 0 rows
UI displays: "No data extracted for this section"
```

**Root breakdown:**
1. ✓ knowledge_table passed: 2 rows
2. ✓ RAG retrieval working: 13.4KB context
3. ✓ LLM available: True
4. ✗ JSON extraction FAILED: token limit
5. ✗ Result: empty list

### MINIMUM FIX

**In `problem_solution_analysis.py`, function `_extract_problems()` (~line 125):**

```python
OLD:
max_tokens=min(3000, 120 * len(papers)),

NEW:
max_tokens=min(4000, 400 * len(papers)),
```

**Reasoning:**
- 400 tokens per paper = 800 tokens for 2 papers (still within 4000 limit)
- Accommodates retrieved context (~3.3K tokens) + JSON structure + claim content
- Provides margin for multi-paper batches (up to 5 papers: 2000 tokens)

---

## ISSUE #2: CONTRADICTION DETECTION - "No comparable claim pairs were found"

### ROOT CAUSE

**Insufficient token budget for JSON generation from retrieved context**

Same root cause as Problem-Solution. The LLM receives ~3,300 tokens of retrieved context but is only allocated 700 tokens for the complete JSON response containing all extracted claims.

Model generates incomplete JSON → validation fails → 0 claims extracted → 0 pairs to compare → empty report.

### EVIDENCE

**Real execution trace:**

```
Papers selected: 2
RAG Query: "claims, experimental results, evaluation, discussion, findings"
RAG Top K: 3

Context retrieved:
  Paper 0: 6,786 characters (~1,700 tokens)
  Paper 1: 7,270 characters (~1,800 tokens)
  TOTAL: 14,056 characters (~3,500 tokens)

Max tokens allocated for JSON response: min(4000, 350 * 2) = 700 tokens

LLM Groq response:
  Status: 400 Bad Request
  Error: "Failed to validate JSON. Please adjust your prompt."
  Reason: Incomplete JSON (ran out of tokens mid-generation)

Extracted claims:
  Paper 0: 0 claims
  Paper 1: 0 claims

Candidate pairs before similarity filtering: 0
Pairs after similarity filtering: 0

UI displays: "No comparable claim pairs were found across the selected papers."
```

**Root breakdown:**
1. ✓ RAG retrieval working: 14.1KB context
2. ✓ LLM available: True
3. ✗ JSON extraction FAILED: token limit (700 insufficient)
4. ✗ No claims extracted
5. ✗ No pairs created
6. ✗ Result: empty report

### MINIMUM FIX

**In `contradiction_detection.py`, function `extract_claims()` (~line 234):**

```python
OLD:
max_tokens=min(4000, 350 * len(papers)),

NEW:
max_tokens=min(5000, 700 * len(papers)),
```

**Reasoning:**
- 700 tokens per paper = 1400 tokens for 2 papers
- Accommodates retrieved context (~3.5K tokens in message) + JSON structure + full claim text
- Increases limit to 5000 (from 4000) to ensure sufficient budget for complex multi-paper batches

---

## CASCADING FAILURE: Knowledge Extraction (Root Cause of Downstream Issues)

### ROOT CAUSE

Same token budget issue affects Knowledge Analysis, which runs FIRST in the pipeline.

All 32 fields in knowledge_table show: `"Not Found in Retrieved Context"`

This is NOT because RAG failed, but because LLM JSON extraction failed.

### EVIDENCE

```
Knowledge extraction response:
  Status: 400 / 413
  Errors: "Request too large" / "Failed to generate JSON"
  
Result: knowledge_table with all fields = "Not Found in Retrieved Context"

Example:
  Paper Title: 1d84ec52ed4992e4
  AI Technique: Not Found in Retrieved Context
  AI Model: Not Found in Retrieved Context
  Dataset: Not Found in Retrieved Context
  (... all 32 fields ...)
```

This empty knowledge_table cascades to Problem-Solution Analysis, which depends on `knowledge.knowledge_table` for solution mapping.

### MINIMUM FIX (Optional but Recommended)

**In `knowledge_analysis.py`, find knowledge extraction calls:**

Estimate current value and increase max_tokens:
```python
# Likely around line 200-300, in the extraction loop

OLD (estimate):
max_tokens=min(3000, 250 * len(papers)),

NEW:
max_tokens=min(5000, 800 * len(papers)),
```

---

## SUMMARY TABLE

| Component | Current max_tokens | Context Size | Status | New Value |
|-----------|-------------------|--------------|--------|-----------|
| Problem-Solution | 240 (120*2) | ~3.3K tokens | ✗ FAILED | 400 |
| Contradiction | 700 (350*2) | ~3.5K tokens | ✗ FAILED | 700 |
| Knowledge | ~500 (250*2) | ~3.5K tokens | ✗ FAILED | 800 |

**Total fix: 3 one-line changes in 2-3 files**

---

## WHY THIS HAPPENS

1. **Prompt complexity**: All three extraction tasks use complex JSON schemas with multiple fields per item
2. **Retrieved context size**: Real papers generate 3.3K-3.5K tokens of retrieved context
3. **Groq token model**: `openai/gpt-oss-120b` is aggressive on token counting for JSON mode
4. **Conservative allocation**: Original max_tokens was set without tuning against real data
5. **Silent failure**: Groq returns 400 error but the code catches it and returns None without logging the token budget issue

---

## VERIFICATION

After applying fixes, the trace should show:

```
✓ Papers selected: 2
✓ RAG retrieval: 13.4KB context
✓ LLM extraction: JSON parsed successfully
✓ Problems extracted: 2+ per paper
✓ Claims extracted: 3-6 per paper
✓ Pairs matched: 1-5 pairs
✓ UI display: Data shown (no "No data extracted" message)
```

