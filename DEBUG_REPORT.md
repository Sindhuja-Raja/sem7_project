# DEBUG REPORT: Problem-Solution & Contradiction Detection Failures

## EXECUTIVE SUMMARY

Both **Problem-Solution Analysis** and **Contradiction Detection** are failing at the LLM JSON extraction stage with the same error: `"Failed to validate JSON. Please adjust your prompt."`

This is NOT an architectural or threshold issue - it's a **model token limit problem** combined with insufficient prompt error handling.

---

## ROOT CAUSE #1: Insufficient Token Budget for JSON Generation

### Evidence from Debug Trace:

**Problem-Solution Analysis:**
- RAG context retrieved: 6,445 + 6,973 = **13,418 characters total**
- Estimated tokens in context: ~3,000-3,500 tokens
- Max tokens allocated for response: `min(4000, 120 * 2 papers) = 240 tokens`
- **Problem**: 240 tokens is NOT ENOUGH to generate JSON with multiple problems per paper

```python
# In problem_solution_analysis.py, line ~125
max_tokens=min(3000, 120 * len(papers)),  # <-- 120 per paper = TOO LOW
```

**Contradiction Detection:**
- RAG context retrieved: 6,786 + 7,270 = **14,056 characters total**
- Estimated tokens in context: ~3,300-3,500 tokens
- Max tokens allocated for response: `min(4000, 350 * 2 papers) = 700 tokens`
- **Problem**: 700 tokens MIGHT be enough, but is marginal when full claim text must be returned

```python
# In contradiction_detection.py, line ~234
max_tokens=min(4000, 350 * len(papers)),  # <-- 350 per paper = MARGINAL
```

### Why This Causes JSON Validation Failure:

1. Model starts generating JSON response
2. Runs out of allocated tokens mid-generation (e.g., cuts off mid-claim)
3. Returns truncated/incomplete JSON
4. Groq's JSON validator rejects incomplete JSON
5. Function receives `None` and returns empty results

---

## ROOT CAUSE #2: Knowledge Extraction Failure (Cascading Effect)

The Knowledge Analysis runs FIRST and already shows all fields as "Not Found in Retrieved Context":

```
AI Technique: Not Found in Retrieved Context
AI Model: Not Found in Retrieved Context
(all 32 fields: Not Found in Retrieved Context)
```

This same LLM failure cascades to Problem-Solution because it depends on knowledge_table fields.

### Error from Knowledge Extraction:
```
"Failed to generate JSON. Please adjust your prompt. 
See 'failed_generation' for more details."
```

Same token limit issue affects all JSON extraction across the pipeline.

---

## MINIMUM FIX REQUIRED

### For Problem-Solution Analysis:
**Change max_tokens from 120 → 400 per paper**

```python
# File: problem_solution_analysis.py
# Around line 125, in _extract_problems() function

OLD:
max_tokens=min(3000, 120 * len(papers)),

NEW:
max_tokens=min(4000, 400 * len(papers)),
```

### For Contradiction Detection:
**Change max_tokens from 350 → 700 per paper**

```python
# File: contradiction_detection.py
# Around line 234, in extract_claims() function

OLD:
max_tokens=min(4000, 350 * len(papers)),

NEW:
max_tokens=min(5000, 700 * len(papers)),
```

### For Knowledge Analysis:
**Also needs adjustment** (same root cause)

```python
# File: knowledge_analysis.py
# Find the max_tokens setting in extraction calls

OLD (estimate):
max_tokens=min(3000, 250 * len(papers)),

NEW:
max_tokens=min(5000, 800 * len(papers)),
```

---

## EVIDENCE SUMMARY

### Test Run Metrics:
- Papers loaded: 2 valid (1d84ec52ed4992e4, 5464c76b2da16e07)
- RAG retrieval: ✓ Working (6.4KB + 6.9KB context)
- LLM availability: ✓ True
- JSON extraction: ✗ FAILED (token limit)

### Error Messages Observed:
1. `Failed to validate JSON. Please adjust your prompt.`
2. `Failed to generate JSON. Please adjust your prompt.`
3. `max completion tokens reached before generating a valid document`

### Data Breakdown:
| Stage | Input | Output | Status |
|-------|-------|--------|--------|
| RAG Retrieval | 2 papers | 13.4KB context | ✓ Working |
| Problem Extraction | 2 papers | 0 problems (NOT_REPORTED) | ✗ FAILED |
| Claim Extraction | 2 papers | 0 claims | ✗ FAILED |
| Knowledge Extraction | 2 papers | 32 fields = "Not Found" | ✗ FAILED |

---

## EXPLANATION

### Why Token Limit Causes This:

1. User query → retrieved chunks from RAG (~3-3.5K tokens consumed)
2. LLM starts formulating JSON response
3. Must include:
   - Opening `{"papers": [`
   - Per-paper data (index, main_problem, secondary_problems)
   - Full claim text for each claim extracted
   - Proper JSON syntax throughout
4. With only 120-350 tokens allocated, model can't output complete structure
5. JSON gets cut off (e.g., `{"papers": [{"index": 1, "claims": [{"topic": "...`, stops mid-claim)
6. Groq's JSON validator sees invalid JSON
7. Validation fails → returns empty string or error
8. `llm.parse_json()` returns None
9. Function returns empty list/dict
10. UI displays "No data extracted"

### Why RAG Works But Extraction Fails:

- RAG retrieval (`retrieve_context()`) returns plain text chunks
- Plain text doesn't need to be complete - partial context is useful
- JSON extraction MUST be complete - incomplete JSON is invalid and unparseable
- This is why retrieval shows 6.9KB but extraction returns nothing

---

## VERIFICATION

To verify this is the root cause, you can:

1. Manually increase max_tokens and re-run
2. Check Groq's actual response tokens in logs (add logging for `resp.json()`)
3. Reduce input complexity (fewer papers per call, or shorter context window)
4. Monitor token usage in Groq API dashboard

---

## WHY THIS WASN'T CAUGHT EARLIER

1. **No token usage logging**: The code doesn't log actual token usage from Groq responses
2. **Silent failures**: JSON parsing failures return `None` without visible warning in UI
3. **Config constants aren't tuned**: `max_tokens` was set conservatively without testing against real retrieved context
4. **Real data vs. test data**: The test papers have large raw text (6K+ words each), causing larger retrieved contexts than might have been tested with smaller papers

---

## SECONDARY ISSUE: No Error Logging When JSON Extraction Fails

When JSON parsing fails, the code logs:
```python
logger.warning("Claim extraction returned unusable JSON: %s", content[:200])
```

But this warning:
1. Only logs if `content` is not None
2. Only shows first 200 chars of response
3. Doesn't show the actual Groq error message
4. Doesn't suggest that token limit might be the cause

This makes debugging difficult.
