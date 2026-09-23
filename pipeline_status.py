"""
Shared result-status vocabulary for every analysis module (Problem-
Solution, Contradiction Detection, ...).

Before this, a stage that produced no rows always collapsed to the same
generic UI message ("No data extracted for this section.", "No
comparable claim pairs were found across the selected papers.") whether
the real cause was:
    - the daily Groq quota being exhausted (an external, temporary
      condition - retrying later fixes it)
    - RAG genuinely retrieving nothing (a paper with no usable text)
    - the LLM call failing/returning unparsable output (a real bug worth
      investigating)
    - or the pipeline running perfectly and genuinely finding nothing to
      report (a valid scientific result, not a failure)

Those are four completely different situations requiring four different
responses from a user reading the UI - conflating them into one message
is what made this class of bug so hard to diagnose from the outside.
Every analysis entry point now returns one of these alongside its
results, so the UI (and logs) can say exactly what happened.
"""

SUCCESS = "success"                        # ran fully; rows/pairs reflect real findings
INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # ran fine end-to-end, genuinely nothing to report -
                                                  # a valid result, not a failure
LLM_UNAVAILABLE = "llm_unavailable"        # GROQ_API_KEY not configured
DAILY_QUOTA_EXCEEDED = "daily_quota_exceeded"  # this account's daily Groq token budget is exhausted
RETRIEVAL_FAILURE = "retrieval_failure"    # RAG returned no usable chunks for any selected paper
EXTRACTION_FAILURE = "extraction_failure"  # the LLM call failed or returned unparsable output
                                            # despite good input - a real bug, not a data limitation
INVALID_INPUT = "invalid_input"            # no papers (or no valid papers) were given at all

# Human-readable text for each status, safe to show directly in the UI.
MESSAGES = {
    SUCCESS: "",
    INSUFFICIENT_EVIDENCE: "Related evidence was found, but not enough to report a confident result here. "
                            "This is a valid outcome, not a failure.",
    LLM_UNAVAILABLE: "GROQ_API_KEY is not configured - this section requires the LLM and cannot run.",
    DAILY_QUOTA_EXCEEDED: "The daily Groq token quota is exhausted for today. This section will populate "
                          "once the quota resets - no data was lost, and already-cached results are unaffected.",
    RETRIEVAL_FAILURE: "No relevant text could be retrieved from the selected paper(s) - they may only have "
                        "an abstract available, or their PDF could not be downloaded.",
    EXTRACTION_FAILURE: "The AI extraction step failed or returned an unusable response. This is worth "
                         "investigating further - check the application logs.",
    INVALID_INPUT: "No valid papers were provided to analyze.",
}
