# Research Paper Retrieval System

AI-powered research paper retrieval and ranking, built with Streamlit.
No database - every search retrieves live from official academic APIs.

## Pipeline

1. **Query Understanding Agent** (`query_understanding.py`) - the LLM (Groq,
   Llama 3.3 by default) reads the raw title and extracts structured research
   intent (domain, problem, primary/secondary AI techniques, application
   domain, keywords, synonyms, excluded domains, search intent), then turns
   that into a search query tuned to each individual API's query syntax -
   see "Query Understanding Agent" below for details.
2. **Parallel retrieval** (`retrieve.py`) - OpenAlex, Crossref, arXiv,
   CORE and Semantic Scholar are queried concurrently via a thread pool,
   each with its own query string from step 1.
3. **Deduplication** (`utils.py`) - records are merged across sources
   using DOI, fuzzy title similarity, author overlap and publication year.
4. **Dense embedding ranking** (`embedding.py`) - `BAAI/bge-large-en-v1.5`
   embeds the original title and every paper's title+abstract; papers are
   ranked by cosine similarity (no keyword matching).
5. **Cross-encoder re-ranking** (`rerank.py`) - the top candidates are
   re-scored with `cross-encoder/ms-marco-MiniLM-L12-v2`, which reads the
   query and paper together for a sharper relevance judgement.
6. **LLM enrichment** (`llm.py`) - the final top-N papers each get a short
   AI summary and a one/two sentence explanation of why they're relevant.
7. **RAG indexing** (`rag_pipeline.py`) - for every displayed paper, once:
   download+clean the full PDF (or fall back to the abstract), chunk it,
   embed every chunk, and build an in-memory FAISS index - see "RAG
   Pipeline" below. This shared per-paper index is what every module below
   retrieves from instead of re-reading full PDF text.
8. **Knowledge Retrieval** (`knowledge_analysis.py`) - for every displayed
   paper, semantically retrieves only the chunks relevant to technical
   fields/gaps/future-work from that paper's RAG index and extracts ~13
   technical fields; aggregation, semantic clustering of research
   gaps/future work, and synthesized insights follow - see "Knowledge
   Retrieval" below.
9. **Problem-Solution Analysis Agent** (`problem_solution_analysis.py`) -
   synthesizes the *set* of displayed papers together (never a per-paper
   summary) into a Research Problem → AI Techniques/Models table - see
   "Problem-Solution Analysis" below.
10. **UI** (`app.py`) - Streamlit renders everything: title, authors, year,
    source, DOI, abstract, similarity score, AI relevance score, AI summary,
    AI reason, PDF link, the extracted Query Understanding JSON, the
    Retrieved Chunks transparency table, and the Knowledge Retrieval /
    Problem-Solution tables (all via `st.dataframe`, no markdown tables).

On demand, after a search, the user can also select 2+ of the displayed
papers and run **Contradiction Detection** (`contradiction_detection.py`) -
see below.

## Query Understanding Agent

Academic APIs mostly do lexical keyword matching, not intent understanding,
so sending a raw title straight to them retrieves a lot of surface-word
noise. `query_understanding.py` asks the LLM to analyze the title first and
return structured JSON (schema in `QUERY_UNDERSTANDING_SCHEMA`):

```
research_domain, research_problem, primary_ai_technique,
secondary_ai_techniques[], application_domain, related_keywords[],
synonyms[], alternative_technical_terms[], exclude_domains[],
search_intent, search_query
```

`build_source_queries()` then turns that one structured result into a
**different query string per API**, because each API parses query syntax
differently:

- **arXiv** has real boolean query grammar (`AND`/`OR` + field prefixes) -
  gets a properly built boolean query, e.g.
  `(all:"Agentic AI" OR all:"AI Agents") AND (all:"Medical Diagnosis")`.
- **CORE** is Elasticsearch-query-string based and also parses `AND`/`OR`/
  parentheses - gets the LLM's canonical boolean `search_query` directly.
- **OpenAlex, Crossref, Semantic Scholar** do relevance-ranked free-text
  search on the raw string rather than parsing boolean syntax - sending them
  literal `AND`/`OR`/parentheses would just search for those words and hurt
  relevance, so they get a flattened, deduplicated bag of the most important
  terms instead.

If `GROQ_API_KEY` isn't set (or the call fails), `understand_query()`
degrades to using the raw title as the search query for every source -
identical to the old direct-search behavior, so retrieval never breaks.

## RAG Pipeline

`rag_pipeline.py` replaces "send the whole extracted PDF to the LLM" with a
real retrieval-augmented pipeline, built **once per paper per search** and
shared by every downstream module:

```
paper.pdf_url
    -> get_full_paper_text()   pdf_extraction.py: download + PyMuPDF
                               extraction of the COMPLETE PDF (no page
                               cap) + best-effort heading-based section
                               split (References/Acknowledgements already
                               dropped) + per-section cleaning (page
                               numbers, repeated headers/footers,
                               whitespace) + on-disk cache at
                               data/papers/<paper_id>.txt
    -> validate_paper_text()    paper_validation.py: full content
                               available, minimum word count, essential
                               sections present - checked on this raw
                               sections dict, BEFORE any chunking/
                               embedding/FAISS work. paper_validation.py
                               has no dependency on rag_pipeline.py at all.
                               An INVALID paper stops here and is
                               discarded; only a VALID paper continues:
    -> chunk_sections()         sentence-boundary chunking (never splits a
                               sentence/table/equation/heading), token-sized
                               via the embedding model's own tokenizer, with
                               sentence-level overlap carried into the next
                               chunk; each Chunk keeps Paper ID/Section
                               Name/Chunk ID/Chunk Text
    -> _embed()                 BAAI/bge-small-en-v1.5 (falls back to
                               intfloat/e5-base-v2 if unreachable),
                               L2-normalized vectors
    -> _build_faiss_index()     faiss.IndexFlatIP - built in memory, per
                               paper, per search session; nothing persists
                               to disk
```

`rag_pipeline.build_paper_index_from_sections(paper, sections)` is the
chunk+embed+FAISS half of this - it takes an already-extracted,
already-validated sections dict as a plain argument and has no idea how
that text was acquired (no `pdf_extraction` import) or that it was
validated (no `paper_validation` import). The orchestration - extract,
validate, and only then index - lives in `app.py`'s `_collect_valid_papers`/
`_extract_validate_and_index`, which also implements the paper-replacement
loop: an INVALID candidate is discarded and automatically replaced by the
next-ranked candidate from the similarity/cross-encoder-ranked pool, until
`display_n` VALID papers are collected or the pool is exhausted. Every
module below - Knowledge Retrieval and Problem-Solution Analysis
automatically, Contradiction Detection/Root Cause Analysis/Inventor Agent
on demand - reuses the resulting `Dict[paper.title -> PaperIndex]`
(`st.session_state.rag_index_cache`) instead of re-parsing or re-embedding.

`retrieve_context()` is the shared retrieval call every module makes: given
a `PaperIndex` and one or more `(query, top_k)` pairs, it embeds each query
(with the model's query-side instruction prefix), searches the FAISS index,
merges + deduplicates + caps the combined results, and returns both an
assembled text blob (tagged `[Section Name] chunk text`) and the structured
`RetrievedChunk` list (section, similarity score, chunk text) for display.
Every module's retrieval query is task-specific (`RAG_QUERY_*` in
`config.py`) - e.g. Knowledge Extraction searches for "AI technique, AI
model, dataset, framework, optimizer, loss function, ...", Research Gap
Analysis searches for "limitations, discussion, conclusion, research gap,
open problems", and so on per module below.

If the retrieved chunks don't contain a field the LLM is asked to extract,
every prompt in this pipeline requires the literal string
`"Not Found in Retrieved Context"` instead of a guess - this is a shared
constant, `NOT_FOUND_IN_CONTEXT` in `rag_pipeline.py`.

**Chunk size note:** the spec target was 800-1000 tokens with 150-200 token
overlap, but `BAAI/bge-small-en-v1.5` has a hard `max_seq_length` of 512
tokens (confirmed at runtime - anything past token 512 is silently
truncated before embedding, which would make the FAISS index score against
a truncated chunk while the untruncated text still gets sent to the LLM if
retrieved). Chunk size is therefore 480 tokens with 96 token overlap
(~19%, matching the requested overlap ratio) - see the comment above
`RAG_CHUNK_SIZE_TOKENS` in `config.py` for the full evidence trail.

## Knowledge Retrieval

Per-paper extraction is owned by `knowledge_extraction_agent.py`, which
`knowledge_analysis.py` calls as a thin adapter. Retrieval is per-field-
group, not one combined query: nine separate semantic queries against that
paper's shared RAG index - `RETRIEVAL_QUERY_GROUPS` in
`knowledge_extraction_agent.py` - covering Problem/Objectives,
Technique/Model/Architecture, Dataset/Preprocessing, Training
Strategy/Optimization, Implementation, Evaluation/Results, Research Gap,
Future Work, and Limitations/Contributions, each with its own top-K
(`RAG_TOP_K_PER_FIELD_GROUP`), deduplicated and capped at
`RAG_MAX_CHUNKS_KNOWLEDGE_COMBINED`. This exists specifically so a field
like Dataset can't get crowded out of a shared top-K pool by a higher-
scoring but unrelated field. Only those retrieved chunks - never the full
PDF - go to one Groq call per paper (run concurrently) to extract the 30
fields declared in `knowledge_extraction_agent.KNOWLEDGE_FIELDS` (Research
Domain, Objectives, AI Technique/Model/Architecture, Dataset,
Preprocessing, Training Strategy, Evaluation Metrics, Research Gap, Future
Work, Key Contributions, and more). Missing fields are `"Not Found in
Retrieved Context"`, never invented.

`knowledge_analysis.py` then produces five outputs (JSON-serializable lists
of flat dicts, rendered via `pd.DataFrame` + `st.dataframe`), with a
**Retrieved Chunks** table (Paper Title / Section Name / Similarity Score /
Chunk Preview) rendered in the UI immediately *before* the Knowledge
Retrieval table itself, so every extracted field is traceable back to the
exact passage it came from:
- `overall_analysis` - one row per category (AI Technique, AI Model, Dataset,
  Framework, Evaluation Metric, Programming Language, Hardware, Research
  Domain), each carrying the FULL frequency breakdown of every distinct value
  extracted across the analyzed papers (e.g. "Agentic AI (4 papers)",
  "Multi-Agent System (3 papers)"), computed by deterministic Python counting
  - every extracted value is listed even when nothing repeats, never
  collapsed to a single winner or hidden behind "No Common Pattern Found".
- `research_gap_analysis` / `future_work_analysis` - research-gap and
  future-work statements are first clustered by dense embedding similarity
  (reusing `embedding.py`'s cached BGE model, greedy clustering against each
  cluster's running centroid) so genuinely-same-theme statements group
  together without conflating merely-same-domain ones, then one batched LLM
  call per table paraphrases each cluster into a concise label (+ a
  practical recommendation for gaps) - "Papers Identified"/"Frequency" come
  from the clustering itself, never guessed by the LLM.
- `research_insights` - exactly 8 fixed categories (Research Trend, Emerging
  AI Technique, Common Limitation, Most Promising Research Direction,
  Recommended AI Model/Dataset/Framework, Novel Research Opportunity),
  generated from the three outputs above rather than a fresh re-read of the
  papers, so insights stay grounded in already-extracted data.

## Problem-Solution Analysis

`problem_solution_analysis.py` runs automatically right after Knowledge
Retrieval, in the same search pipeline (not on-demand). It performs
literature *synthesis* - never a per-paper summary:

```
displayed papers
    -> extract each paper's main + secondary research problem(s) - one
       batched LLM call using chunks retrieved from each paper's shared
       RAG index (`RAG_QUERY_PROBLEM_SOLUTION`: "problem statement,
       research problem, methodology, results" - reusing the same index
       Knowledge Retrieval built, no re-fetch/re-embed); the ONLY new
       extraction this module does - AI Technique/AI Model/best-reported
       performance are reused directly from knowledge_analysis.py's
       already-computed knowledge_table, at zero extra fetch/LLM cost
    -> cluster_texts()  semantically merge similar problem phrases
       ("High Latency" + "High Response Time" -> one row), reusing the
       same embedding-based clustering as Knowledge Retrieval's gap/
       future-work tables (empirically retuned for short 2-5 word
       phrases - see PROBLEM_CLUSTER_SIMILARITY_THRESHOLD in config.py)
    -> one small batched LLM call gives each cluster a clean canonical
       name, then AI Techniques/Models/best performance/paper count are
       aggregated deterministically per cluster from knowledge_table
```

Output: `problem_solution_table` (Research Problem / Frequency / AI
Techniques Used / AI Models Used / Best Reported Performance / Supporting
Papers) plus a `summary` (Most Common Research Problem, Most Used AI
Technique/Model - reused from Knowledge Retrieval's own aggregation,
Problem With Most Proposed Solutions, Problem That Remains Least Solved -
the latter two ranked by count of distinct AI techniques/models applied to
that problem, not just paper count). The output is intentionally shaped so
a future "Inventor Agent" could consume `problem_solution_table` directly
to spot over- vs under-solved problems and recommend research directions.

## Contradiction Detection

`contradiction_detection.py` runs **on demand** - after a search, the user
selects 2+ of the displayed papers and clicks "Detect Contradictions". It
never runs automatically and never touches the Knowledge Retrieval tables.

```
selected papers
    -> retrieve_claim_evidence()   retrieves chunks from each paper's
                             already-built shared RAG index
                             (`RAG_QUERY_CONTRADICTION`: "claims,
                             experimental results, evaluation, discussion,
                             findings") - no new PDF fetch/parse, reuses
                             app.py's validation-gated collection loop's output
    -> extract_claims()     one batched LLM call extracts scientific claims
                             per paper, from retrieved chunks only (method,
                             finding, performance claim, conclusion,
                             advantage, limitation) - never
                             authors/affiliations/references
    -> match_claim_pairs()  dense embedding similarity (reusing
                             embedding.py's cached BGE model) keeps only
                             cross-paper claim pairs above a similarity
                             threshold - a real filter, not just a prompt
                             instruction, so e.g. a latency claim is never
                             sent to the LLM paired with an energy claim
    -> classify_pairs()     one batched LLM call classifies each matched
                             pair into exactly one of Agreement /
                             Contradiction / Partial Contradiction /
                             Different Context / Insufficient Evidence,
                             with a confidence score, reason, and grounded
                             supporting evidence (quotes from the given
                             text - the prompt explicitly forbids inventing
                             section/page numbers)
```

The UI shows a comparison table (Paper A / Paper B / Compared Topic /
Classification / Confidence / Reason, plus the Root Cause Analysis columns
below) for every matched pair, a detail table (adding Claim A / Claim B /
Supporting Evidence) filtered to Contradiction + Partial Contradiction
rows, and a summary metric row with the count per classification.

### Root Cause Analysis

`root_cause_analysis.py` is a read-only, additive layer on top of
Contradiction Detection - it never modifies `contradiction_detection.py`
or its output, it only reads the `ClaimComparison` rows `classify_pairs()`
already produced. It runs automatically, in the same "Detect
Contradictions" click, immediately after classification - but does no work
at all (no fetch, no LLM call) unless at least one pair was classified
Contradiction or Partial Contradiction.

For each such pair, it retrieves evidence chunks (`RAG_QUERY_ROOT_CAUSE`:
"methodology, experimental setup, results, discussion") from the two
papers' already-built shared RAG indexes - deduplicated, so a paper
referenced by several contradictions is only retrieved once, and no new
PDF fetch happens - then one batched LLM call compares Research Objective,
Problem Statement, Application Domain, Dataset, Data Size, Experimental
Setup, AI Technique, AI Model, Framework, Hyperparameters, Training
Strategy, Optimizer, Loss Function, Evaluation Metrics, Hardware,
Cloud/Edge Environment, Baseline Methods and Publication Year, and picks
exactly one root cause category from a fixed list (or an honest
`"Root Cause Could Not Be Determined"` / `"Insufficient Evidence to
Determine Root Cause"` instead of forcing one). Confidence is explicitly
instructed to drop when either paper's evidence is abstract-only rather
than full PDF text.

`app.py` appends the result as three extra columns - **Root Cause**,
**Root Cause Explanation**, **Root Cause Confidence** (named to avoid
colliding with the existing Confidence column) - onto the same
Contradiction Comparison Table; non-contradiction rows (Agreement/
Different Context/Insufficient Evidence) get `"Not Applicable"` rather than
a fabricated root cause.

## Inventor Agent

`inventor_agent.py` is the last on-demand step - its own button, positioned
after Contradiction Detection/Root Cause Analysis - since it synthesizes
across every other module's *already-computed* output rather than reading
papers or fetching PDFs itself. It performs almost no new extraction:
Knowledge Retrieval's `knowledge_table`/`overall_analysis`/
`research_gap_analysis`/`future_work_analysis`, Problem-Solution Analysis's
table/summary, and (if the user ran it) the Contradiction Comparison Table
with its Root Cause columns are condensed into one JSON context. The only
new work is a light supplementary retrieval - one top chunk per paper from
each paper's already-built shared RAG index (`RAG_QUERY_INVENTOR`: "best
results, highest accuracy, datasets, models, future work, research gaps")
- added as extra grounding alongside the structured summaries, not as
primary context. All of this is sent to the LLM in a single call.
Contradiction data is optional - if the user never ran Contradiction
Detection, that step of the reasoning is simply skipped rather than
fabricated.

It is explicitly not a per-paper summarizer and not a random idea
generator: every recommendation must combine a strength identified in the
literature with a fix for a weakness also identified in the literature,
and must cite real paper titles - `_validate_recommendations()` cross-checks
every cited title against the actual retrieved papers and silently drops
any that don't match, as a code-level backstop beyond the prompt's own
"never invent a paper title" instruction. Recommending a technique/model
that appears nowhere in the retrieved data is only allowed when justified
as a logical extension of a gap/future-work theme the data itself
surfaces. If the available data is too thin, it returns
`"Insufficient evidence to recommend an improved solution."` instead of
forcing one.

Output: a 5-column recommendation table (Current Approach / Suggested
Improvement / Why this is Better / Expected Benefit / Supporting Papers)
plus a 9-row Final Research Recommendation (technique, model, dataset,
framework, evaluation metrics, experimental setup, suggested research
title, expected advantages, possible risks).

## Project structure

```
app.py                     Streamlit UI - wires the whole pipeline together
query_understanding.py     LLM query understanding + per-source query builder
retrieve.py                One function per academic API + parallel fan-out
embedding.py                BAAI/bge-large-en-v1.5 embeddings + cosine similarity + shared semantic clustering (ranking, gap/future-work/problem clustering, claim matching)
rerank.py                  cross-encoder/ms-marco-MiniLM-L12-v2 re-ranking
llm.py                     Groq calls with 429 retry/backoff, shared JSON parsing, paper summaries/relevance reasons
pdf_extraction.py          Best-effort full-PDF download + section-aware parsing (References/Acknowledgements dropped) - low-level layer used only by rag_pipeline.py
rag_pipeline.py            RAG core: text cleaning, sentence-aware chunking, BAAI/bge-small-en-v1.5 embedding, FAISS indexing, task-specific semantic retrieval - shared by every module below
knowledge_analysis.py      Per-paper deep extraction (from RAG-retrieved chunks only) + aggregation + semantic clustering + insights
problem_solution_analysis.py  Literature synthesis: common research problems -> AI techniques/models used, reusing knowledge_table
contradiction_detection.py On-demand claim extraction, topic matching, contradiction classification
root_cause_analysis.py     Read-only add-on: root cause of each detected contradiction, evidence-grounded
inventor_agent.py          On-demand: synthesizes every other module's output into an evidence-based improved-solution recommendation
utils.py                   Paper data model + duplicate detection/merging
config.py                  API keys, model names, tunable constants (reads .env)
requirements.txt
.env.example
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in whichever keys you have
streamlit run app.py
```

### API keys

| Source            | Key required?                                   |
|--------------------|--------------------------------------------------|
| OpenAlex           | No                                                |
| Crossref           | No                                                |
| arXiv              | No                                                |
| CORE               | Yes - free at https://core.ac.uk/services/api - source is skipped if missing |
| Semantic Scholar   | Optional - improves rate limits                  |
| Groq (LLM)         | Yes for query understanding/summaries/reasons - free at https://console.groq.com/keys - app still runs without it, falling back to raw-title search and no AI summaries/reasons |

The first run downloads the embedding and cross-encoder models from
Hugging Face (a few GB total) and caches them via `st.cache_resource`,
so subsequent searches in the same session are fast.
