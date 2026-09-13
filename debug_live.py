"""
Live debugging with REAL PAPER DATA from data/papers/
Traces both Problem-Solution Analysis and Contradiction Detection pipelines
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config
import llm
from contradiction_detection import (
    extract_claims, match_claim_pairs, predict_contradiction_pairs, classify_pairs,
    retrieve_claim_evidence
)
from embedding import cluster_texts
from paper_validation import validate_paper_text, VALID
from problem_solution_analysis import (
    _extract_problems, _retrieve_problem_evidence, _build_problem_solution_table,
    NOT_REPORTED
)
from rag_pipeline import PaperIndex, build_paper_index_from_sections
from knowledge_analysis import analyze_papers
from utils import Paper

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER: Load real papers from data/papers/ and build RAG indices
# ============================================================================

def load_and_prepare_papers(limit: int = 3) -> tuple[List[Paper], Dict[str, PaperIndex]]:
    """Load actual paper texts, validate, build RAG indices."""
    
    print("\n" + "="*80)
    print(f"LOADING {limit} REAL PAPERS FROM data/papers/")
    print("="*80)
    
    papers_dir = Path("data/papers")
    paper_files = sorted(papers_dir.glob("*.txt"))[:limit]
    
    rag_cache = {}
    valid_papers = []
    
    for file_path in paper_files:
        # Create a Paper object with minimal info (file is the "full text")
        paper = Paper(
            title=file_path.stem,
            abstract=f"Paper from {file_path.name}",
            authors=[],
            year=2024,
            source="local_file",
            doi="",
            pdf_url=f"file://{file_path}",
        )
        
        # Read text with fallback encoding
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        if not text or len(text) < 100:
            print(f"  ✗ {paper.title}: File too small or empty")
            continue
        
        sections = {"Body": text}
        
        # Validate
        validation = validate_paper_text(paper, sections)
        if validation.status != VALID:
            print(f"  ✗ {paper.title}: {validation.reasons}")
            continue
        
        # Build RAG index
        print(f"  ✓ {paper.title[:40]}: {len(text)} chars, {validation.word_count} words")
        paper_index = build_paper_index_from_sections(paper, sections)
        rag_cache[paper.title] = paper_index
        valid_papers.append(paper)
    
    print(f"\n✓ RAG cache built for {len(rag_cache)} valid papers")
    return valid_papers, rag_cache


# ============================================================================
# DEBUG: PROBLEM-SOLUTION ANALYSIS
# ============================================================================

def debug_problem_solution(papers: List[Paper], rag_cache: Dict[str, PaperIndex]):
    """Full trace of Problem-Solution Analysis pipeline."""
    
    print("\n" + "="*80)
    print("PROBLEM-SOLUTION ANALYSIS - FULL TRACE")
    print("="*80)
    
    print(f"\n[1] Input: {len(papers)} papers")
    for p in papers:
        print(f"    - {p.title}")
    
    # First, extract knowledge (needed by problem-solution analysis)
    print(f"\n[2] Running Knowledge Analysis...")
    print(f"    Config: max_papers={config.TOP_K_AFTER_EMBEDDING}")
    
    knowledge = analyze_papers(papers, rag_cache, max_papers=len(papers))
    
    print(f"    ✓ Knowledge table: {len(knowledge.knowledge_table)} rows")
    if knowledge.knowledge_table:
        print(f"      Columns: {list(knowledge.knowledge_table[0].keys())}")
        print(f"      First row keys: {knowledge.knowledge_table[0]}")
        for row in knowledge.knowledge_table[:2]:
            print(f"      - Title: {row.get('Paper Title', 'N/A')[:40]}")
            print(f"        AI Technique: {row.get('AI Technique', NOT_REPORTED)}")
            print(f"        AI Model: {row.get('AI Model', NOT_REPORTED)}")
    
    print(f"    ✓ Overall analysis: {len(knowledge.overall_analysis)} rows")
    
    # Problem extraction
    print(f"\n[3] Problem Extraction via RAG + LLM")
    print(f"    Query: {config.RAG_QUERY_PROBLEM_SOLUTION}")
    print(f"    Top K: {config.RAG_TOP_K_PROBLEM_SOLUTION}")
    
    # Retrieve evidence
    contexts = _retrieve_problem_evidence(papers, rag_cache)
    print(f"    Retrieved contexts:")
    for i, ctx in contexts.items():
        print(f"      Paper {i}: {len(ctx)} chars")
    
    # Extract problems
    print(f"\n    LLM extraction:")
    print(f"    Available: {llm.is_available()}")
    
    paper_problems = _extract_problems(papers, rag_cache)
    print(f"    Extracted {len(paper_problems)} problem sets:")
    for i, pp in enumerate(paper_problems):
        has_main = pp.main_problem != NOT_REPORTED
        print(f"      Paper {i}: main={has_main} ('{pp.main_problem[:30]}...'), " + 
              f"secondary={len(pp.secondary_problems)}")
        if pp.secondary_problems:
            for sp in pp.secondary_problems:
                print(f"        - {sp}")
    
    # Check if we have any problems
    valid_count = sum(1 for pp in paper_problems if pp.main_problem != NOT_REPORTED)
    if valid_count == 0:
        print("\n    ⚠️  NO PROBLEMS EXTRACTED!")
        print("       This is where the data path breaks.")
        return
    
    # Build problem-solution table
    print(f"\n[4] Building Problem-Solution Table")
    print(f"    Similarity threshold: {config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD}")
    
    # Collect items
    items = []
    for paper, pp in zip(papers, paper_problems):
        if pp.main_problem != NOT_REPORTED:
            items.append((paper.title, pp.main_problem))
        for secondary in pp.secondary_problems:
            items.append((paper.title, secondary))
    
    print(f"    Items collected: {len(items)}")
    
    # Cluster
    clusters = cluster_texts(items, config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD)
    print(f"    Clusters created: {len(clusters)}")
    
    if len(clusters) == 0:
        print("\n    ⚠️  NO CLUSTERS CREATED!")
        print("       Clustering threshold may be too strict.")
        return
    
    # Build table
    table = _build_problem_solution_table(papers, paper_problems, knowledge.knowledge_table)
    print(f"    Final table rows: {len(table)}")
    
    if table:
        print(f"    Table preview:")
        for row in table[:2]:
            print(f"      - Problem: {row.get('Research Problem')}")
            print(f"        Frequency: {row.get('Frequency')}")
            print(f"        Techniques: {row.get('AI Techniques Used', NOT_REPORTED)[:50]}")
    else:
        print(f"\n    ⚠️  TABLE IS EMPTY!")
        print(f"       But we had {len(clusters)} clusters...")
    
    return table, knowledge


# ============================================================================
# DEBUG: CONTRADICTION DETECTION
# ============================================================================

def debug_contradiction_detection(papers: List[Paper], paper_labels: List[str],
                                  rag_cache: Dict[str, PaperIndex]):
    """Full trace of Contradiction Detection pipeline."""
    
    print("\n" + "="*80)
    print("CONTRADICTION DETECTION - FULL TRACE")
    print("="*80)
    
    if len(papers) < 2:
        print("⚠️  Need at least 2 papers for contradiction detection")
        return
    
    print(f"\n[1] Input: {len(papers)} papers")
    for label in paper_labels:
        print(f"    - {label}")
    
    # Retrieve claim evidence
    print(f"\n[2] Retrieve Claim Evidence via RAG")
    print(f"    Query: {config.RAG_QUERY_CONTRADICTION}")
    print(f"    Top K: {config.RAG_TOP_K_CONTRADICTION}")
    
    contexts = retrieve_claim_evidence(papers, rag_cache)
    print(f"    Retrieved contexts:")
    total_chars = 0
    for i, ctx in contexts.items():
        chars = len(ctx)
        total_chars += chars
        print(f"      Paper {i}: {chars} chars")
    
    if total_chars == 0:
        print("\n    ⚠️  NO CONTEXT RETRIEVED!")
        return
    
    # Extract claims
    print(f"\n[3] Extract Claims via LLM")
    print(f"    Max per paper: {config.CONTRADICTION_MAX_CLAIMS_PER_PAPER}")
    print(f"    Available: {llm.is_available()}")
    
    claims_by_paper = extract_claims(papers, paper_labels, rag_cache)
    total_claims = sum(len(c) for c in claims_by_paper.values())
    
    print(f"    Extracted claims:")
    for i, claims in claims_by_paper.items():
        print(f"      Paper {i}: {len(claims)} claims")
        for claim in claims[:2]:
            print(f"        - [{claim.claim_type}] {claim.topic}: {claim.claim_text[:50]}...")
    
    if total_claims == 0:
        print("\n    ⚠️  NO CLAIMS EXTRACTED!")
        return
    
    # Match pairs
    print(f"\n[4] Match Claim Pairs by Similarity")
    print(f"    Threshold: {config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD}")
    print(f"    Max pairs: {config.CONTRADICTION_MAX_CLAIM_PAIRS}")
    
    matched_pairs = match_claim_pairs(claims_by_paper)
    print(f"    Matched pairs: {len(matched_pairs)}")
    
    if matched_pairs:
        for i, (a, b, sim) in enumerate(matched_pairs[:3]):
            print(f"      Pair {i+1} (sim={sim:.4f}):")
            print(f"        A: [{a.topic}] {a.claim_text[:50]}...")
            print(f"        B: [{b.topic}] {b.claim_text[:50]}...")
    else:
        print("\n    ⚠️  NO PAIRS MATCHED!")
        
        # Analyze similarity distribution
        print(f"\n    Analyzing similarity scores...")
        import numpy as np
        from embedding import embed_texts
        
        flat = [(i, c) for i, claims in claims_by_paper.items() for c in claims]
        if len(flat) >= 2:
            vecs = embed_texts([c.claim_text for _, c in flat], is_query=False)
            
            similarities = []
            for i in range(len(flat)):
                idx_i, _ = flat[i]
                for j in range(i + 1, len(flat)):
                    idx_j, _ = flat[j]
                    if idx_i != idx_j:
                        sim = float(np.dot(vecs[i], vecs[j]))
                        similarities.append(sim)
            
            if similarities:
                print(f"    Similarity distribution (cross-paper only):")
                print(f"      Min: {min(similarities):.4f}")
                print(f"      Avg: {sum(similarities)/len(similarities):.4f}")
                print(f"      Max: {max(similarities):.4f}")
                print(f"      Threshold: {config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD}")
                print(f"      Count >= threshold: {len([s for s in similarities if s >= config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD])}")
        return
    
    # Predict contradictions
    print(f"\n[5] Predict Contradictions")
    predictions = predict_contradiction_pairs(matched_pairs)
    print(f"    Predictions:")
    for pred in predictions[:3]:
        print(f"      {pred['prediction']} (score {pred['score']})")
    
    # Classify
    print(f"\n[6] Classify Pairs via LLM")
    comparisons = classify_pairs(matched_pairs)
    print(f"    Classifications:")
    
    summary = {}
    for comp in comparisons:
        cls = comp.classification
        summary[cls] = summary.get(cls, 0) + 1
    
    for cls, count in summary.items():
        print(f"      {cls}: {count}")
    
    return matched_pairs, comparisons


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # Load papers and build RAG indices
        papers, rag_cache = load_and_prepare_papers(limit=3)
        
        if len(papers) == 0:
            print("\n❌ No valid papers loaded!")
            exit(1)
        
        # Debug Problem-Solution Analysis
        ps_result = debug_problem_solution(papers, rag_cache)
        
        # Debug Contradiction Detection
        if len(papers) >= 2:
            paper_labels = [f"Paper {i+1}: {p.title}" for i, p in enumerate(papers)]
            debug_contradiction_detection(papers, paper_labels, rag_cache)
        
        print("\n" + "="*80)
        print("✓ DEBUG TRACE COMPLETE")
        print("="*80)
    
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
