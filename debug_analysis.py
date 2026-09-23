"""
Debug script to trace Problem-Solution Analysis and Contradiction Detection pipelines.
Run this AFTER a search has been performed in the UI to use the session state data.
"""

import json
import logging
from typing import Dict, List

import config
import llm
from contradiction_detection import (
    extract_claims, match_claim_pairs, predict_contradiction_pairs, classify_pairs,
    retrieve_claim_evidence
)
from embedding import embed_texts, cluster_texts
from problem_solution_analysis import (
    _extract_problems, _retrieve_problem_evidence, _build_problem_solution_table,
    analyze_problems_and_solutions, NOT_REPORTED
)
from rag_pipeline import PaperIndex
from utils import Paper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================================================
# PROBLEM-SOLUTION ANALYSIS DEBUG
# ============================================================================

def debug_problem_solution_analysis(
    papers: List[Paper], knowledge_table: List[dict], overall_analysis: List[dict],
    rag_cache: Dict[str, PaperIndex],
):
    """Trace Problem-Solution Analysis with detailed output at each stage."""
    
    print("\n" + "="*80)
    print("PROBLEM-SOLUTION ANALYSIS DEBUG")
    print("="*80)
    
    # Step 0: Input validation
    print(f"\n[STEP 0] Input Validation")
    print(f"  Papers selected: {len(papers)}")
    for i, p in enumerate(papers):
        print(f"    Paper {i+1}: {p.title[:60]}")
    
    print(f"  Knowledge table rows: {len(knowledge_table)}")
    if knowledge_table:
        print(f"    Columns: {list(knowledge_table[0].keys())}")
        print(f"    First row keys: {knowledge_table[0]}")
    
    print(f"  Overall analysis rows: {len(overall_analysis)}")
    if overall_analysis:
        print(f"    Columns: {list(overall_analysis[0].keys())}")
    
    print(f"  RAG cache papers: {len(rag_cache)}")
    
    # Step 1: Retrieve problem evidence
    print(f"\n[STEP 1] Retrieve Problem Evidence via RAG")
    print(f"  Query: {config.RAG_QUERY_PROBLEM_SOLUTION}")
    print(f"  Top K: {config.RAG_TOP_K_PROBLEM_SOLUTION}")
    
    contexts = _retrieve_problem_evidence(papers, rag_cache)
    for i, context in contexts.items():
        context_preview = context[:200].replace('\n', ' ')[:100] if context else "(empty)"
        print(f"    Paper {i}: {len(context)} chars - {context_preview}...")
    
    total_context_chars = sum(len(c) for c in contexts.values())
    print(f"  Total context retrieved: {total_context_chars} chars")
    if total_context_chars == 0:
        print("  ⚠️  NO CONTEXT RETRIEVED - this is the problem!")
        return
    
    # Step 2: Extract problems
    print(f"\n[STEP 2] Extract Problems via LLM")
    print(f"  LLM available: {llm.is_available()}")
    
    paper_problems = _extract_problems(papers, rag_cache)
    for i, pp in enumerate(paper_problems):
        print(f"    Paper {i+1}: main='{pp.main_problem}', secondary={pp.secondary_problems}")
    
    valid_problems = [pp for pp in paper_problems if pp.main_problem != NOT_REPORTED]
    print(f"  Valid problems extracted: {len(valid_problems)}/{len(paper_problems)}")
    
    if len(valid_problems) == 0:
        print("  ⚠️  NO PROBLEMS EXTRACTED - this is the problem!")
        return
    
    # Step 3: Build problem-solution table
    print(f"\n[STEP 3] Build Problem-Solution Table")
    
    # Collect items
    items = []
    for paper, pp in zip(papers, paper_problems):
        if pp.main_problem != NOT_REPORTED:
            items.append((paper.title, pp.main_problem))
        for secondary in pp.secondary_problems:
            items.append((paper.title, secondary))
    
    print(f"  Total problem items (main + secondary): {len(items)}")
    for title, problem in items[:5]:
        print(f"    - {title[:40]}: {problem}")
    
    # Clustering
    print(f"\n[STEP 3.1] Semantic Clustering of Problems")
    print(f"  Similarity threshold: {config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD}")
    
    clusters = cluster_texts(items, config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD)
    print(f"  Clusters created: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"    Cluster {i+1}: {len(cluster)} items")
        for title, text in cluster[:2]:
            print(f"      - {text}")
    
    if len(clusters) == 0:
        print("  ⚠️  NO CLUSTERS CREATED - this is the problem!")
        return
    
    # Build table
    print(f"\n[STEP 3.2] Build Table from Clusters")
    table = _build_problem_solution_table(papers, paper_problems, knowledge_table)
    print(f"  Table rows generated: {len(table)}")
    
    if table:
        for i, row in enumerate(table[:3]):
            print(f"    Row {i+1}:")
            print(f"      Research Problem: {row.get('Research Problem')}")
            print(f"      Frequency: {row.get('Frequency')}")
            print(f"      AI Techniques: {row.get('AI Techniques Used', NOT_REPORTED)[:60]}")
            print(f"      AI Models: {row.get('AI Models Used', NOT_REPORTED)[:60]}")
    else:
        print("  ⚠️  TABLE IS EMPTY - this is the problem!")
    
    # Step 4: Full analysis
    print(f"\n[STEP 4] Full Analysis Call")
    result = analyze_problems_and_solutions(papers, knowledge_table, overall_analysis, rag_cache)
    print(f"  Problem-solution table: {len(result.problem_solution_table)} rows")
    print(f"  Summary: {len(result.summary)} rows")
    
    print(f"\n[FINAL RESULT]")
    print(f"  problem_solution_table length: {len(result.problem_solution_table)}")
    if not result.problem_solution_table:
        print("  ⚠️  EMPTY TABLE - THIS IS WHY UI SHOWS 'No data extracted'")
    
    return result


# ============================================================================
# CONTRADICTION DETECTION DEBUG
# ============================================================================

def debug_contradiction_detection(
    papers: List[Paper], paper_labels: List[str],
    rag_cache: Dict[str, PaperIndex],
):
    """Trace Contradiction Detection with detailed output at each stage."""
    
    print("\n" + "="*80)
    print("CONTRADICTION DETECTION DEBUG")
    print("="*80)
    
    # Step 0: Input validation
    print(f"\n[STEP 0] Input Validation")
    print(f"  Papers selected: {len(papers)}")
    for i, label in enumerate(paper_labels):
        print(f"    {i+1}. {label[:70]}")
    
    print(f"  LLM available: {llm.is_available()}")
    print(f"  RAG cache papers: {len(rag_cache)}")
    
    if len(papers) < 2:
        print("  ⚠️  NEED AT LEAST 2 PAPERS")
        return
    
    # Step 1: Retrieve claim evidence
    print(f"\n[STEP 1] Retrieve Claim Evidence via RAG")
    print(f"  Query: {config.RAG_QUERY_CONTRADICTION}")
    print(f"  Top K: {config.RAG_TOP_K_CONTRADICTION}")
    
    contexts = retrieve_claim_evidence(papers, rag_cache)
    for i, context in contexts.items():
        context_preview = context[:200].replace('\n', ' ')[:100] if context else "(empty)"
        context_len = len(context)
        print(f"    Paper {i} ({paper_labels[i][:40]}): {context_len} chars")
        if context:
            print(f"      Preview: {context_preview}...")
    
    total_context_chars = sum(len(c) for c in contexts.values())
    print(f"  Total context retrieved: {total_context_chars} chars")
    
    # Step 2: Extract claims
    print(f"\n[STEP 2] Extract Claims via LLM")
    print(f"  Max claims per paper: {config.CONTRADICTION_MAX_CLAIMS_PER_PAPER}")
    
    claims_by_paper = extract_claims(papers, paper_labels, rag_cache)
    total_claims = sum(len(c) for c in claims_by_paper.values())
    
    print(f"  Total claims extracted: {total_claims}")
    for i, claims in claims_by_paper.items():
        print(f"    Paper {i} ({paper_labels[i][:40]}): {len(claims)} claims")
        for j, claim in enumerate(claims[:3]):
            print(f"      Claim {j+1}: topic='{claim.topic}', type='{claim.claim_type}'")
            print(f"                text='{claim.claim_text[:60]}'")
    
    if total_claims == 0:
        print("  ⚠️  NO CLAIMS EXTRACTED - this is the problem!")
        return
    
    # Step 3: Match claim pairs
    print(f"\n[STEP 3] Match Claim Pairs by Similarity")
    print(f"  Similarity threshold: {config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD}")
    print(f"  Max pairs: {config.CONTRADICTION_MAX_CLAIM_PAIRS}")
    
    # Collect flat list for analysis
    flat = [(i, c) for i, claims in claims_by_paper.items() for c in claims]
    print(f"  Total flat claims: {len(flat)}")
    
    # Show embedding info
    if flat:
        vecs = embed_texts([c.claim_text for _, c in flat], is_query=False)
        print(f"  Embeddings generated: {len(vecs)} vectors")
        if vecs:
            print(f"    Vector shape: {vecs[0].shape}")
    
    # Find all candidate pairs (before filtering)
    candidate_pairs_count = 0
    same_paper_pairs_count = 0
    
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            idx_i, _ = flat[i]
            idx_j, _ = flat[j]
            candidate_pairs_count += 1
            if idx_i == idx_j:
                same_paper_pairs_count += 1
    
    print(f"  Candidate pairs (all combinations): {candidate_pairs_count}")
    print(f"    Same paper (filtered out): {same_paper_pairs_count}")
    print(f"    Cross-paper candidates: {candidate_pairs_count - same_paper_pairs_count}")
    
    # Match pairs
    matched_pairs = match_claim_pairs(claims_by_paper)
    print(f"  Pairs after similarity filtering: {len(matched_pairs)}")
    
    if matched_pairs:
        for idx, (a, b, sim) in enumerate(matched_pairs[:3]):
            print(f"    Pair {idx+1} (similarity={sim:.4f}):")
            print(f"      A ({a.topic}): {a.claim_text[:60]}")
            print(f"      B ({b.topic}): {b.claim_text[:60]}")
    else:
        print("  ⚠️  NO PAIRS MATCHED AFTER SIMILARITY FILTERING - THIS IS THE PROBLEM!")
        # Analyze why
        if flat:
            import numpy as np
            vecs = embed_texts([c.claim_text for _, c in flat], is_query=False)
            max_sim = 0.0
            min_sim = 1.0
            similarities = []
            
            for i in range(len(flat)):
                idx_i, _ = flat[i]
                for j in range(i + 1, len(flat)):
                    idx_j, _ = flat[j]
                    if idx_i != idx_j:
                        sim = float(np.dot(vecs[i], vecs[j]))
                        similarities.append(sim)
                        max_sim = max(max_sim, sim)
                        min_sim = min(min_sim, sim)
            
            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                print(f"\n  Similarity Analysis:")
                print(f"    Cross-paper similarity stats:")
                print(f"      Min: {min_sim:.4f}")
                print(f"      Avg: {avg_sim:.4f}")
                print(f"      Max: {max_sim:.4f}")
                print(f"      Threshold: {config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD}")
                if max_sim < config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD:
                    print(f"    ⚠️  MAX SIMILARITY ({max_sim:.4f}) < THRESHOLD ({config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD})")
                    print(f"        This threshold is TOO STRICT for this dataset!")
    
    if matched_pairs:
        # Step 4: Predict contradictions
        print(f"\n[STEP 4] Predict Contradictions (Rule-based)")
        predictions = predict_contradiction_pairs(matched_pairs)
        high_risk = [p for p in predictions if "Contradiction" in p.get("prediction", "")]
        print(f"  Predictions: {len(predictions)}")
        print(f"    High risk (Likely/Possible): {len(high_risk)}")
        
        # Step 5: Classify pairs
        print(f"\n[STEP 5] Classify Pairs via LLM")
        comparisons = classify_pairs(matched_pairs)
        contradictions = [c for c in comparisons if c.classification in ("Contradiction", "Partial Contradiction")]
        print(f"  Classifications: {len(comparisons)}")
        print(f"    Contradictions/Partial: {len(contradictions)}")
        
        for i, comp in enumerate(comparisons[:3]):
            print(f"    Pair {i+1}: {comp.classification} (confidence {comp.confidence}%)")
    
    print(f"\n[FINAL RESULT]")
    print(f"  Matched pairs: {len(matched_pairs)}")
    if len(matched_pairs) == 0:
        print("  ⚠️  EMPTY PAIRS - THIS IS WHY UI SHOWS 'No comparable claim pairs were found'")
    
    return claims_by_paper, matched_pairs


# ============================================================================
# MAIN - Run from streamlit session state
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# DEBUGGING SCRIPT FOR ANALYSIS PIPELINES")
    print("#"*80)
    print("\nTo use this script:")
    print("1. Run a search in the Streamlit app")
    print("2. Copy the selected papers from app.py session state")
    print("3. Run this script with those papers as input")
    print("\nFor now, this script provides the framework for debugging.")
    print("You'll need to feed it actual paper data from a session.")
