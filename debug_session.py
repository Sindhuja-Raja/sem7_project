"""
Live debugging script - extract current session state and run full debug trace.
This must be run WHILE the Streamlit app is running after a search.
"""

import json
import pickle
import sys
from pathlib import Path

# Try to import from streamlit's session state
try:
    import streamlit as st
    
    print("\n" + "="*80)
    print("EXTRACTING STREAMLIT SESSION STATE")
    print("="*80)
    
    # Check session state
    if not hasattr(st, 'session_state'):
        print("❌ No streamlit session state found")
        sys.exit(1)
    
    valid_papers = st.session_state.get('valid_papers', [])
    knowledge_analysis = st.session_state.get('knowledge_analysis')
    problem_solution_analysis = st.session_state.get('problem_solution_analysis')
    rag_cache = st.session_state.get('rag_index_cache', {})
    
    print(f"\n✓ Valid papers: {len(valid_papers)}")
    for i, p in enumerate(valid_papers):
        print(f"  {i+1}. {p.title[:60]}")
    
    print(f"\n✓ Knowledge analysis: {knowledge_analysis is not None}")
    if knowledge_analysis:
        print(f"  - knowledge_table: {len(knowledge_analysis.knowledge_table)} rows")
        print(f"  - overall_analysis: {len(knowledge_analysis.overall_analysis)} rows")
    
    print(f"\n✓ Problem-solution analysis: {problem_solution_analysis is not None}")
    if problem_solution_analysis:
        print(f"  - problem_solution_table: {len(problem_solution_analysis.problem_solution_table)} rows")
        print(f"  - summary: {len(problem_solution_analysis.summary)} rows")
    
    print(f"\n✓ RAG cache: {len(rag_cache)} indices")
    
    # Now run debugging
    from debug_analysis import debug_problem_solution_analysis, debug_contradiction_detection
    
    if valid_papers and knowledge_analysis:
        print("\n" + "="*80)
        print("RUNNING PROBLEM-SOLUTION ANALYSIS DEBUG")
        print("="*80)
        
        debug_problem_solution_analysis(
            valid_papers,
            knowledge_analysis.knowledge_table,
            knowledge_analysis.overall_analysis,
            rag_cache
        )
    
    # For contradiction detection, we'd need selected papers
    # This is left as a manual step for now
    print("\n" + "="*80)
    print("CONTRADICTION DETECTION")
    print("="*80)
    print("\nTo debug contradiction detection:")
    print("1. Select 2+ papers in the UI")
    print("2. Run this debug script again")
    
except ImportError:
    print("\n❌ Could not import streamlit - run this script within the Streamlit environment")
    print("\nAlternative: Run this in VS Code terminal while Streamlit is running:")
    print("  streamlit run app.py --logger.level=debug")
    print("  Then in another terminal, modify this script to import the session directly")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
