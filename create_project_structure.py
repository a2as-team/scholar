import os

project_structure = [
    "scholar_verse/scholar_verse/shared_libraries/feedback",
    "scholar_verse/scholar_verse/shared_libraries/state_management",
    "scholar_verse/scholar_verse/shared_libraries/web_utils",

    "scholar_verse/scholar_verse/sub_agents/ingestion/tools",
    "scholar_verse/scholar_verse/sub_agents/ingestion",

    "scholar_verse/scholar_verse/sub_agents/citation_graph/tools",
    "scholar_verse/scholar_verse/sub_agents/citation_graph",

    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/tools",
    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis",

    "scholar_verse/scholar_verse/sub_agents/deep_search/tools",
    "scholar_verse/scholar_verse/sub_agents/deep_search",

    "scholar_verse/scholar_verse/sub_agents/insight/tools",
    "scholar_verse/scholar_verse/sub_agents/insight",

    "scholar_verse/scholar_verse/sub_agents/visualization/tools",
    "scholar_verse/scholar_verse/sub_agents/visualization",

    "scholar_verse/scholar_verse/tools",

    "scholar_verse/deployment",
    "scholar_verse/eval/agent_performance",
    "scholar_verse/eval/insight_quality",
    "scholar_verse/eval/user_feedback",
    "scholar_verse/tests",
    "scholar_verse/web_research_cache",
]

files_to_create = [
    "scholar_verse/scholar_verse/__init__.py",
    "scholar_verse/scholar_verse/agent.py",
    "scholar_verse/scholar_verse/prompt.py",

    "scholar_verse/scholar_verse/tools/dynamic_routing.py",
    "scholar_verse/scholar_verse/tools/agent_evaluation.py",
    "scholar_verse/scholar_verse/tools/feedback_processing.py",

    "scholar_verse/scholar_verse/sub_agents/ingestion/agent.py",
    "scholar_verse/scholar_verse/sub_agents/ingestion/prompt.py",

    "scholar_verse/scholar_verse/sub_agents/citation_graph/agent.py",
    "scholar_verse/scholar_verse/sub_agents/citation_graph/prompt.py",

    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/agent.py",
    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/prompt.py",
    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/tools/compare_methodologies.py",
    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/tools/identify_trends.py",
    "scholar_verse/scholar_verse/sub_agents/cross_paper_analysis/tools/analyze_conflicts.py",

    "scholar_verse/scholar_verse/sub_agents/deep_search/agent.py",
    "scholar_verse/scholar_verse/sub_agents/deep_search/prompt.py",
    "scholar_verse/scholar_verse/sub_agents/deep_search/tools/web_search.py",
    "scholar_verse/scholar_verse/sub_agents/deep_search/tools/content_extraction.py",
    "scholar_verse/scholar_verse/sub_agents/deep_search/tools/validation.py",

    "scholar_verse/scholar_verse/sub_agents/insight/agent.py",
    "scholar_verse/scholar_verse/sub_agents/insight/prompt.py",
    "scholar_verse/scholar_verse/sub_agents/insight/refinement.py",

    "scholar_verse/scholar_verse/sub_agents/visualization/agent.py",
    "scholar_verse/scholar_verse/sub_agents/visualization/prompt.py",
    "scholar_verse/scholar_verse/sub_agents/visualization/tools/cluster_concepts.py",
    "scholar_verse/scholar_verse/sub_agents/visualization/tools/highlight_relationships.py",
]

# Create directories
for folder in project_structure:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path in files_to_create:
    with open(file_path, 'w') as f:
        f.write("# " + os.path.basename(file_path) + "\n")

print("Project structure created successfully.")
