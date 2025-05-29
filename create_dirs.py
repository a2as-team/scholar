"""Script to create the necessary directories for ScholarVerse."""

import os
from pathlib import Path

# Define the project root directory
ROOT_DIR = Path(__file__).parent

# Define the directories to create
DIRECTORIES = [
    # Data directories
    ROOT_DIR / "data",
    ROOT_DIR / "data" / "papers",
    ROOT_DIR / "data" / "embeddings",
    ROOT_DIR / "data" / "exports",
    
    # Temporary directories
    ROOT_DIR / "temp",
    
    # Log directories
    ROOT_DIR / "logs",
    
    # Make sure all package directories exist
    ROOT_DIR / "scholar_verse" / "shared_libraries" / "feedback",
    ROOT_DIR / "scholar_verse" / "shared_libraries" / "state_management",
    ROOT_DIR / "scholar_verse" / "shared_libraries" / "web_utils",
    ROOT_DIR / "scholar_verse" / "shared_libraries" / "db_utils",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "ingestion" / "tools",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "citation_graph" / "tools",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "cross_paper_analysis" / "tools",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "deep_search" / "tools",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "insight" / "tools",
    ROOT_DIR / "scholar_verse" / "sub_agents" / "visualization" / "tools",
    ROOT_DIR / "scholar_verse" / "tools",
]


def create_directories():
    """Create the necessary directories for ScholarVerse."""
    print("Creating directories for ScholarVerse...")
    
    for directory in DIRECTORIES:
        # Create the directory if it doesn't exist
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")
    
    print("\nAll directories created successfully!")


if __name__ == "__main__":
    create_directories()
