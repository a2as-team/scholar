"""Test script to verify imports and basic functionality."""

import sys
import traceback
from pathlib import Path

print("=" * 50)
print("ScholarVerse Import Tester")
print("=" * 50)

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

# Test Google ADK imports
print("\n" + "=" * 50)
print("Testing Google ADK imports...")
try:
    from google.adk.agents.invocation_context import InvocationContext
    print("✅ Successfully imported InvocationContext from google.adk.agents.invocation_context")
except ImportError as e:
    print(f"❌ Failed to import InvocationContext: {e}")
    traceback.print_exc()

try:
    from google.adk.events.event import Event
    print("✅ Successfully imported Event from google.adk.events.event")
except ImportError as e:
    print(f"❌ Failed to import Event: {e}")
    traceback.print_exc()

# Test project imports
print("\n" + "=" * 50)
print("Testing project imports...")

try:
    from scholar_verse.base_agent import BaseAgent
    print("✅ Successfully imported BaseAgent from scholar_verse.base_agent")
except ImportError as e:
    print(f"❌ Failed to import BaseAgent: {e}")
    traceback.print_exc()

try:
    from scholar_verse.agent import RouterAgent
    print("✅ Successfully imported RouterAgent from scholar_verse.agent")
except ImportError as e:
    print(f"❌ Failed to import RouterAgent: {e}")
    traceback.print_exc()

# Test sub-agents imports
try:
    from scholar_verse.sub_agents.ingestion.agent import IngestionAgent
    print("✅ Successfully imported IngestionAgent from scholar_verse.sub_agents.ingestion.agent")
except ImportError as e:
    print(f"❌ Failed to import IngestionAgent: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("Import test completed!")
print("=" * 50)
