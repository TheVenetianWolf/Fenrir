import sys
from pathlib import Path

# Put the project root (the folder containing core/, services/, ui/) on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
