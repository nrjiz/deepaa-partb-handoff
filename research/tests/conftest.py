from __future__ import annotations

import sys
from pathlib import Path


sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

