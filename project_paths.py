"""Keep experiment outputs separate from the code when running in a notebook."""

import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("SYDE411_WORKSPACE", SOURCE)).expanduser().resolve()
