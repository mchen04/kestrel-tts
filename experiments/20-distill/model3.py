"""Legacy shim: MaskHead now lives in fastkoko.models.vocoder."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.models.vocoder import MaskHead  # noqa: F401
from fastkoko.models.dsp import *  # noqa: F401,F403
