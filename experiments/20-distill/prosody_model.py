"""Legacy shim: ProsodyStudent/CNBlock now live in fastkoko.models."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.models.blocks import CNBlock  # noqa: F401
from fastkoko.models.prosody import VOCAB, ProsodyStudent  # noqa: F401
