"""Test-wide setup: CAD outputs go to a temp dir so the tracked fixtures/cad_outputs samples stay untouched."""
import os
import shutil
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "fixtures" / "cad_outputs"
_TMP = Path(tempfile.mkdtemp(prefix="rewind_cad_test_"))
if _SRC.exists():
    shutil.copytree(_SRC, _TMP, dirs_exist_ok=True)
os.environ.setdefault("REWIND_CAD_OUTPUT_DIR", str(_TMP))
