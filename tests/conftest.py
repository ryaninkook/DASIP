import sys
from pathlib import Path

# The transport code and the policy networks are imported as `SiT.*` and
# `SI_policy.*` relative to SIP/, the same way the training scripts do it.
SIP_DIR = Path(__file__).resolve().parent.parent / "SIP"
sys.path.insert(0, str(SIP_DIR))
