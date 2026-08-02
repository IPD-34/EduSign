"""
config.py

Central configuration for the preprocessing pipeline.

Only configuration values should live here.
No processing logic.
"""

from pathlib import Path

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# -------------------------------------------------------
# ASLLRP
# -------------------------------------------------------

ASLLRP_DIR = RAW_DATA_DIR / "asllrp"

ASLLRP_XML_DIR = (
    ASLLRP_DIR
    / "xml"
)

ASLLRP_VIDEO_DIR = (
    ASLLRP_DIR
    / "videos"
)

ASLLRP_PROCESSED_DIR = (
    PROCESSED_DATA_DIR
    / "asllrp"
)

ASLLRP_MAX_SEQUENCE_LENGTH = 250  # continuous-signing utterances are structurally
                                   # longer than WLASL's isolated signs; measured
                                   # distribution (1,234 utterances): median=127,
                                   # p90=236, p95=288. Cap-hit rate at 250 is 8.5%,
                                   # vs. 36.8% at the shared 150 — real, measured
                                   # tradeoff, not a guess.

# -------------------------------------------------------
# WLASL
# -------------------------------------------------------

WLASL_DIR = RAW_DATA_DIR / "wlasl"

WLASL_METADATA = (
    WLASL_DIR
    / "metadata"
    / "WLASL_v0.3.json"
)

WLASL_VIDEO_DIR = (
    WLASL_DIR
    / "videos"
)

WLASL_PROCESSED_DIR = (
    PROCESSED_DATA_DIR
    / "wlasl"
)

# -------------------------------------------------------
# Processing
# -------------------------------------------------------

SAVE_FORMAT = "npz"

TARGET_FPS = 25            # matches WLASL's native fps — no unnecessary downsampling
MAX_SEQUENCE_LENGTH = 150  # ~6s at 25fps; generous over a typical 1-3s isolated sign,
                           # still caps real outliers (e.g. video 70212's 5692 raw frames).
                           # Starting guess, not measured — revisit once frame_sampler
                           # stats are logged across a real batch (Day 7 requirement).

# Keep original MediaPipe coordinates
NORMALIZE_LANDMARKS = False