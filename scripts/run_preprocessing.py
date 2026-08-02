# scripts/run_preprocessing.py
from src.preprocessing.config import WLASL_PROCESSED_DIR, HF_MIRROR_PROCESSED_DIR
from src.preprocessing.metadata_readers.wlasl_reader import WLASLReader
from src.preprocessing.metadata_readers.hf_mirror_reader import HFMirrorReader
from src.preprocessing.dataset_processor import DatasetProcessor

DatasetProcessor(WLASLReader(), WLASL_PROCESSED_DIR).process(limit=20)      # smoke test first
DatasetProcessor(HFMirrorReader(), HF_MIRROR_PROCESSED_DIR).process(limit=20)