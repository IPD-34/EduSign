from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.preprocessing.metadata_readers.wlasl_reader import WLASLReader
from src.preprocessing.dataset_processor import DatasetProcessor

processor = DatasetProcessor(WLASLReader(), WLASL_PROCESSED_DIR)
processor.process()  # no limit — full corpus