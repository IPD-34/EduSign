# src/preprocessing/metadata_readers/hf_mirror_reader.py
from dataclasses import dataclass
import fiftyone as fo
import fiftyone.utils.huggingface as fouh


@dataclass
class HFMirrorSample:
    video_id: str
    gloss: str
    video_path: str
    split: str
    source: str


class HFMirrorReader:
    def __init__(self):
        try:
            self.dataset = fo.load_dataset("Voxel51/WLASL")
        except Exception:
            self.dataset = fouh.load_from_hub("Voxel51/WLASL")

    def samples(self):
        for sample in self.dataset.iter_samples(progress=False):
            yield HFMirrorSample(
                video_id=f"hf_{sample.id}",
                gloss=sample.gloss.label,
                video_path=sample.filepath,
                split="train",  # no original split info in this mirror — flagging this as a call you need to make
                source="hf_mirror",
            )