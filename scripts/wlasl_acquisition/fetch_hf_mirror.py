import fiftyone as fo
import fiftyone.utils.huggingface as fouh

dataset = fouh.load_from_hub("Voxel51/WLASL")
dataset.export(
    export_dir="data/raw/wlasl/hf_mirror",
    dataset_type=fo.types.VideoDirectory,
)