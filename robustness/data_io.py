"""
data_io.py

Thin wrapper around src.preprocessing.sample_writer so that other
robustness/ modules (and scripts like the latency timer) import I/O
through robustness.data_io rather than reaching into src/ directly.

This module intentionally has no logic of its own: no NaN-scanning, no
metadata["segments"] lookups, no flat-array assumptions. The real data
contract lives in src/preprocessing/sample_schema.py and
src/preprocessing/sample_writer.py; see those for details on
ProcessedSample, SampleMetadata, and LandmarkPresence.
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.sample_schema import ProcessedSample
from src.preprocessing.sample_writer import load_sample, write_sample

__all__ = ["load_sample", "write_sample", "load_recording"]


def load_recording(path: Path) -> ProcessedSample:
    """
    Load a ProcessedSample from `path`.

    This is a documented alias for load_sample, kept under the old
    `load_recording` name to minimize churn in callers (e.g. the
    latency timer) that were written against the previous flat
    Recording-based data_io. Despite the name, this now returns a
    ProcessedSample (four separate landmark arrays + presence +
    metadata) -- not the old flat Recording NamedTuple.
    """
    return load_sample(path)