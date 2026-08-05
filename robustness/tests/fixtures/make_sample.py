"""
Re-exports robustness.synthetic for existing test imports
(robustness.tests.fixtures.make_sample.*). Kept for backward
compatibility with tests written against this path.
"""
from robustness.synthetic import make_synthetic_sample, round_trip_sample

__all__ = ["make_synthetic_sample", "round_trip_sample"]