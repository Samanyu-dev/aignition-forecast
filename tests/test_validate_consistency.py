import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from validate_consistency import run_validation


def test_run_validation_on_data_dir():
    # Run validation against repo sample data/
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if os.path.exists(data_dir):
        report = run_validation(data_dir)
        assert "channels" in report
        assert "summary" in report
        assert report["summary"]["total_campaigns"] > 0
