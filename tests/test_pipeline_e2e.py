import os
import subprocess
import sys
import tempfile
import pandas as pd
import pytest


def test_run_sh_execution():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_sh = os.path.join(repo_root, "run.sh")
    data_dir = os.path.join(repo_root, "data")
    model_path = os.path.join(repo_root, "pickle", "model.pkl")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "predictions.csv")
        
        # Execute run.sh script
        proc = subprocess.run([run_sh, data_dir, model_path, out_csv], capture_output=True, text=True)
        assert proc.returncode == 0, f"run.sh failed: {proc.stderr}"
        
        # Verify output/predictions.csv exists and is non-empty
        assert os.path.exists(out_csv)
        df = pd.read_csv(out_csv)
        assert not df.empty
        expected_cols = ["channel", "campaign_type", "campaign_id", "horizon_days", "metric", "p10", "p50", "p90"]
        assert list(df.columns) == expected_cols
        
        # Verify insights files generated in output dir
        insights_json = os.path.join(tmpdir, "insights.json")
        insights_txt = os.path.join(tmpdir, "insights.txt")
        assert os.path.exists(insights_json)
        assert os.path.exists(insights_txt)
