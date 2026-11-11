from pathlib import Path

from prep.replay import run_replay


def test_tiny_offline_replay(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = run_replay(
        root / "data" / "tiny_demo.json",
        root / "experiments" / "configs" / "tiny_demo.json",
        tmp_path,
    )
    assert result["status"] == "COMPLETE"
    assert result["provider_attempts"] == 0
    assert (tmp_path / "RESULT.json").is_file()

