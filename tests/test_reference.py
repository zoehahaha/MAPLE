from pathlib import Path

from prep.reporting import validate_reference


def test_paper_reference_values():
    root = Path(__file__).resolve().parents[1]
    result = validate_reference(root)
    assert result["status"] == "PASS", result

