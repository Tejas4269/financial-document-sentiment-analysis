from pathlib import Path
import pytest

from src.data import load_phrasebank


def test_load_phrasebank_maps_binary_labels_and_excludes_neutral(tmp_path: Path):
    dataset = tmp_path / "sample.txt"
    dataset.write_text(
        "Company reports strong growth@positive\n"
        "Company reports a loss@negative\n"
        "Company outlook remains unchanged@neutral\n",
        encoding="utf-8",
    )

    result = load_phrasebank(dataset)

    assert result["text"].tolist() == [
        "Company reports strong growth",
        "Company reports a loss",
    ]
    assert result["label"].tolist() == [1, 0]


def test_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_phrasebank("does-not-exist.txt")
