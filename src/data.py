"""Loading and validation for the Financial PhraseBank text dataset."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

LABEL_MAP = {"negative": 0, "positive": 1}


def load_phrasebank(path: str | Path) -> pd.DataFrame:
    """Load a Financial PhraseBank file and retain only positive/negative rows.

    The standard dataset format is: sentence@sentiment. rsplit is used so
    that an @ character inside the sentence does not break label parsing.
    Neutral rows are deliberately excluded for binary classification.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. See data/README.md for setup."
        )

    rows: list[tuple[str, int]] = []
    with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip() or "@" not in line:
                continue
            sentence, label = line.rsplit("@", maxsplit=1)
            normalized_label = label.strip().lower()
            if normalized_label not in LABEL_MAP:
                continue
            sentence = sentence.strip()
            if sentence:
                rows.append((sentence, LABEL_MAP[normalized_label]))

    if not rows:
        raise ValueError(
            "No positive/negative records found. Check the dataset format and file."
        )

    frame = pd.DataFrame(rows, columns=["text", "label"])
    if frame["label"].nunique() != 2:
        raise ValueError("Both positive and negative labels are required.")
    return frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
