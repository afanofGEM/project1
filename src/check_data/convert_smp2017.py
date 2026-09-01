"""Convert SMP2017-ECDT Task 1 text files to CSV files.

Task 1 is domain/intent classification. Each txt filename supplies the label,
and each non-empty line is one utterance. The official train/develop/test
split is retained instead of randomly splitting the corpus again.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ZIP_PATH = Path(r"D:\Google\DownLoad\LREC.zip")
OUTPUT_DIR = PROJECT_ROOT / "data" / "smp2017"
TASK1_PREFIX = "LREC/ReleasedData/Task1data/"
SPLIT_NAMES = {"train": "train", "develop": "valid", "test": "test"}
FIELDNAMES = ["ticket_id", "text", "label", "split"]


def read_rows(zip_file: zipfile.ZipFile) -> dict[str, list[dict[str, str]]]:
    rows_by_split = {split: [] for split in SPLIT_NAMES.values()}
    counters = {split: 0 for split in SPLIT_NAMES.values()}

    for member in sorted(zip_file.namelist()):
        if not member.startswith(TASK1_PREFIX) or not member.endswith(".txt"):
            continue

        relative_parts = Path(member.removeprefix(TASK1_PREFIX)).parts
        if len(relative_parts) != 2 or relative_parts[0] not in SPLIT_NAMES:
            continue

        source_split, filename = relative_parts
        split = SPLIT_NAMES[source_split]
        expected_prefix = f"{source_split}_"
        label = Path(filename).stem.removeprefix(expected_prefix)
        content = zip_file.read(member).decode("utf-8-sig")

        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            counters[split] += 1
            rows_by_split[split].append(
                {
                    "ticket_id": f"SMP_{split}_{counters[split]:05d}",
                    "text": text,
                    "label": label,
                    "split": split,
                }
            )

    return rows_by_split


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"SMP2017 archive not found: {ZIP_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zip_file:
        rows_by_split = read_rows(zip_file)

    all_rows = []
    for split in ("train", "valid", "test"):
        rows = rows_by_split[split]
        write_csv(OUTPUT_DIR / f"{split}.csv", rows)
        all_rows.extend(rows)
        print(f"{split}: {len(rows)} rows")

    write_csv(OUTPUT_DIR / "smp2017_all.csv", all_rows)
    print(f"all: {len(all_rows)} rows")


if __name__ == "__main__":
    main()
