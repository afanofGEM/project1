from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# =========================
# 1. 路径与基础配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs" / "check_data"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.csv"
)

valid_PATH = (
    DATA_DIR
    / "valid.csv"
)

TEST_PATH = (
    DATA_DIR
    / "test.csv"
)

SPLIT_DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "split_distribution.csv"
)

CHECK_RESULT_PATH = (
    OUTPUT_DIR
    / "check_splits.json"
)

REQUIRED_COLUMNS = {
    "ticket_id",
    "text",
    "label",
}

SPLIT_PATHS = {
    "train": TRAIN_PATH,
    "valid": valid_PATH,
    "test": TEST_PATH,
}


# =========================
# 2. 读取并检查单个划分
# =========================
def load_split(
    split_name: str,
    file_path: Path,
) -> pd.DataFrame:
    """读取一个数据划分并进行基础检查。"""

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 {split_name} 文件："
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
    )

    if dataframe.empty:
        raise ValueError(
            f"{split_name} 数据集为空。"
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} 缺少必要列："
            f"{missing_columns}"
        )

    dataframe = dataframe.copy()

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["label"] = (
        dataframe["label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return dataframe


# =========================
# 3. 检查单个划分内部问题
# =========================
def inspect_split(
    split_name: str,
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """检查一个划分内部的空值和重复 ID。"""

    empty_text_mask = (
        dataframe["text"]
        .eq("")
    )

    empty_label_mask = (
        dataframe["label"]
        .eq("")
    )

    duplicate_id_mask = (
        dataframe["ticket_id"]
        .duplicated(
            keep=False,
        )
    )

    print(
        f"\n========== {split_name} =========="
    )

    print(
        "样本数量：",
        len(dataframe),
    )

    print(
        "空文本数量：",
        int(
            empty_text_mask.sum()
        ),
    )

    print(
        "空标签数量：",
        int(
            empty_label_mask.sum()
        ),
    )

    print(
        "重复 ticket_id 行数：",
        int(
            duplicate_id_mask.sum()
        ),
    )

    if duplicate_id_mask.any():
        print(
            "重复 ticket_id："
        )
        print(
            dataframe.loc[
                duplicate_id_mask,
                [
                    "ticket_id",
                    "text",
                    "label",
                ],
            ].sort_values(
                by="ticket_id",
            )
        )

    print(
        "标签数量："
    )
    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "标签比例："
    )
    print(
        dataframe["label"]
        .value_counts(
            normalize=True,
        )
        .sort_index()
        .round(4)
    )

    return {
        "sample_count": int(
            len(dataframe)
        ),
        "empty_text_count": int(
            empty_text_mask.sum()
        ),
        "empty_label_count": int(
            empty_label_mask.sum()
        ),
        "duplicate_ticket_id_count": int(
            duplicate_id_mask.sum()
        ),
    }


# =========================
# 4. 检查不同划分是否重叠
# =========================
def check_cross_split_overlap(
    train_dataframe: pd.DataFrame,
    valid_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, list]:
    """检查 train、valid、test 之间是否有相同 ticket_id。"""

    train_ids = set(
        train_dataframe["ticket_id"]
    )

    valid_ids = set(
        valid_dataframe["ticket_id"]
    )

    test_ids = set(
        test_dataframe["ticket_id"]
    )

    train_valid_overlap = sorted(
        train_ids
        & valid_ids
    )

    train_test_overlap = sorted(
        train_ids
        & test_ids
    )

    valid_test_overlap = sorted(
        valid_ids
        & test_ids
    )

    print(
        "\n========== 跨数据集重复检查 =========="
    )

    print(
        "train 与 valid 重复数量：",
        len(train_valid_overlap),
    )

    print(
        "train 与 test 重复数量：",
        len(train_test_overlap),
    )

    print(
        "valid 与 test 重复数量：",
        len(valid_test_overlap),
    )

    if train_valid_overlap:
        print(
            "train 与 valid 重复示例：",
            train_valid_overlap[:10],
        )

    if train_test_overlap:
        print(
            "train 与 test 重复示例：",
            train_test_overlap[:10],
        )

    if valid_test_overlap:
        print(
            "valid 与 test 重复示例：",
            valid_test_overlap[:10],
        )

    return {
        "train_valid_overlap": (
            train_valid_overlap
        ),
        "train_test_overlap": (
            train_test_overlap
        ),
        "valid_test_overlap": (
            valid_test_overlap
        ),
    }


# =========================
# 5. 检查文本是否跨集合重复
# =========================
def check_cross_split_text_overlap(
    train_dataframe: pd.DataFrame,
    valid_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """检查相同非空文本是否出现在不同数据划分中。"""

    train_texts = set(
        train_dataframe.loc[
            train_dataframe["text"].ne(""),
            "text",
        ]
    )

    valid_texts = set(
        valid_dataframe.loc[
            valid_dataframe["text"].ne(""),
            "text",
        ]
    )

    test_texts = set(
        test_dataframe.loc[
            test_dataframe["text"].ne(""),
            "text",
        ]
    )

    train_valid_overlap = sorted(
        train_texts
        & valid_texts
    )

    train_test_overlap = sorted(
        train_texts
        & test_texts
    )

    valid_test_overlap = sorted(
        valid_texts
        & test_texts
    )

    print(
        "\n========== 跨数据集文本重复检查 =========="
    )

    print(
        "train 与 valid 重复文本数量：",
        len(train_valid_overlap),
    )

    print(
        "train 与 test 重复文本数量：",
        len(train_test_overlap),
    )

    print(
        "valid 与 test 重复文本数量：",
        len(valid_test_overlap),
    )

    if train_valid_overlap:
        print(
            "train 与 valid 重复文本示例：",
            train_valid_overlap[:5],
        )

    if train_test_overlap:
        print(
            "train 与 test 重复文本示例：",
            train_test_overlap[:5],
        )

    if valid_test_overlap:
        print(
            "valid 与 test 重复文本示例：",
            valid_test_overlap[:5],
        )

    return {
        "train_valid_text_overlap": (
            train_valid_overlap
        ),
        "train_test_text_overlap": (
            train_test_overlap
        ),
        "valid_test_text_overlap": (
            valid_test_overlap
        ),
    }


# =========================
# 6. 检查标签集合
# =========================
def check_label_sets(
    train_dataframe: pd.DataFrame,
    valid_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """检查三个划分是否包含相同的标签集合。"""

    train_labels = sorted(
        set(
            train_dataframe["label"]
        )
    )

    valid_labels = sorted(
        set(
            valid_dataframe["label"]
        )
    )

    test_labels = sorted(
        set(
            test_dataframe["label"]
        )
    )

    print(
        "\n========== 标签集合检查 =========="
    )

    print(
        "train 标签：",
        train_labels,
    )

    print(
        "valid 标签：",
        valid_labels,
    )

    print(
        "test 标签：",
        test_labels,
    )

    return {
        "train_labels": train_labels,
        "valid_labels": valid_labels,
        "test_labels": test_labels,
    }


# =========================
# 7. 汇总三个划分的标签分布
# =========================
def build_split_distribution(
    split_dataframes: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """整理 train、valid、test 的标签数量和百分比。"""

    results = []

    for split_name, dataframe in (
        split_dataframes.items()
    ):
        distribution = (
            dataframe["label"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )

        distribution["percentage"] = (
            distribution["count"]
            / len(dataframe)
            * 100
        ).round(2)

        distribution.insert(
            0,
            "split",
            split_name,
        )

        results.append(
            distribution
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


# =========================
# 8. 保存检查结果
# =========================
def save_results(
    distribution: pd.DataFrame,
    check_result: dict,
) -> None:
    """保存划分分布和 JSON 检查结果。"""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution.to_csv(
        SPLIT_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8",
    )

    with CHECK_RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            check_result,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


# =========================
# 9. 主程序
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_dataframes = {
        split_name: load_split(
            split_name,
            file_path,
        )
        for split_name, file_path
        in SPLIT_PATHS.items()
    }

    split_results = {
        split_name: inspect_split(
            split_name,
            dataframe,
        )
        for split_name, dataframe
        in split_dataframes.items()
    }

    train_dataframe = (
        split_dataframes["train"]
    )

    valid_dataframe = (
        split_dataframes["valid"]
    )

    test_dataframe = (
        split_dataframes["test"]
    )

    id_overlap_result = (
        check_cross_split_overlap(
            train_dataframe=(
                train_dataframe
            ),
            valid_dataframe=(
                valid_dataframe
            ),
            test_dataframe=(
                test_dataframe
            ),
        )
    )

    text_overlap_result = (
        check_cross_split_text_overlap(
            train_dataframe=(
                train_dataframe
            ),
            valid_dataframe=(
                valid_dataframe
            ),
            test_dataframe=(
                test_dataframe
            ),
        )
    )

    label_result = (
        check_label_sets(
            train_dataframe=(
                train_dataframe
            ),
            valid_dataframe=(
                valid_dataframe
            ),
            test_dataframe=(
                test_dataframe
            ),
        )
    )

    split_distribution = (
        build_split_distribution(
            split_dataframes
        )
    )

    total_size = sum(
        len(dataframe)
        for dataframe
        in split_dataframes.values()
    )

    check_result = {
        "total_samples": int(
            total_size
        ),
        "split_results": (
            split_results
        ),
        "id_overlap": (
            id_overlap_result
        ),
        "text_overlap": (
            text_overlap_result
        ),
        "label_sets": (
            label_result
        ),
    }

    save_results(
        distribution=(
            split_distribution
        ),
        check_result=check_result,
    )

    critical_issues = []

    for split_name, result in (
        split_results.items()
    ):
        if result[
            "empty_text_count"
        ] > 0:
            critical_issues.append(
                f"{split_name} 存在空文本"
            )

        if result[
            "empty_label_count"
        ] > 0:
            critical_issues.append(
                f"{split_name} 存在空标签"
            )

        if result[
            "duplicate_ticket_id_count"
        ] > 0:
            critical_issues.append(
                f"{split_name} 内部存在重复 ticket_id"
            )

    if any(
        id_overlap_result.values()
    ):
        critical_issues.append(
            "不同划分之间存在重复 ticket_id"
        )

    if any(
        text_overlap_result.values()
    ):
        critical_issues.append(
            "不同划分之间存在重复文本"
        )

    same_label_set = (
        label_result["train_labels"]
        == label_result["valid_labels"]
        == label_result["test_labels"]
    )

    if not same_label_set:
        critical_issues.append(
            "三个划分的标签集合不一致"
        )

    print(
        "\n三个数据集样本总数：",
        total_size,
    )

    print(
        "\n标签分布汇总："
    )
    print(split_distribution)

    if critical_issues:
        raise ValueError(
            "数据划分检查未通过："
            + "；".join(
                critical_issues
            )
        )

    print(
        "\n所有数据划分检查通过。"
    )


if __name__ == "__main__":
    main()
