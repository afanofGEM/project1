from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
data_path = (project_root/"data"/"data.csv")
output_dir = (project_root/"outputs"/"check_data")
class_distribution_path = (output_dir/"class_distribution.csv")
text_lengths_path = (output_dir/"text_lengths.csv")
duplicate_texts_path = (output_dir/"duplicate_texts.csv")
conflicting_labels_path = (output_dir/"conflicting_label_texts.csv")
check_result_path = (output_dir/"check_data.json")

required_columns = {
    "ticket_id",
    "text",
    "label",
    "split",
}

valid_labels = {
    "app",
    "bus",
    "calc",
    "chat",
    "cinemas",
    "contacts",
    "cookbook",
    "datetime",
    "email",
    "epg",
    "flight",
    "health",
    "lottery",
    "map",
    "match",
    "message",
    "music",
    "news",
    "novel",
    "poetry",
    "radio",
    "riddle",
    "schedule",
    "stock",
    "telephone",
    "train",
    "translation",
    "tvchannel",
    "video",
    "weather",
    "website",
}


def load_data(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"找不到原始数据文件：{file_path}")

    dataframe = pd.read_csv(file_path)
    if dataframe.empty:
        raise ValueError("原始数据为空。")

    return dataframe


def check_required_columns(dataframe):
    """检查必要字段是否完整。"""

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError("原始数据缺少必要列："f"{missing_columns}")

    return missing_columns


def normalize_dataframe(dataframe):
    """统一 text 和 label 的缺失值、类型与首尾空格。"""

    dataframe = dataframe.copy()
    dataframe["text"] = (dataframe["text"].fillna("").astype(str).str.strip())
    dataframe["label"] = (dataframe["label"].fillna("").astype(str).str.strip())

    return dataframe


def check_empty_values(dataframe):
    """检查空文本和空标签。"""

    empty_text_mask = (dataframe["text"].eq(""))
    empty_label_mask = (dataframe["label"].eq(""))

    empty_text_count = int(empty_text_mask.sum())
    empty_label_count = int(empty_label_mask.sum())
    print("空文本数量：",empty_text_count)

    if empty_text_count > 0:
        print("空文本所在行：")
        print(dataframe.loc[empty_text_mask,["ticket_id","text","label"]])
    print("空标签数量：",empty_label_count)

    if empty_label_count > 0:
        print("空标签所在行：")
        print(
            dataframe.loc[
                empty_label_mask,
                [
                    "ticket_id",
                    "text",
                    "label",
                ],
            ]
        )

    return (
        empty_text_mask,
        empty_label_mask,
    )


# =========================
# 6. 检查重复 ticket_id
# =========================
def check_duplicate_ticket_ids(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """检查 ticket_id 是否重复。"""

    duplicate_id_mask = (
        dataframe["ticket_id"]
        .duplicated(
            keep=False,
        )
    )

    duplicate_id_count = int(
        duplicate_id_mask.sum()
    )

    print(
        "重复 ticket_id 行数：",
        duplicate_id_count,
    )

    if duplicate_id_count > 0:
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

    return duplicate_id_mask


# =========================
# 7. 检查重复文本
# =========================
def check_duplicate_texts(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """检查非空文本是否重复，并保存重复文本明细。"""

    duplicate_text_mask = (
        dataframe["text"]
        .duplicated(
            keep=False,
        )
        & dataframe["text"].ne("")
    )

    duplicate_data = (
        dataframe.loc[
            duplicate_text_mask,
            [
                "ticket_id",
                "text",
                "label",
            ],
        ]
        .sort_values(
            by=[
                "text",
                "label",
                "ticket_id",
            ],
        )
        .reset_index(
            drop=True,
        )
    )

    print(
        "重复文本行数：",
        len(duplicate_data),
    )

    if not duplicate_data.empty:
        print(
            "重复文本："
        )
        print(duplicate_data)

        duplicate_data.to_csv(
            duplicate_texts_path,
            index=False,
            encoding="utf-8",
        )

    return duplicate_data


# =========================
# 8. 检查重复文本的标签冲突
# =========================
def check_conflicting_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """检查相同文本是否对应多个不同标签。"""

    non_empty_dataframe = (
        dataframe.loc[
            dataframe["text"].ne("")
        ]
    )

    label_count_by_text = (
        non_empty_dataframe
        .groupby("text")["label"]
        .nunique()
    )

    conflicting_texts = (
        label_count_by_text[
            label_count_by_text > 1
        ]
        .index
    )

    conflicting_data = (
        dataframe.loc[
            dataframe["text"].isin(
                conflicting_texts
            ),
            [
                "ticket_id",
                "text",
                "label",
            ],
        ]
        .sort_values(
            by=[
                "text",
                "label",
                "ticket_id",
            ],
        )
        .reset_index(
            drop=True,
        )
    )

    print(
        "存在标签冲突的文本数量：",
        len(conflicting_texts),
    )

    if not conflicting_data.empty:
        print(
            "标签冲突的重复文本："
        )
        print(conflicting_data)

        conflicting_data.to_csv(
            conflicting_labels_path,
            index=False,
            encoding="utf-8",
        )

    return conflicting_data


# =========================
# 9. 检查未知标签
# =========================
def check_unknown_labels(
    dataframe: pd.DataFrame,
) -> list[str]:
    """检查 label 是否都属于预期类别。"""

    unknown_label_mask = (
        dataframe["label"].ne("")
        & ~dataframe["label"].isin(
            valid_labels
        )
    )

    unknown_labels = sorted(
        dataframe.loc[
            unknown_label_mask,
            "label",
        ]
        .unique()
        .tolist()
    )

    print(
        "未知标签：",
        unknown_labels,
    )

    if unknown_labels:
        print(
            "未知标签所在行："
        )
        print(
            dataframe.loc[
                unknown_label_mask,
                [
                    "ticket_id",
                    "text",
                    "label",
                ],
            ]
        )

    return unknown_labels


# =========================
# 10. 统计类别分布
# =========================
def calculate_class_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """统计各标签的数量和百分比。"""

    valid_dataframe = (
        dataframe.loc[
            dataframe["label"].ne("")
        ]
    )

    class_distribution = (
        valid_dataframe["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    valid_label_count = len(
        valid_dataframe
    )

    if valid_label_count == 0:
        class_distribution[
            "percentage"
        ] = 0.0
    else:
        class_distribution[
            "percentage"
        ] = (
            class_distribution["count"]
            / valid_label_count
            * 100
        ).round(2)

    print(
        "类别分布："
    )
    print(class_distribution)

    return class_distribution


# =========================
# 11. 统计文本长度
# =========================
def calculate_text_lengths(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.Series,
    dict[str, float | int],
]:
    """统计文本字符长度。"""

    text_lengths = (
        dataframe["text"]
        .str.len()
    )

    text_length_result = {
        "min": int(
            text_lengths.min()
        ),
        "max": int(
            text_lengths.max()
        ),
        "mean": round(
            float(
                text_lengths.mean()
            ),
            2,
        ),
        "median": round(
            float(
                text_lengths.median()
            ),
            2,
        ),
    }

    print(
        "文本长度分布："
    )
    print(text_length_result)

    return (
        text_lengths,
        text_length_result,
    )


# =========================
# 12. 保存检查结果
# =========================
def save_results(
    dataframe: pd.DataFrame,
    class_distribution: pd.DataFrame,
    text_lengths: pd.Series,
    check_result: dict,
) -> None:
    """保存类别分布、文本长度和 JSON 检查结果。"""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_distribution.to_csv(
        class_distribution_path,
        index=False,
        encoding="utf-8",
    )

    text_length_dataframe = pd.DataFrame({
        "ticket_id": dataframe[
            "ticket_id"
        ],
        "text_length": text_lengths,
    })

    text_length_dataframe.to_csv(
        text_lengths_path,
        index=False,
        encoding="utf-8",
    )

    with check_result_path.open(
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
# 13. 主程序
# =========================
def main() -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_data(
        data_path
    )

    print(
        "数据尺寸：",
        dataframe.shape,
    )
    print(
        "字段名称：",
        dataframe.columns.tolist(),
    )

    missing_columns = (
        check_required_columns(
            dataframe
        )
    )

    print(
        "原始数据表不存在列缺失。"
    )

    dataframe = normalize_dataframe(
        dataframe
    )

    (
        empty_text_mask,
        empty_label_mask,
    ) = check_empty_values(
        dataframe
    )

    duplicate_id_mask = (
        check_duplicate_ticket_ids(
            dataframe
        )
    )

    duplicate_data = (
        check_duplicate_texts(
            dataframe
        )
    )

    conflicting_data = (
        check_conflicting_labels(
            dataframe
        )
    )

    unknown_labels = (
        check_unknown_labels(
            dataframe
        )
    )

    class_distribution = (
        calculate_class_distribution(
            dataframe
        )
    )

    (
        text_lengths,
        text_length_result,
    ) = calculate_text_lengths(
        dataframe
    )

    check_result = {
        "total_samples": int(
            len(dataframe)
        ),
        "missing_columns": (
            missing_columns
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
        "duplicate_text_row_count": int(
            len(duplicate_data)
        ),
        "conflicting_label_row_count": int(
            len(conflicting_data)
        ),
        "unknown_labels": (
            unknown_labels
        ),
        "text_length_result": (
            text_length_result
        ),
    }

    save_results(
        dataframe=dataframe,
        class_distribution=(
            class_distribution
        ),
        text_lengths=text_lengths,
        check_result=check_result,
    )

    critical_issues = []

    if empty_text_mask.any():
        critical_issues.append(
            "存在空文本"
        )

    if empty_label_mask.any():
        critical_issues.append(
            "存在空标签"
        )

    if duplicate_id_mask.any():
        critical_issues.append(
            "存在重复 ticket_id"
        )

    if not conflicting_data.empty:
        critical_issues.append(
            "相同文本存在标签冲突"
        )

    if unknown_labels:
        critical_issues.append(
            "存在未知标签"
        )

    if critical_issues:
        raise ValueError(
            "原始数据检查未通过："
            + "；".join(
                critical_issues
            )
        )

    print(
        "\n原始数据检查通过。"
    )


if __name__ == "__main__":
    main()
