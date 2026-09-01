'''核心思路：通过选择低于截断率上限的最短max_len'''

from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
data_root = (project_root / "data" / "data.csv")
output_dir = (project_root / "outputs" / "decide_max_len")
text_length_root = (project_root/ "outputs" / "check_data" / "text_lengths.csv")
long_texts_root = (output_dir / "long_texts.csv")
distribution_png_root = (output_dir / "distribution_img.png")
CANDIDATE_MAX_LENGTHS = [32,48,64]

# 1.获得统计数据
def calculate_length_statistics(text_lengths:pd.Series):
    statistics = {
        "min": int(text_lengths.min()),
        "mean": float(text_lengths.mean()),
        "median": float(text_lengths.median()),
        "p90": float(text_lengths.quantile(0.90)),
        "p95": float(text_lengths.quantile(0.95)),
        "p99": float(text_lengths.quantile(0.99)),
        "max": int(text_lengths.max()),
    }

    return statistics


#保存不同max_len数据的截断率
def compare_max_lengths(text_lengths:pd.Series):
    results = []

    total_samples = len(text_lengths)

    for max_len in (CANDIDATE_MAX_LENGTHS):

        '''统计有多少超过max_len需要截断'''
        truncated_count = int((text_lengths>max_len).sum())

        truncated_ratio = truncated_count / total_samples

        results.append(
            {
                "max_length": max_len,
                "truncated_count": truncated_count,
                "truncated_ratio": truncated_ratio,
            })
        
    '''   
            max_length  truncated_count  truncated_ratio
    0          16                4              0.8
    1          32                2              0.4
    2          64                1              0.2'''
    return pd.DataFrame(results) # 合并dict同名的数据


# **选择max_len**
def choose_max_length(comparison: pd.DataFrame):

    '''[comparison["truncated_ratio"]<= 0.05]返回布尔值序列
    comparison[布尔值序列] 使用布尔索引，只保留结果为True的行。'''
    acceptable = (comparison[comparison["truncated_ratio"]<= 0.05])

    # 保存低于截断率上限的最小max_length，节省padding成本
    if not acceptable.empty:
        return int(acceptable["max_length"].min())

    return int(comparison["max_length"].max())


def save_long_text_cases(selected_max_length):
    dataframe = pd.read_csv(data_root)
    text_length_dataframe = pd.read_csv(text_length_root)

    # 合并两个csv，on表示双方的共同列，how表示保留dataframe的全部内容
    dataframe = dataframe.merge(text_length_dataframe,on="ticket_id",how="left",)

    # 保留需要截断的所有文本
    long_text_cases = (dataframe[dataframe["text_length"]>selected_max_length]
                       [
                        [
                            "ticket_id",
                            "text",
                            "label",
                            "text_length",
                        ]
                       ]
        .sort_values(by="text_length",ascending=False).reset_index(drop=True))
    
    long_text_cases.to_csv(long_texts_root,index=False,encoding="utf-8-sig")


# 绘制长度分布图
import matplotlib.pyplot as plt
def plot_distribution(text_lengths,statistics):
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths,bins=30,edgecolor="black")
    plt.axvline(statistics["p90"],linestyle="--",label="P90")
    plt.axvline(statistics["p95"],linestyle="--",label="P95")
    for max_length in (CANDIDATE_MAX_LENGTHS):
        plt.axvline(max_length,linestyle=":",label=(f"max_length="f"{max_length}"),)

    plt.xlabel("Token Length")
    plt.ylabel("Sample Count")
    plt.title("Token Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(distribution_png_root,dpi=150,)
    plt.close()


def test():
    statistics = calculate_length_statistics(pd.read_csv(text_length_root)['text_length'])
    print(type(pd.read_csv(text_length_root)['text_length']))
    print(statistics)


if __name__ == "__main__":
    test()
    statistics = calculate_length_statistics(pd.read_csv(text_length_root)['text_length'])

    results = compare_max_lengths(pd.read_csv(text_length_root)['text_length'])
    '''   
            max_length  truncated_count  truncated_ratio
    0          16                4              0.8
    1          32                2              0.4
    2          64                1              0.2'''

    max_len = choose_max_length(results)

    max_len_dict = {
        'max_len':max_len
    }
    max_len_path = output_dir / "max_len.json"

    import json
    with max_len_path.open('w',encoding='utf-8') as file:   
        json.dump(
        max_len_dict,
        file,
        ensure_ascii=False, # 写中文的格式
        indent=2,# 缩进2格
        )
    
    save_long_text_cases(max_len)
    plot_distribution(pd.read_csv(text_length_root)['text_length'],statistics)