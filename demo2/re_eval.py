import os
import json
import pandas as pd
from tqdm import tqdm

# ====================== 配置部分 ======================
# GPU 配置：指定使用3号显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# HuggingFace 镜像与 Token 配置（之前你提供的）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
HF_TOKEN = "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"
os.environ['HF_TOKEN'] = HF_TOKEN

# 数据集与输出配置
DATASET_PATH = "demo2/log/eval_results_gemma_7b_it_Qwen2.5_7B_Instruct.json"  # 替换为你的数据集路径
OUTPUT_PATH = "demo2/re_eval_res/gemma_7b_it_Qwen2.5_7B_Instruct_Harmbench_cls.json"  # 评测结果保存路径
# ======================================================


def load_dataset(file_path):
    """加载并处理数据集，提取核心字段"""
    print(f"📂 加载数据集: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data.get("results", []):
        sample = {
            "id": item.get("id"),
            "forbidden_prompt": item.get("intent"),  # intent 映射为 forbidden_prompt
            "response": item.get("best_response"),    # best_response 映射为 response
            "success": item.get("success"),
            "best_score": item.get("best_score")
        }
        dataset.append(sample)
    
    df = pd.DataFrame(dataset)
    print(f"✅ 数据集加载完成，共 {len(df)} 条样本")
    return df


def evaluate_with_harmbench(df):
    """使用 HarmBench 评测器批量评测"""
    print("🚀 开始使用 HarmBench 评测器进行评测...")
    
    # 导入 strong_reject 的 evaluate 函数
    from strong_reject.evaluate import evaluate
    
    df["harmbench_score"] = 0.0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评测进度"):
        try:
            # 调用 HarmBench 评测器
            result = evaluate(
                forbidden_prompt=row["forbidden_prompt"],
                response=row["response"],
                evaluators=["harmbench"]  # 核心：使用方案一的 harmbench
                # evaluators=["strongreject_finetuned"]  # 核心：使用方案一的 harmbench
                
            )
            # 提取分数（0–1 之间的连续分数）
            df.loc[idx, "harmbench_score"] = result[0]["score"]
        except Exception as e:
            print(f"❌ 第 {row['id']} 条样本评测失败: {str(e)}")
            df.loc[idx, "harmbench_score"] = -1.0  # 标记失败
    
    avg_score = df[df["harmbench_score"] != -1.0]["harmbench_score"].mean()
    print(f"✅ 评测完成，平均 HarmBench 分数: {avg_score:.4f}")
    return df


def save_results(df, output_path):
    """保存结果到JSON文件"""
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"💾 评测结果已保存至: {output_path}")


def main():
    # 1. 加载数据集
    df = load_dataset(DATASET_PATH)
    
    # 2. 使用 HarmBench 评测
    df = evaluate_with_harmbench(df)
    
    # 3. 保存结果
    save_results(df, OUTPUT_PATH)
    
    print("\n🎉 所有流程执行完毕！")


if __name__ == "__main__":
    main()