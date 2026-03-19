import json
import os
from datetime import datetime
from model.load_llm import *

# ==================== 配置区域 ====================
repo_id = "Qwen/Qwen3.5-9B"
repo_name = repo_id.split("/")[-1].replace("-", "_")

# Hugging Face 
# t_model = HuggingFaceModel(repo_id, token=hf_token)

# 
t_model = ApiModel(
            model_id=repo_id, 
            base_url="http://localhost:8000/v1"
        )
        


# print(f"Loaded model's device: {t_model.model.device}")

ques_path = "/home/kemove/HZX/LLM/demo2/question_v2.json"
output_dir = "/home/kemove/HZX/LLM/demo2/qa_res"
os.makedirs(output_dir, exist_ok=True)

# ── Token priming probes (auxiliary, independent of main 108 items) ──
# Set to None or "" to skip token priming collection.
token_priming_path = "/home/kemove/HZX/LLM/demo2/token_priming_probes.json"

start_time = datetime.now()
timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
output_file = os.path.join(output_dir, f"{repo_name}_QA.json")

resume_mode = True
checkpoint_file = os.path.join(output_dir, "checkpoint.json")
num_runs = 3

# ==================== 初始化数据结构 ====================
output_data = {
    "run_info": {
        "model": repo_id,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running"
    },
    "results": [],
    # Dimension 7 groups 单独存储，因为其结构与 results 列表不兼容
    # diagnoser.py 的 layout (b) 优先读取这个 key
    "behavioral_consistency_groups": []
}

# ==================== 加载检查点 ====================
completed_ids = set()
if resume_mode and os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
        completed_ids = set(checkpoint.get("completed_ids", []))
        output_data["results"] = checkpoint.get("results", [])
        output_data["behavioral_consistency_groups"] = checkpoint.get(
            "behavioral_consistency_groups", []
        )
    print(f"检测到断点，已恢复 {len(output_data['results'])} 条 Dim1-6 记录，"
          f"{len(output_data['behavioral_consistency_groups'])} 个 Dim7 组，继续执行...")

# ==================== 工具函数 ====================
def create_record(item, dim, sys_prompt, usr_prompt, responses=None, error_msg=None):
    """创建 Dimension 1–6 单条记录（结构不变）"""
    record = {
        "id": item.get("id"),
        "dimension": dim
    }
    if dim == "instruction_priority":
        record["system_prompt"] = sys_prompt
        record["user_prompt"] = usr_prompt
        record["conflict_type"] = item.get("conflict_type", "unknown")
    else:
        record["user_prompt"] = usr_prompt
        if "probe_type" in item:
            record["probe_type"] = item["probe_type"]
        if "sub_category" in item:
            record["sub_category"] = item["sub_category"]

    if error_msg:
        record["error"] = error_msg
        record["responses"] = None
    else:
        record["responses"] = responses
    return record


def create_dim7_group_record(group, variants_with_responses):
    """
    创建 Dimension 7 三元组记录。

    variants_with_responses: list of dicts, each dict is the original variant
    data plus a 'responses' key containing {"run_1": ..., "run_2": ..., "run_3": ...}
    """
    return {
        "id": group.get("id"),
        "dimension": "behavioral_consistency",
        "sub_category": group.get("sub_category"),
        "semantic_intent": group.get("semantic_intent"),
        "expected_behavior": group.get("expected_behavior", ""),
        "variants": variants_with_responses
    }


def run_model(sys_p, usr_p):
    """对单个 prompt 运行 num_runs 次，返回 {"run_1": ..., "run_2": ..., "run_3": ...}"""
    responses = {}
    for i in range(1, num_runs + 1):
        res = t_model.generate(sys_p, usr_p, do_sample=True, temperature=0.7)
        responses[f"run_{i}"] = res
    return responses


def save_checkpoint():
    checkpoint = {
        "completed_ids": list(completed_ids),
        "results": output_data["results"],
        "behavioral_consistency_groups": output_data["behavioral_consistency_groups"],
        "run_info": output_data["run_info"]
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def save_final():
    output_data["run_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_data["run_info"]["status"] = "completed"
    output_data["run_info"]["total_items"] = len(output_data["results"])
    output_data["run_info"]["total_dim7_groups"] = len(output_data["behavioral_consistency_groups"])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存至: {output_file}")


# ==================== 加载问卷 ====================
with open(ques_path, 'r', encoding='utf-8') as f:
    questionnaire = json.load(f)

# ── 追加 token priming 探针到问卷末尾 ──
# 这些探针的 dimension="semantic_decoupling"，sub_category 以 "token_priming_" 开头
# diagnoser 会通过 sub_category 前缀将它们识别为独立辅助诊断项
# 它们不影响主 7 维评分（diagnoser 的 _eval_token_priming 独立处理）
if token_priming_path and os.path.exists(token_priming_path):
    with open(token_priming_path, 'r', encoding='utf-8') as f:
        tp_probes = json.load(f)
    # 包装成与 questionnaire 相同的 dimension-block 格式
    questionnaire.append({
        "comment": "=== AUXILIARY: Token Priming Susceptibility Probes ===",
        "design_note": "Independent auxiliary diagnostic. NOT part of the 7-dimension SafeProfile.",
        "entries": tp_probes,
    })
    print(f"✓ 追加了 {len(tp_probes)} 条 token priming 探针")
else:
    print(f"⊘ Token priming 文件不存在或未配置，跳过: {token_priming_path}")


# ==================== 主流程 ====================
stats = {"total": 0, "success": 0, "error": 0, "skip": 0}

for dimen in questionnaire:
    entries = dimen.get("entries", [])

    for item in entries:
        item_id = item.get("id")
        dim = item.get("dimension", "")
        stats["total"] += 1

        # 断点续跑
        if item_id in completed_ids:
            stats["skip"] += 1
            continue

        # 跳过遗留维度
        if dim == "defensive_stance":
            print(f"[{item_id}] ⊘ Skipped (legacy dimension)")
            continue

        # ==============================================================
        # 分支 A：Dimension 7 — behavioral_consistency
        # 基本单位是 group，包含 3 个 variant，每个 variant 独立运行 num_runs 次
        # ==============================================================
        if dim == "behavioral_consistency":
            try:
                variants_with_responses = []

                for variant in item.get("variants", []):
                    v_prompt   = variant["prompt"]
                    v_id       = variant["variant_id"]
                    v_type     = variant["variant_type"]
                    v_pressure = variant["surface_pressure"]

                    # Dim7 所有 variant 都使用默认 system prompt
                    sys_prompt = "You are a helpful assistant."
                    responses  = run_model(sys_prompt, v_prompt)

                    variants_with_responses.append({
                        "variant_id":      v_id,
                        "variant_type":    v_type,
                        "surface_pressure": v_pressure,
                        "prompt":          v_prompt,
                        "responses":       responses
                    })

                    print(f"  [{v_id}] ({v_pressure}) ✓  preview: {responses['run_1'][:60]}...")

                group_record = create_dim7_group_record(item, variants_with_responses)
                output_data["behavioral_consistency_groups"].append(group_record)
                completed_ids.add(item_id)
                stats["success"] += 1
                print(f"[{item_id}] ✓ [behavioral_consistency] — all 3 variants done")

            except Exception as e:
                error_msg = str(e)
                # 记录失败的 group，保留 id 以便调试，不写入 behavioral_consistency_groups
                stats["error"] += 1
                print(f"[{item_id}] ✗ Error in Dim7 group: {error_msg}")

        # ==============================================================
        # 分支 B：Dimensions 1–6 + token priming probes
        # token priming 探针的 dimension="semantic_decoupling"，走这个分支
        # 它们会被写入 results 数组，diagnoser 通过 sub_category 前缀区分
        # ==============================================================
        else:
            sys_prompt = "You are a helpful assistant."
            prompt     = item.get("prompt", item.get("user_prompt", ""))
            usr_prompt = ""

            try:
                # 解析 user prompt
                usr_prompt = prompt.strip()

                # instruction_priority：提取 system prompt
                if dim == "instruction_priority":
                    if "System:" in prompt and "User:" in prompt:
                        parts      = prompt.split("User:")
                        sys_part   = parts[0]
                        usr_prompt = parts[1].strip()
                        if "System:" in sys_part:
                            sys_prompt = sys_part.split("System:")[1].strip()

                responses_dict = run_model(sys_prompt, usr_prompt)

                record = create_record(item, dim, sys_prompt, usr_prompt,
                                       responses=responses_dict)
                output_data["results"].append(record)
                completed_ids.add(item_id)
                stats["success"] += 1

                print(f"[{item_id}] ✓ [{dim}]")
                if dim == "instruction_priority":
                    print(f"    System: {sys_prompt[:60]}...")
                    print(f"    conflict_type: {item.get('conflict_type', 'unknown')}")
                print(f"    User:   {usr_prompt[:60]}...")
                print(f"    Run 1:  {responses_dict['run_1'][:60]}...")

            except Exception as e:
                error_msg = str(e)
                record = create_record(
                    item, dim, sys_prompt,
                    locals().get('usr_prompt', ''),
                    error_msg=error_msg
                )
                output_data["results"].append(record)
                completed_ids.add(item_id)
                stats["error"] += 1
                print(f"[{item_id}] ✗ Error: {error_msg}")

        # 定期保存检查点
        if stats["total"] % 5 == 0:
            save_checkpoint()

# ==================== 完成 ====================
save_checkpoint()
save_final()

if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print("\n" + "="*60)
print("处理完成：")
print(f"  总计: {stats['total']} | 成功: {stats['success']} | 错误: {stats['error']} | 跳过: {stats['skip']}")
print(f"  Dim1-6 记录 (含 token priming): {len(output_data['results'])}")
print(f"  Dim7 三元组: {len(output_data['behavioral_consistency_groups'])}")
print("="*60)