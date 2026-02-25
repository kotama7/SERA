# SERA 要件定義書 — CLI・ログ仕様

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 16. CLI仕様（必須）

### 16.1 実装
- フレームワーク: **Typer**（型ヒント駆動、自動ヘルプ生成）
- エントリポイント: `sera` コマンド（`pyproject.toml` の `[project.scripts]` で定義）

### 16.2 コマンド
```bash
# Phase 0-1: 初期化
sera init input1.yaml                # Input-1 読み込み、workspace作成
sera phase0-related-work             # Phase 0: 先行研究収集
sera freeze-specs                    # Phase 1: 全Spec確定、ExecutionSpec固定

# Phase 2-6: 研究ループ
sera research                        # Phase 2-6 統合ループ実行
sera research --resume               # チェックポイントから再開

# 出力
sera export-best                     # best成果物を outputs/best/ に集約

# Phase 7-8: 論文
sera generate-paper                  # Phase 7: 論文生成
sera evaluate-paper                  # Phase 8: 論文評価・改善ループ

# ユーティリティ
sera status                          # 現在の探索状態サマリ表示
sera show-node <node_id>             # ノード詳細表示
sera replay --node-id <id> --seed <s>  # 特定ノードの実験再実行
sera validate-specs                  # Spec整合性チェック
```

### 16.3 引数（全コマンド共通）
```bash
# Phase 0 関連
--topk 20                  # top_k_papers
--teacher-papers 8         # teacher_papers
--citation-depth 2         # citation_graph_depth
--years-bias 3             # recent_years_bias
--api-priority "semantic_scholar,crossref,arxiv"

# Phase 2 関連
--max-nodes 200
--max-depth 12
--branch-factor 5
--lambda-cost 0.2
--beta 0.1

# Phase 4 関連
--repeats 5
--lcb-coef 2.576           # 99%CI
--no-sequential
--seq-topk 10

# Phase 5 関連
--rank 32
--alpha 64
--lr 5e-5
--clip 0.1
--ppo-steps 256
--no-kl

# モデル関連
--base-model "meta-llama/Llama-3.1-8B-Instruct"
--dtype "4bit"
--agent-llm "openai:gpt-4o"    # provider:model_id 形式

# リソース関連
--executor "slurm"
--gpu-count 2                  # GPU数（ComputeConfig.gpu_count）
--memory-gb 64                 # メモリGB（ComputeConfig.memory_gb）
--cpu-cores 16                 # CPUコア数（ComputeConfig.cpu_cores）
--gpu-type "A100"              # GPU種別制約（ComputeConfig.gpu_type）
--no-gpu-required              # GPU不要（ComputeConfig.gpu_required=False）
--work-dir "./my_workspace"
--timeout 7200
--no-web
--auto                      # Phase 1 のユーザ承認をスキップ
```

**重要**：引数反映は `freeze-specs` 実行まで。以後はSpec固定で動く。

### 16.4 終了コード
| コード | 意味 |
|--------|------|
| 0 | 正常終了 |
| 1 | 一般エラー（設定ミス、ファイル不在等） |
| 2 | ExecutionSpec 改竄検知 |
| 3 | adapter_spec_hash 不整合 |
| 10 | Phase 0 全API失敗 |
| 11 | Phase 3 全実験失敗（有効ノードゼロ） |
| 12 | 予算超過による強制終了 |
| 20 | ユーザによる中断（Ctrl+C → チェックポイント保存後終了） |

---

## 17. ログ仕様（必須）

### 17.1 search_log.jsonl
```json
{
  "event": "node_selected",
  "node_id": "uuid-...",
  "parent_id": "uuid-...",
  "depth": 3,
  "priority": 0.72,
  "lcb": 0.68,
  "chosen_reason": "highest_priority",
  "open_list_size": 15,
  "total_nodes": 42,
  "budget_consumed": {"gpu_minutes": 23.5},
  "timestamp": "2026-02-21T12:00:00Z"
}
```

### 17.2 eval_log.jsonl
```json
{
  "event": "evaluation_complete",
  "node_id": "uuid-...",
  "repeat_idx": 2,
  "metrics": {"primary": {"name": "score", "value": 0.73}, "constraints": [...]},
  "mu": 0.71,
  "se": 0.015,
  "lcb": 0.681,
  "ucb": 0.739,
  "n_repeats_done": 3,
  "cost_sec": 125.3,
  "timestamp": "2026-02-21T12:05:00Z"
}
```

### 17.3 ppo_log.jsonl
```json
{
  "event": "ppo_update",
  "adapter_node_id": "adapter-uuid-...",
  "parent_adapter_node_id": "adapter-uuid-...",
  "search_node_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "mean_reward": 0.65,
  "kl_divergence": 0.018,
  "policy_loss": -0.023,
  "value_loss": 0.15,
  "entropy": 2.3,
  "kl_coef_current": 0.01,
  "total_steps": 128,
  "delta_norm_l2": 0.0023,
  "timestamp": "2026-02-21T12:10:00Z"
}
```

### 17.4 paper_log.jsonl
```json
{
  "event": "paper_evaluation",
  "iteration": 1,
  "overall_score": 6.8,
  "scores": {"statistical_rigor": 7, "baseline_coverage": 8, "ablation_quality": 5, ...},
  "passed": false,
  "missing_items": [{"category": "ablation_quality", "description": "..."}],
  "actions_taken": ["added_ablation_experiment", "revised_introduction"],
  "additional_experiments_run": 2,
  "timestamp": "2026-02-21T13:00:00Z"
}
```

### 17.5 agent_llm_log.jsonl（新規）
```json
{
  "event": "llm_call",
  "call_id": "call-uuid-...",
  "phase": "phase2",
  "purpose": "branch_generation",
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "adapter_node_id": "adapter-uuid-...",
  "prompt_tokens": 1523,
  "completion_tokens": 412,
  "temperature": 0.7,
  "prompt_hash": "sha256:...",
  "response_hash": "sha256:...",
  "latency_ms": 2340,
  "tool_calls": null,
  "turn_rewards": null,
  "timestamp": "2026-02-21T12:01:00Z"
}
```

tool-calling 有効時（Phase C）の agent_llm_log エントリ例：
```json
{
  "event": "llm_call",
  "call_id": "call-uuid-...",
  "phase": "phase3",
  "purpose": "experiment_code",
  "model_id": "Qwen/Qwen3-8B",
  "adapter_node_id": "adapter-uuid-...",
  "prompt_tokens": 2048,
  "completion_tokens": 856,
  "temperature": 0.7,
  "prompt_hash": "sha256:...",
  "response_hash": "sha256:...",
  "latency_ms": 3120,
  "tool_calls": [
    {"tool_name": "search_api", "arguments": {"query": "pytorch distributed training"}, "reasoning": "Need reference implementation"}
  ],
  "turn_rewards": {"phase2": 0.8, "phase3": 1.0},
  "timestamp": "2026-02-21T12:01:00Z"
}
```

---
