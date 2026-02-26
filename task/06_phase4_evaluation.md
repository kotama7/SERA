# SERA 要件定義書 — Phase 4: 統計評価

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 8. Phase 4：統計評価（Evaluator）

### 8.1 統計値（必須）
- `repeats` 回実行（または測定）して：
  - 平均 `μ = mean(values)`
  - 標準誤差 `SE = std(values) / sqrt(n)`
  - `LCB = μ - c * SE`（c = ExecutionSpec.evaluation.lcb_coef）
  - `UCB = μ + c * SE`（参考記録用、eval_log.jsonl に `ucb` フィールドとして記録）

### 8.2 逐次評価（sequential eval：具体アルゴリズム）
```python
def evaluate_node_sequential(node: SearchNode, executor: Executor, exec_spec: ExecutionSpec):
    """
    逐次評価アルゴリズム:
    1. 初回: sequential_eval_initial 回（既定1回）だけ実行
    2. μ, SE, LCB を暫定計算（n=1の場合 SE=inf, LCB=-inf）
    3. 全open_listノードのLCBでソートし、Top-k（sequential_eval_topk=5）に入るか判定
    4. Top-kに入るなら、repeats（既定3回）まで追加実行
    5. 追加実行ごとに μ, SE, LCB を再計算
    """
    # Step 1: 初回実行
    for i in range(exec_spec.evaluation.sequential_eval_initial):
        result = executor.run(node.node_id, seed=base_seed + i)
        node.metrics_raw.append(load_metrics(result.metrics_path))
        node.eval_runs += 1

    update_stats(node, exec_spec)  # μ, SE, LCB 計算

def evaluate_node_full(node: SearchNode, executor: Executor, exec_spec: ExecutionSpec):
    """Top-kノードに対して残りの反復を実行"""
    remaining = exec_spec.evaluation.repeats - node.eval_runs
    for i in range(remaining):
        result = executor.run(node.node_id, seed=base_seed + node.eval_runs + i)
        node.metrics_raw.append(load_metrics(result.metrics_path))
        node.eval_runs += 1

    update_stats(node, exec_spec)

def update_stats(node: SearchNode, exec_spec: ExecutionSpec):
    values = [m["primary"]["value"] for m in node.metrics_raw if m is not None]
    n = len(values)
    if n == 0:
        node.mu, node.se, node.lcb = None, None, None
        return
    node.mu = sum(values) / n
    if n >= 2:
        variance = sum((v - node.mu) ** 2 for v in values) / (n - 1)
        node.se = (variance ** 0.5) / (n ** 0.5)
    else:
        node.se = float('inf')
    node.lcb = node.mu - exec_spec.evaluation.lcb_coef * node.se
```

### 8.3 多指標（必須）
- primary / constraints / secondary を区別
- 制約は ε-constraint として feasibility 判定に使用：
  ```python
  def check_feasibility(node: SearchNode, problem_spec: ProblemSpec) -> bool:
      for constraint in problem_spec.constraints:
          metric = find_constraint_metric(node.metrics_raw, constraint.name)
          if constraint.type == "bool":
              if not metric["satisfied"]:
                  return False
          elif constraint.type == "ge":
              if metric["value"] < constraint.threshold - constraint.epsilon:
                  return False
          elif constraint.type == "le":
              if metric["value"] > constraint.threshold + constraint.epsilon:
                  return False
      return True
  ```
- secondary はタイブレーク時に使用：
  - LCBが同値（差が `plateau_min_improvement` 以下）の場合、secondary の加重和で順位決定

---
