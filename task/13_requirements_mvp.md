# SERA 要件定義書 — 非機能要件・受け入れ基準・MVP

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 18. 非機能要件（必須）

### 18.1 言語・ランタイム
- Python >= 3.11
- パッケージマネージャ: pip（pyproject.toml で依存管理）

### 18.2 主要依存ライブラリ
```toml
[project]
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "safetensors>=0.4.0",
    "typer>=0.12.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",           # API呼び出し（async対応）
    "tenacity>=8.2.0",         # リトライ制御
    "matplotlib>=3.8.0",       # 図生成
    "seaborn>=0.13.0",         # 統計可視化
    "graphviz>=0.20.0",        # 探索木可視化
    "numpy>=1.26.0",
    "pydantic>=2.6.0",         # Spec のバリデーション
    "rich>=13.7.0",            # CLIの出力整形
    "structlog>=24.1.0",       # 構造化ログ
]

[project.optional-dependencies]
slurm = ["submitit>=1.5.0"]    # SLURM実行
docker = ["docker>=7.0.0"]     # Docker実行
latex = ["pandoc"]             # LaTeX変換（システムパッケージ）

[project.scripts]
sera = "sera.cli:app"
```

### 18.3 Spec バリデーション
- 全Spec は **Pydantic v2 モデル** で定義し、ロード時に自動バリデーション
- ExecutionSpec は `model_validator` でハッシュ検証を実施

### 18.4 プラグイン設計
- `Executor`（LocalExecutor / SlurmExecutor / DockerExecutor）
- `Evaluator`（StatisticalEvaluator / BootstrapEvaluator）
- `TreeOps`（LLMTreeOps / TemplateTreeOps — draft/debug/improve 3オペレータ）
- `PaperComposer`（MarkdownComposer / LaTeXComposer）
- 各プラグインは ABC（抽象基底クラス）を継承し、`ResourceSpec` / `PlanSpec` で切り替え

### 18.5 エラーハンドリング・リトライ
```text
API呼び出し（Phase 0）:
  - tenacity: retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
  - HTTP 429 → backoff、HTTP 5xx → リトライ、HTTP 4xx → 次のAPI

実験実行（Phase 3）:
  - タイムアウト → status="timeout"、リトライなし（新ノードとして扱う）
  - OOM → status="oom"、リトライなし
  - その他エラー → status="failed"、リトライなし

PPO更新（Phase 5）:
  - NaN/Inf 検出 → 更新スキップ、警告ログ
  - GPU OOM → batch_size を半減してリトライ（最大2回）

中断復帰:
  - Ctrl+C → SIGINT ハンドラで checkpoints/ にスナップショット保存
  - sera research --resume でチェックポイントからロード
  - チェックポイント: open_list, closed_set, best_node, ppo_buffer, step
```

### 18.6 ネットワーク制御
- `ResourceSpec.network.allow_internet` に従う
- Phase 3 の実験中に allow_internet=false の場合、サブプロセスの環境変数 `http_proxy`/`https_proxy` を無効化

### 18.7 決定論的再現
- seed固定: `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- LLM temperature=0 を設定しても完全決定論にはならないため、全LLM入出力をログ保存（agent_llm_log.jsonl）

---

## 19. 受け入れ基準（Acceptance Criteria）

最低限、以下を満たすこと：

1. Input-1 から Input-2（全9 Spec）を生成し、Phase 1で ExecutionSpec を SHA-256 ハッシュ付きで固定できる
2. Semantic Scholar API 優先＋フォールバック（CrossRef→arXiv→Web）が動作する（Phase 0）
3. Best-First探索が動作し、max_nodes 内で複数ノードを生成・評価・選択できる（Phase 2–4）
4. LCB（μ, SE, LCB）が repeats 回の実行から算出され、優先度に反映される（Phase 4）
5. 逐次評価により Top-k ノードのみが全 repeats を実行し、それ以外は early stop する（Phase 4）
6. PPOで LoRA-only 更新が動作し、差分 Δ が adapter_delta.safetensors に保存される（Phase 5）
7. 差分継承の materialize が整合し、depth >= squash_depth で自動 squash される（Phase 6）
8. Pareto 剪定が動作し、支配されるノードが open_list から除去される（Phase 6）
9. best成果物（best_node.json + adapter.safetensors + metrics_summary.json + report.json）が `outputs/best/` に揃う
10. PaperSpec に沿った論文草稿が CI 付き図表を含んで生成される（Phase 7）
11. 自動引用検索ループが Semantic Scholar API で動作し、citation_search_log.jsonl に記録される（Phase 7）
12. ライティング内反省ループが最大 n_writeup_reflections 回動作し、構文エラー・未使用図等を自己修正する（Phase 7）
13. VLM が有効な場合、図記述生成・キャプションレビュー・重複図検出が動作する（Phase 7）
14. PaperScoreSpec に基づき LLM-as-judge 評価が動作し、アンサンブルレビュー（num_reviews_ensemble 件）+レビュアー反省ループが機能する（Phase 8）
15. 改善ループが最大 paper_revision_limit 回回り、Phase 7（内部反省）と Phase 8（外部評価）の二重ループ構造が機能する（Phase 7-8）
16. `docs/` が仕様通り揃い、quickstart 通りに第三者が `pip install` → `sera init` → `sera research` → `sera generate-paper` を再現できる
17. 全 LLM 呼び出しが agent_llm_log.jsonl に記録され、API 呼び出しが queries.jsonl に記録される（再現性）
18. ExecutionSpec の改竄（ハッシュ不一致）を検知し、exit code 2 で停止する
19. ターンレベル報酬が各Phaseで計算され ppo_log に turn_rewards として記録される（Phase A: MT-GRPO）
20. 失敗ノードの知識が FailureKnowledgeExtractor で抽出され、兄弟ノードの failure_context に注入される（Phase B: ECHO軽量版）
21. AgentLLM がツール定義を受け取り GenerationOutput（text + tool_calls）を返す（Phase C: Tool-Calling Agent）

---

## 20. MVP優先順位（実装順）

1) **Phase 0**（先行研究：Semantic Scholar API + フォールバック + queries.jsonl 記録）
2) **Phase 1**（Spec 確定 + ExecutionSpec SHA-256 固定 + Pydantic バリデーション）
3) **Phase 2–4**（Best-First 探索 + LLM分岐生成 + LocalExecutor + LCB 逐次評価）
4) **Phase 5–6**（PPO LoRA-only + delta safetensors + materialize + squash + Pareto 剪定）
5) **Phase 7**（PaperComposer + 図生成 + VLMレビュー + 自動引用検索 + ライティング内反省ループ + EvidenceStore）
6) **Phase 8**（LLM-as-judge アンサンブル評価 + レビュアー反省ループ + 改善ループ）
7) **拡張**（SlurmExecutor / DockerExecutor / LaTeX出力 / Bootstrap評価）
8) **Phase A: MT-GRPO ターン報酬**（前提: Phase 5-6完了）
   - `src/sera/learning/turn_reward.py` 新規実装
   - PPORolloutV2, compute_reward_v2 への拡張
   - PlanSpec.turn_rewards の追加
9) **Phase B: ECHO軽量版 失敗知識注入**（前提: Phase A完了）
   - `src/sera/search/failure_extractor.py` 新規実装
   - SearchNode.failure_context フィールド追加
   - SearchManager.build_context() 拡張
10) **Phase C: HiPER + Tool-Calling Agent**（前提: Phase A+B完了）
   - `src/sera/agent/tool_registry.py` 新規実装
   - `src/sera/learning/hierarchical_ppo.py` 新規実装
   - AgentLLM.generate_with_tools() 実装

---
