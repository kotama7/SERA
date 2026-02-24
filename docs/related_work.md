# SERA 関連研究サーベイ

本文書では、SERAの設計に影響を与えた先行研究を体系的に整理し、各研究がSERAのどのフェーズに対応するかをマッピングテーブルとともに示す。


## A-1: 自律研究 / 論文自動生成 (AI-Scientist 系)

### AI-Scientist (Lu et al., 2024)

**参照:** Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." arXiv:2408.06292.

AI-Scientistは、基盤モデルを用いた端対端の自律的科学研究システムである。アイデア生成から実験実装、結果分析、研究論文の完成までを自動化する。

**SERAとの対応関係:**

- SERAのPhase 7--8（論文生成と評価ループ）の設計に影響を与えた
- `PaperComposer`（`src/sera/paper/paper_composer.py`）が実装する6ステップの論文執筆パイプライン（ログ要約、プロット集約、引用検索、VLM図表記述、本文生成+リフレクション、最終統合）は、AI-Scientistの「実験結果から論文を自動生成する」というコンセプトを踏襲している
- `PaperEvaluator`（`src/sera/paper/paper_evaluator.py`）のアンサンブル評価+修正ループも、AI-Scientistの反復的改善サイクルから着想を得ている

**SERAとの主な差異:**

- AI-Scientistはツリー探索を用いない線形パイプラインであるのに対し、SERAはベストファースト木探索を中核とする
- AI-Scientistにはパラメータ効率的なモデル適応（LoRA/PPO）の仕組みがない
- AI-Scientistには統計的に厳密な評価（LCBベースのスコアリング）がない

---

### AI-Scientist-v2 (Yamada et al., 2025)

AI-Scientist-v2は、初代AI-Scientistを大幅に拡張したシステムであり、以下の重要な改良を導入した。

**SERAとの対応関係:**

| AI-Scientist-v2 の機能 | SERAの対応箇所 | 実装詳細 |
|----------------------|--------------|---------|
| エージェント型ツリー探索 | Phase 2 (`SearchManager`) | `src/sera/search/search_manager.py` にベストファースト探索ループを実装。ノード選択、評価、分岐を統合的に管理 |
| 管理エージェント | Phase 2 (`select_next_node`) | ノード状態に基づくオペレータ自動選択（pending→evaluate、failed→debug、evaluated→improve）を実装 |
| VLM統合による図表レビュー | Phase 7 (`VLMReviewer`) | `src/sera/paper/vlm_reviewer.py` にOpenAI/Anthropicのビジョンモデルを用いた図表品質評価・重複検出を実装 |
| 執筆時のリフレクションループ | Phase 7 (`PaperComposer._step5_paper_body`) | `n_writeup_reflections`回（デフォルト3回）のリフレクションループで、図表参照漏れ・引用キー不整合・セクション欠落を検出して修正 |
| アンサンブルレビューア | Phase 8 (`PaperEvaluator`) | `num_reviews_ensemble`名の独立レビューア（bias_modeでcritical/generousを交互切替）、各レビューアは`num_reviewer_reflections`回のリフレクションを実施 |
| メタレビュー | Phase 8 (`_generate_meta_review`) | Area Chairスタイルのメタレビューを生成し、レビュー間のコンセンサスと不一致を統合 |
| 引用検索ループ | Phase 7 (`CitationSearcher`) | `src/sera/paper/citation_searcher.py` に最大20ラウンドの反復的引用発見ループを実装。LLMが不足引用を特定し、Semantic Scholar APIで検索、最適な論文を選択してBibTeXを生成 |


## A-2: 自律コード実験

### CodeScientist (Ifargan et al., 2024)

**参照:** Ifargan, T., Hafner, L., Kern, M., & Gerstenberg, T. (2024). "CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-Capable LLMs."

CodeScientistは、コード生成能力を持つLLMを用いた自律的な科学実験パイプラインであり、コード生成→実行→評価のループを自動化する。

**SERAとの対応関係:**

- **Phase 3（実験コード生成とサンドボックス実行）:** `ExperimentGenerator`（`src/sera/execution/experiment_generator.py`）がLLMを用いて仮説と実験設定から完全な実験スクリプトを生成する。`EXPERIMENT_CODE_PROMPT`テンプレートにより、シード引数受付、メトリクスJSON出力、エラーハンドリングなどの要件をLLMに伝達する。生成されたスクリプトは`Executor`（Local/SLURM/Docker）でサンドボックス実行される
- **Phase 4（メトリクスベースの評価）:** `StatisticalEvaluator`（`src/sera/evaluation/statistical_evaluator.py`）が実験出力のmetrics.jsonを解析し、平均値(mu)・標準誤差(SE)・下側信頼限界(LCB)を計算する

---

### AIDE (Weco AI, 2025)

**参照:** Jiang, W., et al. (2024). "AIDE: An LLM Agent for Data Science." arXiv:2502.13138.

AIDEは、コード空間におけるAI駆動探索フレームワークであり、ベストファーストツリー探索を用いてデータサイエンス問題を解く。3つのオペレータ（Drafting、Debugging、Improving）を定義し、候補解のツリーを展開する。

**SERAはAIDEの3オペレータ設計を直接採用し、以下のように拡張している:**

| AIDEの概念 | SERAの対応箇所 | 拡張内容 |
|-----------|--------------|---------|
| ノード = Pythonスクリプト + メトリクス | `SearchNode` = 仮説 + `experiment_config` + `metrics_raw` + `adapter_node_id` | LoRAリネージへの参照を追加 |
| Drafting（新規解の生成） | `TreeOps.draft`（`src/sera/search/tree_ops.py`） | ルート初期化時に3カテゴリ（baseline / open_problem / novel）に分割して生成。関連研究コンテキスト（`baseline_candidates`、`open_problems`）を注入 |
| Debugging（エラー修復） | `TreeOps.debug` | 深さ制限付き（`max_debug_depth=3`、`SearchConfig`で設定可能）。実験コード+エラーメッセージをLLMに提示して修正コードを生成 |
| Improving（既存解の改善） | `TreeOps.improve` | 1-2変数のアトミックな変更のみ許可。変更キー数が1を超えると警告ログ出力。LCBベースの統計的判定とシブリングコンテキスト（`_build_sibling_context`で構築：同一親の評価済みノードをLCB降順でtop-k表示、グローバルベストノード含む）を活用 |
| 最良の有効解を選択 | LCBベースの優先度関数（`compute_priority`） | イプシロン制約法（実行不可能ノードは`-inf`）+ コストペナルティ(`lambda_cost`) + 探索ボーナス(`beta_exploration`) |
| 実メトリクスで実行 | 反復実行 + mu/SE/LCB計算 | 2段階逐次評価：`evaluate_initial`（初期seeds、デフォルト1回）→ top-kのみ`evaluate_full`（デフォルト3回） |


## A-3: 探索（ツリー探索 / エージェント的探索）

### ベストファースト探索

古典的な探索アルゴリズムであり、常に最も有望なノードを展開する。

**SERAの実装（`src/sera/search/priority.py`、`src/sera/search/search_manager.py`）:**

優先度の計算式:

```
priority = LCB - lambda_cost * total_cost + beta_exploration * exploration_bonus
```

ここで:
- `LCB`: ノードの下側信頼限界（後述）
- `lambda_cost`: コストペナルティ係数（デフォルト 0.1、`SearchConfig.lambda_cost`）
- `beta_exploration`: 探索ボーナス係数（デフォルト 0.05、`SearchConfig.beta_exploration`）
- `exploration_bonus`: `1 / sqrt(eval_runs + 1)`（UCB1スタイル）

特殊ケース:
- 実行不可能ノード（`feasible == False`）: `priority = -inf`（展開されない）
- 未評価ノード（`lcb is None`）: `priority = +inf`（優先的に探索）

ノード選択は最小ヒープ（優先度の符号反転）で管理され、`select_next_node`がノード状態に基づいてオペレータを自動選択する。

---

### 多腕バンディット / LCB (Lower Confidence Bound)

**参照:**
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning.
- Lai, T. L., & Robbins, H. (1985). "Asymptotically Efficient Adaptive Allocation Rules." Advances in Applied Mathematics.

**SERAの実装（`src/sera/evaluation/statistical_evaluator.py`）:**

LCB計算式:

```
LCB = mu - c * SE
```

ここで:
- `mu`: 複数シード実行のメトリクス平均値
- `SE`: 標準誤差 = `sqrt(variance / n)`
- `c`: LCB係数（デフォルト 1.96、95%信頼区間に対応、`SearchConfig.lcb_coef`）

`update_stats`関数（`statistical_evaluator.py`内）がノードの`metrics_raw`から値を抽出し、`mu`、`se`、`lcb`を計算する。`eval_runs == 1`の場合は`se = inf`、`lcb = -inf`となり、十分なデータが揃うまで保守的な評価を行う。

**SERAにおけるLCBの使用箇所:**

| 使用箇所 | フェーズ | 詳細 |
|---------|--------|------|
| ノード選択（優先度計算） | Phase 2 | `compute_priority`でLCBを主要成分として使用 |
| ベストノード判定 | Phase 2 | `_update_best`でLCBが最大（同値の場合muで決着）のノードを選択 |
| 枝刈り閾値 | Phase 6 | `_lcb_threshold_prune`: 明示的閾値がない場合、自動的に `best_lcb * 0.5` を閾値として使用 |
| 実行可能性チェック | Phase 4 | `check_feasibility`（イプシロン制約法）: 実行不可能ノードの優先度を`-inf`に設定 |
| Top-k選択（逐次評価） | Phase 4 | `_is_topk`でLCB順にtop-kノードを選び、フル評価を実施 |


## A-4: LoRA / PEFT / 継続学習 / アダプタ管理

### LoRA (Hu et al., 2022)

**参照:** Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

LoRAは、ベースモデルの重みを凍結し、低ランク分解 A*B のみを学習するパラメータ効率的なファインチューニング手法である。

**SERAの実装:**

- Phase 5の`PPOTrainer`（`src/sera/learning/ppo_trainer.py`）がLoRAパラメータのみを更新する。`lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]`で対象パラメータを特定
- LoRAのrank、alpha、target_modules、dropoutは`ModelSpec.adapter_spec`で設定可能
- Phase 6の`LineageManager`（`src/sera/lineage/lineage_manager.py`）がLoRAアダプタのリネージツリーを管理

---

### PEFT (HuggingFace)

**参照:** Mangrulkar, S., et al. (2022). "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods." HuggingFace library.

PEFTは、LoRAを含む各種パラメータ効率的ファインチューニング手法の統一的なインターフェースを提供するライブラリである。

**SERAにおけるPEFT APIの使用箇所:**

- `peft.get_peft_model_state_dict`: PPO更新前後の重み抽出（`_ppo_update_core`内）およびデルタ抽出（`LineageManager.extract_delta_from_model`内）
- `peft.get_peft_model`: モデルへのアダプタ適用
- `peft.set_peft_model_state_dict`: マテリアライズされた重みのモデルへの復元
- vLLMエクスポート（`LineageManager.export_for_vllm`）: `adapter_config.json`にpeft互換の設定（`peft_type: "LORA"`、`task_type: "CAUSAL_LM"`等）を出力

---

### デルタ継承 (Delta Inheritance)

**重要:** これはDelta-LoRA (Zi et al., 2023) とは異なるメカニズムである。Delta-LoRAは低ランクデルタを用いてフルランクの事前学習済み重みを更新するが、SERAのデルタ継承はアダプタ重みの差分のみを保存・合成する仕組みである。

**SERAの実装（`src/sera/lineage/lineage_manager.py`）:**

`LineageManager`が管理するデルタ継承の仕組み:

1. **デルタ保存 (`save_delta`):** 子アダプタは `delta = child_weights - parent_weights` のみを保存する。safetensors形式で`adapter_delta.safetensors`に格納
2. **マテリアライズ (`materialize`):** ルートからノードまでのリネージパスを構築（`build_lineage_path`）し、デルタを順次加算して完全な重みを復元する。LRUキャッシュ（`LRUCache`、デフォルト10エントリ）で高速化
3. **スナップショット/スカッシュ (`maybe_squash`):** 設定可能な深さ閾値（デフォルト: `max_depth // 2`、`SearchConfig.squash_depth`）を超えたノードに対し、マテリアライズ結果を`adapter_snapshot.safetensors`に保存。以後のマテリアライズはスナップショットを起点とすることで、再構成コストを抑制
4. **互換性検証 (`validate_compatibility`):** `adapter_spec_hash`の一致と、テンソル形状の一致をチェックし、デルタ合成の安全性を保証

ディレクトリ構造:

```
lineage/
  nodes/
    <adapter_node_id>/
      meta.json                    # メタデータ（parent_id、depth、adapter_spec_hash等）
      adapter_delta.safetensors    # 親からのデルタ
      adapter_snapshot.safetensors # 完全な重み（スカッシュ後のみ）
```

---

### アダプタ継続学習

**参照:**
- Ke, Z., et al. (2021). "Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks." NAACL 2021.
- Wang, Z., et al. (2023). "Rehearsal-Free Continual Language Learning via Efficient Parameter Isolation." ACL 2023.

**SERAとの対応関係:**

- 各PPO更新が新しいリネージノードを生成し、探索ブランチ間でアダプタを共有する
- パフォーマンスが劣化した場合、親アダプタへのロールバックが可能（リネージツリーの任意のノードをマテリアライズ可能）
- ブランチ構造により、異なるサブタスク向けのアダプタ特化が可能


## A-5: 強化学習 / PPO

### PPO (Schulman et al., 2017)

**参照:** Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

PPOは、クリッピングされたサロゲート目的関数を用いた近接方策最適化アルゴリズムである。

**SERAの実装（`src/sera/learning/ppo_trainer.py`）:**

`PPOTrainer`は以下の仕様でLoRAパラメータのみを更新する:

- **サロゲート損失:** `L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)` ここで`clip_range`はデフォルト0.2（`LearningConfig.clip_range`）
- **価値関数損失:** MSE損失（係数`value_loss_coef = 0.5`）
- **エントロピーボーナス:** `trl.trainer.utils.entropy_from_logits`を使用（係数`entropy_coef = 0.01`）
- **勾配クリッピング:** `accelerate.Accelerator.clip_grad_norm_`（`max_grad_norm = 1.0`）
- **GAE (Generalised Advantage Estimation):** `gamma = 0.99`、`gae_lambda = 0.95`。各ロールアウトは独立した単一ステップエピソードとして扱われるため、`advantage = reward - value`、`returns = reward`
- **適応的KL制御:** `kl > kl_target * 1.5`の場合`kl_coef *= 2`、`kl < kl_target / 1.5`の場合`kl_coef /= 2`
- **PPOトリガー:** `ppo_trigger_interval`（デフォルト5）回の評価済みノードごとに更新。プラトー検出（`plateau_patience`ステップ間改善なし）でも追加トリガー

**報酬関数（`src/sera/learning/reward.py`）:**

```
R = primary_value
    - constraint_penalty * num_violated_constraints
    - lambda_cost * normalized_cost
    - kl_coef * kl_divergence
```

- 失敗/タイムアウト/OOMノード: ハードコード報酬 `-100.0`（`_FAILURE_REWARD`）
- コスト正規化: `min(total_cost / budget_limit, 1.0)` で [0, 1] 範囲に正規化
- `budget_limit`は`termination.max_wall_time_hours * 3600`から導出（デフォルト14400秒）

**vLLMとの協調:** PPO更新前に`vllm_engine.sleep()`でGPUメモリを解放し、更新後に`vllm_engine.wake()`で復帰する。

---

### TRL (HuggingFace)

**参照:** HuggingFace TRL (Transformer Reinforcement Learning) ライブラリ。

**SERAにおけるTRLの使用:**

- `trl.trainer.utils.entropy_from_logits`: エントロピー計算に使用。ロジットテンソルからトークンレベルのエントロピーを算出（`_ppo_update_core`内）
- これにより、手動実装の近似ではなく、テスト済みの実装によるエントロピーボーナスを実現


## A-6: 学術API / 文献収集

### Semantic Scholar API (Allen AI)

**参照:** Kinney, R., et al. (2023). "The Semantic Scholar Open Data Platform." arXiv:2301.10140.

**SERAの実装:**

- **Phase 0（主要文献ソース）:** `SemanticScholarClient`（`src/sera/phase0/api_clients/semantic_scholar.py`）がGraph API v1（`https://api.semanticscholar.org/graph/v1`）を使用。論文検索（`/paper/search`）、参照取得（`/paper/{id}/references`）、引用取得（`/paper/{id}/citations`）をサポート。`tenacity`によるリトライ（最大5回、指数バックオフ）を実装
- **Phase 7（自動引用検索）:** `CitationSearcher`（`src/sera/paper/citation_searcher.py`）が論文執筆時にSemantic Scholar APIを使用し、不足引用を反復的に発見する

---

### CrossRef API

**参照:** Hendricks, G., et al. (2020). "CrossRef: The Sustainable Source of Community-Owned Scholarly Metadata."

**SERAの実装（`src/sera/phase0/api_clients/crossref.py`）:**

- DOIベースのメタデータを提供する二次ソースとして使用
- `https://api.crossref.org/works` エンドポイントで論文検索
- 引用/参照グラフのエンドポイントは提供しないため、`get_references`/`get_citations`は空リストを返す

---

### arXiv API

**参照:** Warner, S., et al. (2021). "arXiv API Access." arXiv documentation.

**SERAの実装（`src/sera/phase0/api_clients/arxiv.py`）:**

- プレプリントアクセスのための三次ソースとして使用
- Atom XML APIを使用し、クライアント側で年フィルタリングを実施（arXiv APIはサーバー側の年フィルタを提供しない）
- レート制限遵守: リクエスト間に最低3秒の待機時間（`_MIN_DELAY_SECONDS = 3.0`）
- 引用/参照グラフは提供しない

---

### SerpAPI (Web検索フォールバック)

**SERAの実装（`src/sera/phase0/api_clients/web_search.py`）:**

- Google Scholar検索へのラストリゾートフォールバックとして使用
- `WebSearchClient`が`https://serpapi.com/search`（`engine: google_scholar`）を使用
- 引用/参照グラフは提供しない

---

### Phase 0パイプライン全体

`RelatedWorkEngine`（`src/sera/phase0/related_work_engine.py`）が上記APIクライアントを統合し、以下の6ステップを実行する:

1. LLM（またはフォールバックヒューリスティック）による検索クエリ生成
2. 複数APIクライアントによるフォールバック付き検索
3. Semantic Scholarの引用グラフによる論文集合の拡張（`citation_graph_depth`で制御）
4. 引用数+関連度スコアによるランキング（`rank_papers`）
5. テーマ別クラスタリング（`cluster_papers`）
6. 結果のSpec化（`RelatedWorkSpec`、`PaperSpec`、`PaperScoreSpec`、`TeacherPaperSet`）


## マッピングテーブル: 先行研究とSERAフェーズの対応

| 先行研究 | 対応SERAフェーズ | SERAにおける使用内容 |
|---------|----------------|-------------------|
| AI-Scientist | Phase 7--8 | 論文生成パイプライン（`PaperComposer`の6ステップ）と評価改善ループ（`PaperEvaluator`のアンサンブル評価+修正） |
| AI-Scientist-v2 | Phase 2, 7, 8 | ツリー探索アーキテクチャ（`SearchManager`）、VLMレビュー（`VLMReviewer`）、執筆リフレクション（`_step5_paper_body`の反復修正）、引用検索ループ（`CitationSearcher`）、アンサンブルレビュー+リフレクション（`PaperEvaluator`）、メタレビュー（`_generate_meta_review`） |
| CodeScientist | Phase 3--4 | LLMによる実験コード生成（`ExperimentGenerator`）とメトリクスベースの評価（`StatisticalEvaluator`） |
| AIDE | Phase 2 | 3オペレータ設計: draft/debug/improve（`TreeOps`）、ベストファーストノード選択 |
| ベストファースト探索 | Phase 2 | 優先度ベースのノード選択（`compute_priority`、最小ヒープ管理） |
| LCB / 多腕バンディット | Phase 2, 4, 6 | 統計評価（`update_stats`: mu - c*SE）、優先度計算（`compute_priority`）、枝刈り閾値（`_lcb_threshold_prune`: auto = best_lcb * 0.5） |
| LoRA / PEFT | Phase 5--6 | LoRAアダプタ学習（`PPOTrainer`でLoRAパラメータのみ更新）、peft APIによるアダプタ重み管理（`get_peft_model_state_dict`）、vLLMエクスポート |
| デルタ継承 | Phase 6 | デルタ保存（`save_delta`）、マテリアライズ（`materialize`: ルートからのデルタ加算）、スカッシュ（`maybe_squash`: 深さ閾値超過時にスナップショット作成） |
| PPO / TRL | Phase 5 | LoRA限定PPO更新（`_ppo_update_core`）、適応的KL制御、`trl.trainer.utils.entropy_from_logits`によるエントロピー計算、`accelerate.Accelerator`による勾配操作 |
| Semantic Scholar | Phase 0, 7 | 文献収集（`SemanticScholarClient`: 検索+引用グラフ展開）、論文執筆時の自動引用検索（`CitationSearcher`） |
| CrossRef | Phase 0 | DOIベースのメタデータ取得（二次ソース、`CrossRefClient`） |
| arXiv | Phase 0 | プレプリントアクセス（三次ソース、`ArxivClient`、Atom XML API） |
| SerpAPI | Phase 0 | Google Scholar検索フォールバック（ラストリゾート、`WebSearchClient`） |
| アダプタ継続学習 | Phase 5--6 | リネージツリーによるアダプタ分岐・ロールバック・特化（`LineageManager`） |


## 8フェーズパイプラインの凡例

| フェーズ | 名称 | 主要な先行研究の影響 |
|--------|------|-------------------|
| Phase 0 | 関連研究収集 | Semantic Scholar、CrossRef、arXiv、SerpAPI |
| Phase 1 | Spec凍結 | (SERAの独自設計: 三層分離の不変性保証) |
| Phase 2 | ベストファーストツリー探索 | AIDE、AI-Scientist-v2、ベストファースト探索、LCB/バンディット |
| Phase 3 | 実験実行 | CodeScientist |
| Phase 4 | 統計評価 | LCB/バンディット、CodeScientist |
| Phase 5 | PPO学習 | PPO、LoRA、PEFT、TRL、アダプタ継続学習 |
| Phase 6 | リネージ管理 | デルタ継承、LCB（枝刈り閾値）、アダプタ継続学習 |
| Phase 7 | 論文生成 | AI-Scientist、AI-Scientist-v2（VLM、リフレクション、引用検索）、Semantic Scholar |
| Phase 8 | 論文評価 | AI-Scientist-v2（アンサンブルレビュー+リフレクション、メタレビュー） |
