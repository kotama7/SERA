# SERA 要件定義書 — Phase 2: 探索木生成

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 6. Phase 2：探索木生成（AIDE参考 Best-First）

> **設計参考**: AIDE (AI-Driven Exploration in the Space of Code, Weco AI, 2025) の探索木アーキテクチャを参考とする。AIDEはML工学をコード空間の木探索として定式化し、Drafting / Debugging / Improving の3オペレータで解空間を探索する。SERAはこの3オペレータ設計を採用しつつ、統計評価（LCB）・LoRA差分継承・PPO学習を追加した拡張版として設計する。

### 6.1 AIDE との対応関係

| AIDE の概念 | SERA での対応 | 拡張点 |
|------------|-------------|--------|
| ノード = Pythonスクリプト + メトリクス | ノード = 仮説 + experiment_config + メトリクス + LoRA参照 | LoRA系譜の追加 |
| Drafting（新規解の起草） | `draft` オペレータ（ルート生成 + 新アプローチ導入） | 先行研究ベースの初期化 |
| Debugging（エラー修復） | `debug` オペレータ（実験失敗時の修復、深度制限付き） | AIDE同様の深度制限 |
| Improving（原子的改善） | `improve` オペレータ（単一変数の測定可能な変更） | LCBベースの統計的判断 |
| 最良の有効解を選択 | LCBベースの優先度関数 | ε-constraint + 多指標 |
| 実行して実メトリクスで評価 | 反復実行 + μ/SE/LCB | 逐次評価の追加 |

### 6.2 ノード定義（必須）
各探索ノード `node_id` は以下を持つ：

```python
@dataclass
class SearchNode:
    node_id: str                    # UUID v4
    parent_id: str | None           # ルートは None
    depth: int
    created_at: str                 # ISO 8601 UTC

    # 外部状態（AIDE: 各ノードが1つの完全な解を表す）
    hypothesis: str                 # 自然言語の仮説（例："learning rate を 1e-3 にすると精度向上"）
    experiment_config: dict         # 操作変数の具体値 {"lr": 1e-3, "batch_size": 32, ...}
    experiment_code: str | None     # LLM生成の実験コード（Phase 3 で設定）
    branching_op: str               # 適用されたオペレータ名: "draft" | "debug" | "improve"
    rationale: str                  # LLM が生成した分岐理由

    # 失敗知識コンテキスト（ECHO軽量版、§26.4.3 — 実装済み）
    failure_context: list[dict] = field(default_factory=list)
    # 失敗兄弟ノードから注入された知識（§6.8.2）
    # 各要素は FailureSummary.to_dict() 形式:
    #   {"node_id": str, "hypothesis": str, "error_category": str,
    #    "error_message": str, "lesson": str}

    # 内部状態参照
    adapter_node_id: str | None     # LoRA系譜ノードID（Phase 5 後に設定）

    # 評価統計
    eval_runs: int                  # 実行済み反復数
    metrics_raw: list[dict]         # 各反復の metrics.json
    mu: float | None                # primary の平均
    se: float | None                # 標準誤差
    lcb: float | None               # LCB = mu - c*se

    # コスト
    total_cost: float               # 累計コスト（秒 or GPU分）
    wall_time_sec: float            # 実壁時間

    # 探索制御
    priority: float | None          # 計算済み優先度
    status: str                     # "pending" | "running" | "evaluated" | "failed" | "pruned" | "expanded"
    children_ids: list[str]         # 子ノードID群
    feasible: bool                  # ε-constraint を満たすか
    debug_depth: int                # debug 連鎖の深さ（AIDE参考: 深度制限用）
    error_message: str | None       # status="failed" 時のエラー内容
```

### 6.3 選択方針（AIDE参考）

AIDEは「最高性能の有効解をベースに次の改善を行う」方針を取る。SERAではこれをLCBベースの優先度関数として一般化する。

```python
def compute_priority(node: SearchNode, exec_spec: ExecutionSpec) -> float:
    """
    AIDE参考の選択方針:
    - AIDEは最高性能の有効スクリプトを選択する（シンプルな greedy）
    - SERAはこれを LCB + コスト + 探索ボーナスで拡張する
    """
    if not node.feasible:
        return -float('inf')  # 制約違反は最低優先度
    if node.lcb is None:
        return float('inf')   # 未評価ノードは最優先（探索促進）

    lcb_primary = node.lcb
    cost = node.total_cost
    bonus = compute_exploration_bonus(node)

    return lcb_primary - exec_spec.search.lambda_cost * cost + exec_spec.search.beta_exploration * bonus

def compute_exploration_bonus(node: SearchNode) -> float:
    """未探索領域の優遇。最小実装は 0。"""
    # 拡張実装: 1.0 / sqrt(node.eval_runs + 1)（UCB1風）
    # または: 同一 branching_op の使用回数の逆数
    return 0.0
```

### 6.4 ノード選択アルゴリズム（必須）

```python
def select_next_node(open_list, all_nodes, exec_spec) -> tuple[SearchNode, str]:
    """
    AIDE参考: 状態に応じて適用するオペレータを自動決定する。

    選択アルゴリズム:
    1. open_list が空 → 終了
    2. 未評価ノード（status="pending"）があれば最優先で実行
    3. 失敗ノード（status="failed", debug_depth < max_debug_depth）があれば debug 対象として選択
    4. それ以外 → 最高 priority の評価済みノードを improve 対象として選択
    5. 評価済みノードが一定数に達し、全体の多様性が不足 → draft で新アプローチ導入

    Returns:
        (selected_node, operator_name)  # operator_name: "draft" | "debug" | "improve"
    """
    # Step 1: 未評価ノードの処理
    pending = [n for n in open_list if n.status == "pending"]
    if pending:
        return (max(pending, key=lambda n: n.priority or float('inf')), "evaluate")

    # Step 2: 失敗ノードの修復（AIDE Debugging）
    failed = [n for n in all_nodes.values()
              if n.status == "failed"
              and n.debug_depth < exec_spec.search.max_debug_depth
              and n.node_id not in closed_set]
    if failed:
        # 最も浅い（修復しやすい）失敗ノードを選択
        target = min(failed, key=lambda n: n.debug_depth)
        return (target, "debug")

    # Step 3: 多様性不足時の新規起草（AIDE Drafting）
    evaluated = [n for n in all_nodes.values() if n.status == "evaluated"]
    unique_methods = len({n.experiment_config.get("method", "") for n in evaluated})
    if unique_methods < exec_spec.search.min_diverse_methods and len(evaluated) >= exec_spec.search.draft_trigger_after:
        return (None, "draft")  # 親なしの新規ノード

    # Step 4: 最良ノードの改善（AIDE Improving）
    best = max(evaluated, key=lambda n: n.lcb or float('-inf'), default=None)
    if best:
        return (best, "improve")

    return (None, "draft")  # フォールバック
```

### 6.5 3オペレータ設計（AIDE参考：必須）

AIDEの Drafting / Debugging / Improving を SERA に適応した3オペレータを定義する。
§6.4 の `select_next_node()` がオペレータを自動選択し、以下の各オペレータが子ノードを生成する。

#### 6.5.1 Draft オペレータ（新規解の起草）

```text
目的: 既存の探索木に存在しない、まったく新しいアプローチを導入する

発動条件（§6.4 で自動判定）:
  - ルート初期化時（tree_ops.draft() の初回呼び出し）
  - 探索中に多様性が不足した場合（unique_methods < min_diverse_methods）
  - 全ノードが制約違反の場合（feasible なノードがゼロ）

特徴:
  - 親ノードなし（または仮想ルートが親）
  - experiment_config は先行研究・LLMの自由提案から構築
  - AIDEの「完全に新しいスクリプトを起草する」に対応

出力: 1〜n 個の新規 SearchNode（parent_id=None, depth=0, branching_op="draft"）
```

```python
def draft(self, specs: AllSpecs, agent_llm: AgentLLM, rng, n: int) -> list[SearchNode]:
    """
    ルート生成時:
      1. baseline_candidates → ベースライン再現ノード
      2. open_problems → 課題解決ノード
      3. LLM自由提案 → 新規アプローチノード
      配分: 各 n//3（端数は自由提案に割当）

    探索中の再 draft 時:
      1. 既存の全ノードの hypothesis 一覧を LLM に提示
      2. 「これまでと異なるアプローチ」を明示的に要求
      3. 1〜2 個のみ生成（大量生成は探索の焦点を散らす）
    """
    pass
```

#### 6.5.2 Debug オペレータ（エラー修復：AIDE最重要の差別化要素）

```text
目的: 実験が失敗（exit_code != 0）したノードのコードを修復し、有効な結果を得る

発動条件:
  - node.status == "failed" かつ node.debug_depth < max_debug_depth

特徴（AIDE準拠）:
  - 親ノード = 失敗したノード自身（修復チェーン）
  - experiment_config は変更しない（コードのみ修正）
  - debug_depth をインクリメント（深度制限で無限修復ループを防止）
  - エラーメッセージ（stderr）をLLMに提示して修正案を生成
  - 原子的修正: 1つのエラーに対して1つの修正のみ

深度制限（AIDE参考）:
  - max_debug_depth（既定3）を超えたら修復を諦め、そのノードは "failed" のまま閉じる
  - 理由: 根本的にアプローチが間違っている場合、修復より draft（新規起草）が効率的
```

```python
def debug(self, failed_node: SearchNode, agent_llm: AgentLLM,
          failure_extractor: "FailureKnowledgeExtractor | None" = None) -> SearchNode:
    """
    1. failed_node の experiment_code と error_message (stderr) を取得
    1a. failure_extractor が有効な場合、失敗パターンを分析し兄弟ノードへ知識注入（§26.4.3）
    2. LLM に以下のプロンプトを与え、修正コードを生成:

    プロンプト:
    ---
    以下の実験コードがエラーで失敗しました。エラーを修正してください。
    修正は最小限にし、実験の意図（仮説）は変更しないでください。

    ## 実験コード
    {failed_node.experiment_code}

    ## エラーメッセージ
    {failed_node.error_message}

    ## 修正方針
    - import エラー → 正しいモジュール名に修正
    - 型エラー → 型変換を追加
    - 形状エラー → テンソル形状を修正
    - ファイル不在 → パスを修正、またはデータ生成コードを追加
    - それ以外 → 最小限の修正で動作するようにする

    ## 出力: 修正後の完全なコード（差分ではなく全体）
    ---

    3. 修正コードで新ノードを構築:
       - parent_id = failed_node.node_id
       - experiment_config = failed_node.experiment_config（変更なし）
       - experiment_code = LLM生成の修正コード
       - hypothesis = failed_node.hypothesis（変更なし）
       - branching_op = "debug"
       - debug_depth = failed_node.debug_depth + 1
       - status = "pending"
    """
    pass
```

#### 6.5.3 Improve オペレータ（原子的改善：AIDE核心）

```text
目的: 動作する最良のノードに対して、単一の測定可能な変更を加えて改善する

発動条件:
  - 最高LCBの有効ノードが選択された場合（通常の探索ステップ）

特徴（AIDE準拠）:
  - 親ノード = 最良（または選択された）の有効ノード
  - 変更は「原子的」= 1つの操作変数のみ変更、または意味的に1つの変更
  - 変更の影響が測定可能であること（CI で比較できる大きさ）
  - AIDEの「each change's impact is directly measurable」に対応

原子的変更の定義:
  - experiment_config の diff が1キーのみ（推奨）
  - 2キー以上変更する場合は、rationale で「なぜ同時変更が必要か」を明示
  - branch_factor 個の子を生成する場合、各子は異なる変数を変更する（多様性確保）
```

```python
def improve(self, parent: SearchNode, specs: AllSpecs, agent_llm: AgentLLM,
            all_nodes: dict, rng, n_children: int) -> list[SearchNode]:
    """
    1. parent の結果と全兄弟の結果を収集
    2. LLM に「原子的改善」を要求するプロンプトを構築
    3. n_children 個の改善案を生成（各案は1変数のみ変更を推奨）
    4. validate_experiment_config でバリデーション
    5. 原子性チェック: diff が2キー以上なら警告ログ（棄却はしない）

    プロンプト:
    ---
    あなたは研究アシスタントです。以下の実験結果を踏まえ、
    {n_children}個の **原子的な** 改善案を提案してください。

    「原子的」とは: 1つの変数のみを変更し、その変更の効果を測定可能にすること。

    ## 研究目的
    {problem_spec.objective.description}

    ## 操作可能な変数（これ以外の変数は変更禁止）
    {manipulated_variables}

    ## 現在の最良実験（改善ベース）
    仮説: {parent.hypothesis}
    設定: {parent.experiment_config}
    結果: {parent.mu} ± {parent.se} (LCB: {parent.lcb})
    制約: {constraint_status}

    ## これまでの兄弟・子ノードの結果（CI付き）
    {sibling_summaries}

    ## 最良ノードとの差
    最良LCB: {best_node.lcb}

    ## 統計的ガイダンス
    - 各提案は1つの変数のみ変更せよ（原子的変更）
    - CI が重複する変更は無意味。SE の 2倍以上の変化を狙え
    - {n_children}個の提案はそれぞれ異なる変数を変更せよ

    ## 出力形式（JSON配列）
    [
      {
        "hypothesis": "learning_rate を 5e-4 に下げると過学習が抑制される",
        "experiment_config": {"learning_rate": 5e-4},  ← 1変数のみ変更
        "rationale": "現在の lr=1e-3 で train_loss < val_loss の乖離が大きい",
        "changed_variable": "learning_rate"  ← 変更した変数名を明示
      },
      ...
    ]
    ---
    """
    pass
```

### 6.6 変更可能変数の境界

分岐生成（improve/draft）で変更できるのは **ProblemSpec.manipulated_variables に定義された変数のみ** である。

```text
■ 変更可能（操作層）: ProblemSpec.manipulated_variables に列挙された変数
  例: 実験対象の learning_rate, batch_size, method, optimizer, data_split

■ 変更禁止（固定層）: ExecutionSpec に属する全パラメータ
  例: PPOのlr, clip_range, repeats, lcb_coef, lambda_cost, beta, rank, alpha

■ 変更禁止（固定層）: ModelSpec, ResourceSpec に属する全パラメータ
  例: base_model.id, adapter_spec.rank, executor_type, timeout

■ LoRA形状パラメータの扱い:
  - rank, alpha, target_modules は ModelSpec.adapter_spec に属する（固定層）
  - Phase 1 で確定後、全ノードで同一のLoRA形状を使用する
  - 形状が変わると delta inheritance の加減算が不可能になるため、変更は原理的に禁止
  - adapter_spec_hash が異なる分岐は生成してはならない
```

> **原則**: experiment_config に含めてよいキーは `ProblemSpec.manipulated_variables[].name` と完全一致するもののみ。それ以外のキーはバリデーションエラーとして棄却する。

#### 6.6.1 experiment_config のバリデーション

```python
def validate_experiment_config(config: dict, problem_spec: ProblemSpecModel) -> tuple[bool, list[str]]:
    """
    分岐生成の出力をホワイトリスト検証する。

    Returns:
        (is_valid, error_messages)
    """
    allowed_keys = {v.name for v in problem_spec.manipulated_variables}
    errors = []

    # 1. ホワイトリスト検証: 未知のキーを拒否
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        errors.append(f"Unknown keys (not in manipulated_variables): {unknown_keys}")

    # 2. 型・範囲検証: 各変数の制約を満たすか
    for var in problem_spec.manipulated_variables:
        if var.name not in config:
            continue  # 親からの暗黙継承（変更なし）
        value = config[var.name]
        if var.type == "float":
            if not (var.range[0] <= value <= var.range[1]):
                errors.append(f"{var.name}={value} out of range {var.range}")
        elif var.type == "int":
            if not (var.range[0] <= value <= var.range[1]):
                errors.append(f"{var.name}={value} out of range {var.range}")
        elif var.type == "categorical":
            if value not in var.choices:
                errors.append(f"{var.name}='{value}' not in choices {var.choices}")

    return (len(errors) == 0, errors)
```

### 6.7 LoRA系譜との接続

分岐生成は **外部探索木（仮説分岐）のみ** を扱う。LoRA系譜は以下のルールで **自動的に** 決定される：

```text
■ 分岐生成時（Phase 2）:
  - 子ノード.adapter_node_id = None（未割当）
  - 子ノードは親ノードの LoRA を暗黙的に継承する（参照のみ）

■ 実験実行時（Phase 3）:
  - 子ノードの実験は親の materialize 済み LoRA で AgentLLM を設定して実行

■ PPO更新時（Phase 5）:
  - PPO更新対象のノード群の「最良ノード」の親LoRAをベースに更新
  - 更新結果のΔを新しい adapter_node として lineage に登録
  - 当該ノード.adapter_node_id を新 adapter_node に設定

■ PPO未実行のノード:
  - adapter_node_id = 親の adapter_node_id のまま（差分Δなし = 同一アダプタ）

つまり:
  - 「LoRAを親から継承するか」 → 常に継承する（選択の余地なし）
  - 「新規分岐するか」 → PPO更新が走ったノードのみ自動的に新分岐
  - 「既存アダプタを再利用するか」 → PPO未実行ノードは親と同一アダプタ

分岐生成が「仮説 × 専門化方策」を決める必要はない。
仮説分岐は Phase 2、方策専門化は Phase 5 の責務であり、完全に分離される。
```

### 6.8 統計的コンテキストの提示

分岐生成（improve）自体は統計的判断を行わない（それは §6.3 の優先度関数と §10 の剪定の責務）。
ただし、LLMが統計的に有意義な提案を行えるよう、プロンプトに以下の統計的コンテキストを含める：

```text
■ 必須提示項目:
  - 親ノードの μ ± SE（95% CI）
  - 兄弟ノードの μ ± SE（95% CI）
  - 親と最良ノードの LCB 差（Δ_LCB）
  - 親ノードの制約充足状況

■ LLMへの指示（プロンプトに明記）:
  - 「CI が重複する兄弟との微小な変更は避け、統計的に区別可能な変更を提案せよ」
  - 「primary の改善幅が SE の 2 倍未満の変更は無意味である」

■ 統計的判断の責務分離:
  - LCB差の閾値判定 → 分岐生成の責務ではない。剪定（§10）と優先度関数（§6.3）が担う
  - 有意差検定 → 分岐生成は実施しない。プロンプトにCIを提示し、LLMの自律判断に委ねる
  - 逐次評価との関係 → 分岐生成は status="evaluated" のノードのみ参照する。
    eval_runs < repeats のノードも暫定 μ/SE を持つため参照対象に含む。
    ただし eval_runs=1 のノードは SE=inf であり、LLMへの提示時に
    「(暫定: n=1)」と注記する
```

#### 6.8.1 兄弟ノードコンテキストの構成規則

improve オペレータ（§6.5.3）がLLMに提示する「兄弟ノード情報」の範囲と粒度を定義する。

```text
■ 参照対象の範囲:
  1. 親ノードの直接の子（同一 parent_id）のうち status="evaluated" のもの
  2. 上記を LCB 降順でソートし、上位 sibling_context_k 件を選択
     - sibling_context_k = min(branch_factor * 2, 10)
     - 理由: LLMコンテキスト長の節約 + 情報過多による判断品質低下の防止
  3. 最良ノード（best_node）は parent_id に関わらず常に含める（参照点として）

■ 剪定済みノードの扱い:
  - status="pruned" のノードは参照対象に含めない
    理由: 剪定済みノードは探索的に不要と判断されたもの。
    コンテキスト枠の浪費が大きい
  - ただし、制約違反（feasible=false）で剪定されたノードは例外として
    最大2件まで含める（constraint_violation_examples）
    理由: 制約境界付近の情報は有用な負例として機能する

■ 失敗ノードの扱い:
  - status="failed" のノードは参照対象に含めない（debug オペレータが担当）
  - status="oom" / "timeout" のノードも含めない

■ 各ノードの提示粒度:
  提示項目（必須）:
    - hypothesis（1行要約）
    - experiment_config の親ノードとの diff（変更キーと値のみ）
    - μ ± SE（95% CI）
    - LCB
    - feasible フラグ
    - eval_runs 数（信頼度の指標として「n=3」等を付記）

  提示しない項目:
    - 実行ログ（stdout/stderr） → コンテキスト長の爆発を防止
    - experiment_code → improve は config レベルの変更であり、コード詳細は不要
    - metrics_raw → μ/SE に集約済み
    - rationale → 兄弟の分岐理由は改善提案に不要

■ draft オペレータの場合:
  兄弟ノードではなく「全評価済みノードの hypothesis 一覧」を提示する（§6.5.1）。
  提示粒度は hypothesis + method名 のみ（詳細不要、多様性確保が目的）。

■ debug オペレータの場合:
  兄弟ノードは参照しない。失敗ノード自身の experiment_code + error_message のみ（§6.5.2）。
```

#### 6.8.2 失敗知識コンテキスト（ECHO軽量版統合）

> **実装状況**: 実装済み。ソースコードは `src/sera/search/failure_extractor.py`, `src/sera/search/tree_ops.py`, `src/sera/search/search_manager.py` を参照。

失敗した兄弟ノードの知識を improve プロンプトに含め、同じ失敗の繰り返しを防ぐ。
これは ECHO軽量版（§26.4.3 FailureKnowledgeExtractor）で抽出された `FailureSummary` を利用する。

```text
■ 参照範囲:
  - 同一親ノードの子ノードのうち status="failed" のもの
  - FailureKnowledgeExtractor.extract() で生成された FailureSummary を参照
  - SearchNode.failure_context に格納済みの情報を使用（list[dict] 形式）

■ FailureSummary の構造（実装済み）:
  @dataclass
  class FailureSummary:
      node_id: str
      hypothesis: str
      error_category: str  # "runtime", "oom", "timeout", "logical", "unknown"
      error_message: str   # summary_max_tokens で切り詰め
      lesson: str          # 自動生成された教訓

■ エラーカテゴリ分類（ヒューリスティック、実装済み）:
  - status が "oom" → "oom"
  - status が "timeout" → "timeout"
  - メッセージに "memory", "cuda out of memory", "oom", "alloc" → "oom"
  - メッセージに "timeout", "timed out", "deadline" → "timeout"
  - メッセージに "runtime", "exception", "traceback", "error" → "runtime"
  - メッセージに "nan", "inf", "diverge", "negative loss" → "logical"
  - それ以外 → "unknown"

■ 提示粒度:
  各失敗ノードについて以下を提示:
    - error_category（カテゴリラベル）
    - hypothesis（何を試みたか）
    - lesson（FailureKnowledgeExtractor が生成した教訓）

  提示しない項目:
    - 完全な experiment_code（コンテキスト長の節約）
    - metrics_raw（失敗ノードには有意な metrics がない）

■ improve プロンプトへの注入方法（実装済み）:
  IMPROVE_PROMPT に {failure_context} プレースホルダを配置。
  TreeOps._build_failure_context(parent) が以下のフォーマットでテキスト化:

  Failed approaches to avoid:
  - [runtime] My hypothesis: Approach 'My hypothesis' raised a runtime error: ...
  - [oom] Another approach: Approach 'Another approach' caused OOM. ...

  failure_context が空の場合は空文字列が注入される。

■ failure_context の構築タイミング（実装済み）:
  - SearchManager の debug オペレータ実行後に FailureKnowledgeExtractor.extract() を実行
  - extract() の結果を同一親の兄弟ノードに inject() する
  - inject() は SearchNode.failure_context に FailureSummary.to_dict() を追加する
  - 重複防止: 同一 node_id の FailureSummary は追加しない
  - 上限制御: max_summaries_per_node（デフォルト3）を超えると追加をスキップ

■ 有効化条件:
  - plan_spec.echo.enabled == True の場合のみ FailureKnowledgeExtractor が初期化される
  - echo.enabled == False（デフォルト）の場合、failure_context は空のまま
```

### 6.9 失敗処理

```text
■ JSONパース失敗:
  - 最大3回リトライ（temperature += 0.1）
  - ```json ... ``` ブロック抽出の前処理
  - 3回失敗: 親ノードの config をそのままコピーした「同一実験」ノードを1つ生成
    （探索を完全停止させない安全策。ただし priority は低い）

■ バリデーション失敗（ホワイトリスト違反・範囲外）:
  - エラーメッセージをLLMにフィードバックして最大2回リトライ
  - リトライ後も全棄却: JSONパース失敗と同じフォールバック

■ 実験失敗（exit_code != 0）:
  - debug オペレータが自動発動（§6.5.2）
  - max_debug_depth（既定3）までリトライ
  - 超過したら諦めて "failed" のまま閉じる

■ 有効だが実験不能な設定（リソース超過）:
  - 事前検証: ResourceSpec.sandbox.experiment_memory_limit_gb を超えるメモリ要求は棄却
  - 実験実行時の判定: Phase 3 の Executor がOOMを検知（§7.5）→ status="oom"

■ OOM / timeout と debug オペレータの関係:
  - debug オペレータの対象は status="failed"（exit_code != 0 のロジックエラー）のみ
  - status="oom" / "timeout" は debug 対象外（コード修正では解決不能）
  - これらのノードは closed_set に入り、以後の探索・分岐生成から除外される
  - PPOバッファにも含めない（有意な metrics が存在しないため）

■ 制約違反ノードの学習対象としての扱い:
  - status="evaluated" かつ feasible=false のノードは PPO バッファに含める
  - 報酬計算（§9.2 compute_reward）で constraint_penalty が適用され、負の報酬信号となる
  - これにより PPO は制約違反領域を避けるように学習する
  - ただし、全ノードが制約違反の場合は PPO 更新をスキップする（§9.5）
```

### 6.10 探索と学習の境界

```text
■ 責務分離の原則:
  Phase 2（分岐生成）: 「何を探索するか」を決定する（3オペレータ: draft/debug/improve）
  Phase 5（PPO学習）: 「どれだけ上手に探索できるか」を改善する

■ PPO更新結果の反映タイミング:
  PPO更新によりLoRAが変わると、AgentLLMの生成品質が変化する。
  この変化は「次にAgentLLM.generate()が呼ばれたとき」に自然に反映される。
  つまり、PPO更新後に生成される子ノードは、更新前より質の高い仮説を生成する。

  具体的なタイミング:
  1. ステップN: ノードA を評価 → ppo_buffer に追加
  2. ステップN+k: ppo_buffer が ppo_trigger_interval に到達 → PPO更新実行
     → AgentLLM のLoRAが更新される
  3. ステップN+k+1: ノードB を展開 → AgentLLM.generate() で子ノード生成
     → この呼び出しは更新後のLoRAを使用する（自然に反映）

■ 分岐生成が学習に依存しない設計:
  分岐生成の入力は以下のみ:
  - parent の外部状態（hypothesis, experiment_config, metrics, error_message）
  - PlanSpec（固定）
  - ProblemSpec（固定）
  - RelatedWorkSpec（固定）

  学習結果（LoRA差分）は AgentLLM の内部状態として間接的にのみ影響する。
  分岐生成ロジック自体は学習の有無に関わらず同一のコードパスを通る。
```

### 6.11 探索メインループ（AIDE参考・疑似コード）
```python
def research_loop(specs, agent_llm):
    open_list = []                # SearchNode のリスト（priority でソート）
    closed_set = set()
    all_nodes = {}                # node_id → SearchNode
    best_node = None

    # Phase 2 初期化: ルートノード生成（draft オペレータ）
    root_children = tree_ops.draft(specs, agent_llm, rng, n=specs.exec.search.initial_root_children)
    for child in root_children:
        open_list.append(child)
        all_nodes[child.node_id] = child

    ppo_buffer = []
    step = 0

    while not should_terminate(step, best_node, specs):
        # Phase 2: select_next_node で次のノードとオペレータを自動決定（§6.4）
        selected, operator = select_next_node(open_list, all_nodes, specs.exec)

        if operator == "evaluate":
            # ── 未評価ノードの実行・評価 ──
            node = selected

            # Phase 3: 実験コード生成 + サンドボックス実行
            result = executor.run(node, specs, agent_llm)
            if result.exit_code != 0:
                node.status = "failed"
                node.error_message = result.stderr
                step += 1
                continue  # 次のループで debug が自動選択される

            # Phase 4: 統計評価（逐次）
            evaluate_node(node, specs)  # 初回は sequential_eval_initial 回
            if is_topk(node, all_nodes, k=specs.exec.evaluation.sequential_eval_topk):
                evaluate_node_full(node, specs)  # repeats まで追加
            node.status = "evaluated"
            node.priority = compute_priority(node, specs.exec)
            update_best(best_node, node)

        elif operator == "debug":
            # ── 失敗ノードの修復（§6.5.2） ──
            debug_child = tree_ops.debug(selected, agent_llm)
            open_list.append(debug_child)
            all_nodes[debug_child.node_id] = debug_child

        elif operator == "draft":
            # ── 新規アプローチの起草（§6.5.1） ──
            new_nodes = tree_ops.draft(specs, agent_llm, rng, n=2)
            for n in new_nodes:
                open_list.append(n)
                all_nodes[n.node_id] = n

        elif operator == "improve":
            # ── 最良ノードの原子的改善（§6.5.3） ──
            if selected.depth < specs.exec.search.max_depth:
                children = tree_ops.improve(
                    selected, specs, agent_llm, all_nodes, rng,
                    n_children=specs.exec.search.branch_factor
                )
                for child in children:
                    open_list.append(child)
                    all_nodes[child.node_id] = child

        # Phase 5: PPO 更新（評価済みバッファが溜まったら）
        evaluated_new = [n for n in all_nodes.values()
                         if n.status == "evaluated" and n.node_id not in closed_set]
        ppo_buffer.extend(evaluated_new)
        if len(ppo_buffer) >= specs.exec.learning.ppo_trigger_interval:
            ppo_trainer.update(ppo_buffer, agent_llm, specs)
            ppo_buffer.clear()

        # Phase 6: 系譜管理・剪定
        lineage_manager.maybe_squash(specs)
        pruned = pruner.prune(open_list, closed_set, specs)
        for p in pruned:
            closed_set.add(p.node_id)

        if selected and selected.node_id not in closed_set:
            if selected.status in ("evaluated", "failed"):
                closed_set.add(selected.node_id)
        step += 1

    return best_node
```

---
