# SERA 要件定義書 — 対話型セットアップウィザード

> 本ファイルは TASK.md v13.3 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 26. 対話型セットアップウィザード（Interactive Setup Wizard）

### 26.1 目的

現在の SERA は研究セットアップに3つの個別コマンドを順次実行する必要がある：

```text
sera init input1.yaml          # Input-1 YAML を手書き → workspace 初期化
sera phase0-related-work       # Phase 0: 先行研究収集
sera freeze-specs              # Phase 1: Spec確定・ExecutionSpec固定
```

この手順には以下の課題がある：

| 課題 | 詳細 |
|------|------|
| **YAML手書きの障壁** | Input-1 の YAML 構造を知っている必要がある。`direction`, `constraints[].type` 等の列挙値を間違えやすい |
| **Phase間の断絶** | 各コマンドが独立しており、Phase 0 の結果を見て Phase 1 のパラメータを調整する、といったフィードバックループが困難 |
| **レビュー機会の欠如** | Phase 0 で収集された論文や Phase 1 で生成された Spec を対話的にレビュー・修正する手段がない（`--auto` か手動 YAML 編集のみ） |
| **環境検出の不在** | GPU 有無、SLURM 環境等をユーザが手動で `ResourceSpec` に記述する必要がある |

**`sera setup`** コマンドは、これら3フェーズを **step-by-step の対話型ウィザード** に統合し、YAML の知識なしでも研究セットアップを完了できるようにする。

### 26.2 設計原則

| 原則 | 説明 |
|------|------|
| **ゼロ知識スタート** | YAML フォーマットや Spec 構造を知らなくても、自然言語の質問に答えるだけで Input-1 を構築できる |
| **段階的詳細化** | 必須項目のみ対話で収集し、オプション項目は既定値を提示してスキップ可能にする |
| **戻り可能** | `back` コマンドで前のステップに戻れる。`goto N` で任意ステップにジャンプ可能 |
| **中断再開** | `Ctrl+C` または `quit` で中断しても `.wizard_state.json` に状態保存。`--resume` で再開 |
| **既存コード再利用** | `init_cmd.run_init()`, `phase0_cmd.run_phase0()`, `phase1_cmd.run_freeze_specs()`, `SpecBuilder`, `SpecFreezer` を内部で呼び出す。ロジックの重複を避ける |
| **i18n 対応** | `--lang` オプションで `ja`/`en` を切り替え可能（デフォルト: `ja`）。プロンプト文字列を辞書化し `MESSAGES["ja"]` / `MESSAGES["en"]` で管理 |

### 26.3 CLI インターフェース

#### 26.3.1 コマンド定義

```python
@app.command()
def setup(
    work_dir: Annotated[str, typer.Option("--work-dir", "-w")] = "./sera_workspace",
    resume: Annotated[bool, typer.Option("--resume", "-r")] = False,
    from_input1: Annotated[Optional[str], typer.Option("--from-input1")] = None,
    skip_phase0: Annotated[bool, typer.Option("--skip-phase0")] = False,
    lang: Annotated[str, typer.Option("--lang", "-l")] = "ja",
):
    """対話型セットアップウィザード。Input-1構築→Phase 0→Phase 1をガイド付きで実行"""
    from sera.commands.setup_cmd import run_setup
    run_setup(work_dir, resume, from_input1, skip_phase0, lang)
```

#### 26.3.2 オプション

| オプション | 型 | 既定値 | 説明 |
|-----------|-----|--------|------|
| `--work-dir`, `-w` | `str` | `"./sera_workspace"` | ワークスペースディレクトリ |
| `--resume`, `-r` | `bool` | `False` | `.wizard_state.json` から中断箇所を再開 |
| `--from-input1` | `str?` | `None` | 既存の Input-1 YAML パスを指定。Phase A (Step 1-7) をスキップ |
| `--skip-phase0` | `bool` | `False` | Phase 0 をスキップ（既に `related_work_spec.yaml` がある場合） |
| `--lang`, `-l` | `str` | `"ja"` | UI言語。`"ja"` または `"en"` |

#### 26.3.3 使用例

```bash
# フルウィザード（デフォルト）
sera setup

# 英語UIで実行
sera setup --lang en

# 既存 Input-1 から開始（Phase A スキップ）
sera setup --from-input1 my_research.yaml

# 中断した箇所から再開
sera setup --resume

# Phase 0 スキップ（先行研究を手動で用意済み）
sera setup --skip-phase0

# カスタムワークスペース
sera setup --work-dir ./my_project
```

### 26.4 ウィザード全体フロー

```text
┌─────────────────────────────────────────────────────────┐
│                    sera setup                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Phase A: Input-1 構築 (Step 1-7)                       │
│  ┌─────────────────────────────────────────────┐        │
│  │ Step 1: data.*       (データ情報)            │        │
│  │ Step 2: domain.*     (分野情報)              │        │
│  │ Step 3: task.*       (タスク情報)            │        │
│  │ Step 4: goal.*       (目標情報)              │        │
│  │ Step 5: constraints[] (制約条件)             │        │
│  │ Step 6: notes        (自由記述)              │        │
│  │ Step 7: Input-1 プレビュー + 確認            │        │
│  └─────────────────────────────────────────────┘        │
│           │                                             │
│           ▼                                             │
│  Phase B: Phase 0 実行 + レビュー (Step 8-9)            │
│  ┌─────────────────────────────────────────────┐        │
│  │ Step 8: Phase 0 パラメータ確認 + 実行        │        │
│  │ Step 9: 収集結果レビュー + 除外/追加         │        │
│  └─────────────────────────────────────────────┘        │
│           │                                             │
│           ▼                                             │
│  Phase C: Phase 1 Spec確定 (Step 10-11)                 │
│  ┌─────────────────────────────────────────────┐        │
│  │ Step 10: 各Spec LLM生成 + レビュー + 編集   │        │
│  │ Step 11: 全Spec サマリ + freeze + ロック     │        │
│  └─────────────────────────────────────────────┘        │
│           │                                             │
│           ▼                                             │
│  ✓ セットアップ完了                                     │
│    → sera research で探索開始可能                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**スキップ条件**：

| オプション | 効果 |
|-----------|------|
| `--from-input1 path` | Step 1-7 をスキップし、指定 YAML を読み込んで Step 8 から開始 |
| `--skip-phase0` | Step 8-9 をスキップし、既存の `related_work_spec.yaml` を読み込んで Step 10 から開始 |
| `--resume` | `.wizard_state.json` の `current_step` から再開 |

### 26.5 各ステップ詳細

#### 26.5.1 Step 1: データ情報（`data.*`）

**収集フィールド**：

| フィールド | プロンプト (ja) | プロンプト (en) | バリデーション | 既定値 |
|-----------|----------------|----------------|--------------|--------|
| `data.description` | 「データの説明を入力してください（例: "画像分類用のCIFAR-10データセット"）」 | "Describe your data (e.g., 'CIFAR-10 image classification dataset')" | 必須、1文字以上 | — |
| `data.location` | 「データの場所を入力してください（パス/URI/リポジトリURL）」 | "Enter data location (path/URI/repository URL)" | 必須、1文字以上 | — |
| `data.format` | 「データ形式を選択してください」 | "Select data format" | 選択式: `csv`, `json`, `parquet`, `code`, `pdf`, `mixed` | `csv` |
| `data.size_hint` | 「データサイズの目安を選択してください」 | "Select approximate data size" | 選択式: `small(<1GB)`, `medium(1-100GB)`, `large(>100GB)` | `small(<1GB)` |

**UIモックアップ**：

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 1/11] データ情報
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  データの説明を入力してください
  例: "画像分類用のCIFAR-10データセット"
  > 姫野ベンチマーク（HPC性能評価用ポアソン方程式ソルバー）

  データの場所を入力してください（パス/URI/リポジトリURL）
  > /home/user/himeno_benchmark/

  データ形式を選択してください:
    [1] csv    [2] json    [3] parquet
    [4] code   [5] pdf     [6] mixed
  > 4

  データサイズの目安を選択してください:
    [1] small(<1GB)    [2] medium(1-100GB)    [3] large(>100GB)
  > 1

  ✓ Step 1 完了
  (back: 戻る / help: ヘルプ / quit: 中断保存)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 26.5.2 Step 2: 分野情報（`domain.*`）

**収集フィールド**：

| フィールド | プロンプト (ja) | バリデーション | 既定値 |
|-----------|----------------|--------------|--------|
| `domain.field` | 「研究分野を入力してください（例: HPC, NLP, CV, materials）」 | 必須 | — |
| `domain.subfield` | 「より具体的な分野を入力してください（例: compiler optimization）」 | 任意（空可） | `""` |

#### 26.5.3 Step 3: タスク情報（`task.*`）

**収集フィールド**：

| フィールド | プロンプト (ja) | バリデーション | 既定値 |
|-----------|----------------|--------------|--------|
| `task.brief` | 「研究タスクを1〜3文で説明してください」 | 必須、10文字以上推奨 | — |
| `task.type` | 「タスクの種類を選択してください」 | 選択式: `optimization`, `prediction`, `generation`, `analysis`, `comparison` | `optimization` |

#### 26.5.4 Step 4: 目標情報（`goal.*`）

**収集フィールド**：

| フィールド | プロンプト (ja) | バリデーション | 既定値 |
|-----------|----------------|--------------|--------|
| `goal.objective` | 「研究の目標を記述してください（例: "実行時間を最小化"）」 | 必須 | — |
| `goal.direction` | 「目標の方向を選択してください」 | 選択式: `minimize`, `maximize` | 自動推定 |
| `goal.baseline` | 「既知のベースライン値があれば入力してください」 | 任意（空可） | `""` |

**direction 自動推定ロジック**：

```python
def estimate_direction(objective: str) -> str:
    """goal.objective の文字列から direction を推定する。"""
    minimize_keywords = [
        "最小", "minimize", "reduce", "lower", "decrease", "短縮",
        "削減", "抑制", "loss", "error", "latency", "runtime", "time",
    ]
    maximize_keywords = [
        "最大", "maximize", "increase", "improve", "higher", "向上",
        "精度", "accuracy", "score", "throughput", "performance",
    ]
    obj_lower = objective.lower()
    min_score = sum(1 for kw in minimize_keywords if kw in obj_lower)
    max_score = sum(1 for kw in maximize_keywords if kw in obj_lower)

    if min_score > max_score:
        return "minimize"
    elif max_score > min_score:
        return "maximize"
    else:
        return None  # 推定不能 → ユーザに明示的に選択させる
```

推定結果が得られた場合、確認プロンプトを表示する：

```text
  目標「実行時間を最小化」から direction = "minimize" と推定しました。
  正しいですか？ [Y/n] >
```

#### 26.5.5 Step 5: 制約条件（`constraints[]`）

**繰り返し入力**（0個以上）：

```text
  制約条件を追加しますか？ [y/N] > y

  制約条件 #1:
    名前: > format_valid
    種類: [1] ge (>=)  [2] le (<=)  [3] eq (==)  [4] bool (true/false)
    > 4
    閾値: > true

  制約条件をさらに追加しますか？ [y/N] > n
```

**バリデーション**：
- `name`: 必須、英数字+アンダースコア
- `type`: 選択式 `ge`, `le`, `eq`, `bool`
- `threshold`: `bool` 型なら `true`/`false`、`ge`/`le`/`eq` なら数値

#### 26.5.6 Step 6: 自由記述（`notes`）

```text
  追加のメモがあれば入力してください（空でスキップ可）:
  > 特になし
```

**バリデーション**: 任意（空文字可）

#### 26.5.7 Step 7: Input-1 プレビュー + 確認

Step 1-6 で収集した全データを YAML 形式でプレビュー表示し、確認を求める。

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 7/11] 入力確認
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  以下の Input-1 YAML を生成しました：

  ┌──────────────────────────────────────────────┐
  │ version: 1                                   │
  │ data:                                        │
  │   description: "姫野ベンチマーク..."          │
  │   location: "/home/user/himeno_benchmark/"   │
  │   format: "code"                             │
  │   size_hint: "small(<1GB)"                   │
  │ domain:                                      │
  │   field: "HPC"                               │
  │   subfield: "stencil computation"            │
  │ task:                                        │
  │   brief: "ポアソン方程式ソルバーの..."        │
  │   type: "optimization"                       │
  │ goal:                                        │
  │   objective: "実行時間を最小化"               │
  │   direction: "minimize"                      │
  │   baseline: "12.5 sec"                       │
  │ constraints: []                              │
  │ notes: ""                                    │
  └──────────────────────────────────────────────┘

  この内容で確定しますか？
    [1] 確定して次へ進む
    [2] ステップに戻って修正する (goto N)
    [3] YAML を直接編集する
  > 1

  ✓ Input-1 を specs/input1.yaml に保存しました
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**「YAML を直接編集する」選択時**：`$EDITOR`（未設定なら `vi`）で一時ファイルを開く。保存後に `Input1Model` でバリデーションし、失敗なら再編集を促す。

**確定時の処理**：
1. `Input1Model(**collected_data)` で Pydantic バリデーション
2. `init_cmd.run_init()` と同等の処理でワークスペース初期化
3. `specs/input1.yaml` に保存

#### 26.5.8 Step 8: Phase 0 パラメータ確認 + 実行

**8a: APIキーチェック**

Phase 0 の実行に必要な API キーの存在を確認し、不足があれば警告する。

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 8/11] Phase 0 実行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  APIキーチェック:
    ✓ SEMANTIC_SCHOLAR_API_KEY  (設定済み)
    ✗ SERPAPI_API_KEY           (未設定 — Web検索フォールバック無効)
    ✓ CROSSREF_EMAIL            (設定済み)

  ※ 未設定のキーがあっても Phase 0 は実行可能です。
    設定済みAPI のみ使用します。
```

**8b: パラメータ確認**

```text
  Phase 0 パラメータ:
    ┌──────────────────────────────────────┐
    │ top_k_papers:         10             │
    │ recent_years_bias:    5              │
    │ citation_graph_depth: 1              │
    │ teacher_papers:       5              │
    └──────────────────────────────────────┘

  このパラメータで実行しますか？
    [1] このまま実行
    [2] パラメータを変更する
  > 1
```

パラメータ変更選択時は、個別フィールドの入力を受け付ける。バリデーション：`top_k_papers` ≥ 1, `recent_years_bias` ≥ 1, `citation_graph_depth` ≥ 0, `teacher_papers` ≥ 1。

**8c: Phase 0 実行（進捗表示）**

```text
  Phase 0 実行中...

  [████████░░░░░░░░░░░░] 40%
  ├─ Semantic Scholar 検索: 3/5 クエリ完了 (18 論文取得)
  ├─ CrossRef 検索: 待機中
  └─ arXiv 検索: 待機中

  経過時間: 00:01:23
```

実行には `phase0_cmd.run_phase0()` の内部ロジックを呼び出す。進捗は `Rich.Progress` で表示する。

**エラー時の動作**：
- 特定 API の失敗 → 警告表示して続行（他 API で補完）
- 全 API 失敗 → エラー表示、リトライ or スキップを選択可能

#### 26.5.9 Step 9: 収集結果レビュー

Phase 0 の結果（論文リスト、クラスタ、教師論文候補）をテーブル形式で表示し、ユーザが除外/追加操作を行える。

**9a: 収集論文一覧**

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 9/11] 結果レビュー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  収集論文 (10件):
  ┌────┬──────────────────────────────────┬──────┬───────┬───────────┐
  │ #  │ タイトル                          │ 年   │ 被引用│ 関連度    │
  ├────┼──────────────────────────────────┼──────┼───────┼───────────┤
  │ 1  │ AI-Scientist: Automated Resea... │ 2024 │  150  │ 0.92      │
  │ 2  │ MCTS for LLM Reasoning...        │ 2024 │   85  │ 0.88      │
  │ ...│ ...                              │ ...  │  ...  │ ...       │
  └────┴──────────────────────────────────┴──────┴───────┴───────────┘

  操作:
    [1] この結果で確定
    [2] 論文を除外する (番号指定: 例 "3,5,7")
    [3] 論文を手動追加する (DOI/arXiv ID/タイトル)
    [4] 詳細を表示する (番号指定)
  >
```

**9b: クラスタ表示**

```text
  論文クラスタ:
  ┌──────────────────────┬────────────────────────────┬───────┐
  │ クラスタ名            │ 説明                       │ 論文数│
  ├──────────────────────┼────────────────────────────┼───────┤
  │ tree_search_methods  │ 木構造探索による研究自動化  │   4   │
  │ lora_adaptation      │ LoRAベースのモデル適応      │   3   │
  │ automated_science    │ 自動科学発見               │   3   │
  └──────────────────────┴────────────────────────────┴───────┘
```

**9c: 教師論文候補**

```text
  教師論文候補 (5件):
  ┌────┬──────────────────────────────────┬────────────────────┐
  │ #  │ タイトル                          │ 役割               │
  ├────┼──────────────────────────────────┼────────────────────┤
  │ 1  │ AI-Scientist: Automated Resea... │ structure_reference│
  │ ...│ ...                              │ ...                │
  └────┴──────────────────────────────────┴────────────────────┘

  この教師論文セットで確定しますか？ [Y/n] >
```

**除外操作**：番号をカンマ区切りで入力（例: `3,5`）。除外後にテーブルを再表示。
**追加操作**：DOI、arXiv ID、またはタイトルを入力。Semantic Scholar API で検索し、結果を表示して確認後追加。

#### 26.5.10 Step 10: Spec 生成 + レビュー + 編集

Phase 0 出力 + Input-1 を基に、LLM で各 Spec を順次生成し、ユーザにレビューを求める。

**対象 Spec と生成順序**：

| 順序 | Spec | 生成方法 | レビュー |
|------|------|---------|---------|
| 1 | `ProblemSpec` | LLM生成（`SpecBuilder.build_problem_spec()`） | フィールド単位で編集可能 |
| 2 | `ModelSpec` | テンプレート + 環境検出 | provider/model_id の選択（**既定: local**） |
| 3 | `ResourceSpec` | 環境自動検出 + テンプレート | executor_type/GPU設定/MCP サーバーの確認 |
| 4 | `PlanSpec` | LLM生成 + テンプレート | reward/agent_commands/MCP/関数の確認 |
| 5 | `ExecutionSpec` | テンプレート + 既定値 | 主要パラメータの確認（**既定: learning.enabled=true**） |
| 6 | `LanguageConfig` | プリセット + 対話入力 | 言語選択 + コンパイル設定 + multi_file（**既定: True**） |
| 7 | `DependencyConfig` | 対話入力（オプション） | 依存管理ツール + LLM生成ビルドファイル |
| 8 | `ContainerConfig` | 対話入力（executor=slurm時のみ） | コンテナランタイム + イメージ + バインドマウント |

> `PaperSpec`, `PaperScoreSpec`, `TeacherPaperSet`, `RelatedWorkSpec` は Phase 0/1 で自動確定済み。

**デフォルト設計原則**：

| パラメータ | デフォルト値 | 根拠 |
|-----------|-------------|------|
| `agent_llm.provider` | `"local"` | LoRA学習にはローカルモデルが必須 |
| `learning.enabled` | `true` | PPO + LoRA差分継承がSERAの中核機能 |
| `language.multi_file` | `true` | 複数ファイルプロジェクトがデフォルト（§7.2.1） |
| `use_streaming` | `true` | ストリーミング実行がデフォルト（§7.5） |
| `agent_commands.tools.enabled` | `true` | AgentLoop によるツール実行を標準動作とする |
| `agent_commands.functions.available_functions` | 全19関数 | 全関数をデフォルトで実行可能にする |
| `reward.method` | `"hiper"` | HiPER 3層階層PPOが最も高性能 |
| `echo.enabled` | `true` | 失敗知識の再利用は探索効率に直結 |

**10a: ProblemSpec レビュー**

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 10/11] Spec確定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1/5] ProblemSpec を生成中...  ✓

  ┌──────────────────────────────────────────────┐
  │ problem_spec:                                │
  │   title: "Stencil Computation Optimization   │
  │           via Automated Code Generation"     │
  │   objective:                                 │
  │     description: "Minimize execution time..."│
  │     metric_name: "wall_time_sec"             │
  │     direction: "minimize"                    │
  │   manipulated_variables:                     │
  │     - name: "block_size"                     │
  │       type: "int"                            │
  │       range: [16, 512]                       │
  │       scale: "log"                           │
  │     - name: "optimization_level"             │
  │       type: "categorical"                    │
  │       choices: ["O1", "O2", "O3", "Ofast"]  │
  │   ...                                        │
  └──────────────────────────────────────────────┘

  操作:
    [1] この内容で確定
    [2] フィールドを編集する
    [3] LLM で再生成する
    [4] YAML を直接編集する
  >
```

**フィールド編集（操作 2）**：

```text
  編集するフィールドを選択:
    [1] title
    [2] objective.description
    [3] objective.metric_name
    [4] objective.direction
    [5] manipulated_variables
    [6] constraints
  > 3
  新しい値 (現在: "wall_time_sec"): > execution_time_sec
  ✓ 更新しました
```

**10b: ModelSpec — 環境検出**

```python
def detect_environment() -> dict:
    """実行環境を自動検出する。"""
    env = {}
    # GPU検出
    try:
        import torch
        env["gpu_available"] = torch.cuda.is_available()
        if env["gpu_available"]:
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_type"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except ImportError:
        env["gpu_available"] = False

    # SLURM検出
    env["slurm_available"] = shutil.which("sbatch") is not None
    if env["slurm_available"]:
        env["slurm_partition"] = os.environ.get("SLURM_PARTITION", "")
        env["slurm_account"] = os.environ.get("SLURM_ACCOUNT", "")

    # メモリ検出
    import psutil
    env["system_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)

    # CPU検出
    env["cpu_cores"] = os.cpu_count()

    return env
```

検出結果の表示例：

```text
  [2/5] ModelSpec — 環境自動検出結果:
  ┌──────────────────────────────────────────────┐
  │ GPU:    NVIDIA A100 80GB x 2                 │
  │ SLURM:  利用可能 (partition: gpu)            │
  │ Memory: 256 GB                               │
  │ CPU:    64 cores                             │
  └──────────────────────────────────────────────┘

  LLM プロバイダを選択してください:
    [1] local  — ローカルモデル (GPU必須、LoRA学習対応) ← 推奨
    [2] openai — OpenAI API (GPT-4o等、LoRA学習不可)
    [3] anthropic — Anthropic API (Claude等、LoRA学習不可)
  > 1

  ベースモデルID [Qwen/Qwen2.5-Coder-7B-Instruct]: >
    ✓ デフォルト: Qwen/Qwen2.5-Coder-7B-Instruct
  LoRA rank [16]: >
  LoRA alpha [32]: >

  ⚠ provider=local の場合、learning.enabled=true が推奨されます。
    openai/anthropic を選択した場合、LoRA学習は自動的に無効化されます。
```

**10c: ResourceSpec — 環境検出の反映**

検出された環境情報を `ResourceSpec` の初期値に反映する：

```python
def build_resource_defaults(env: dict) -> dict:
    """環境検出結果から ResourceSpec の既定値を構築する。"""
    defaults = {
        "compute": {
            "executor_type": "slurm" if env.get("slurm_available") else "local",
            "gpu_required": env.get("gpu_available", False),
            "gpu_type": env.get("gpu_type", ""),
            "gpu_count": min(env.get("gpu_count", 0), 1),  # デフォルトは1GPU
            "cpu_cores": min(env.get("cpu_cores", 4), 8),
            "memory_gb": min(env.get("system_memory_gb", 16), 32),
        },
    }
    if env.get("slurm_available"):
        defaults["compute"]["slurm"] = {
            "partition": env.get("slurm_partition", "gpu"),
            "account": env.get("slurm_account", ""),
        }
    return defaults
```

**10c-slurm: ResourceSpec — 分散SLURMノード設定**

SLURM環境が検出された場合（`env.slurm_available == True`）、分散SLURMアーキテクチャ（§23.9）の設定を対話的に収集する。ログインノードにGPUがない環境では分散モードが推奨される。

```text
  [3/5] ResourceSpec — SLURM設定:

  SLURM環境を検出しました:
    パーティション: gpu, gpu-large, cpu
    デフォルトアカウント: my_project

  分散SLURMモードを使用しますか？
  （すべてのGPU処理を計算ノードで実行します）
    [1] はい（推奨: ログインノードにGPUがない環境）
    [2] いいえ（ヘッドノードでvLLM/PPOを実行）
  > 1

  ── vLLM推論ノード設定 ──
  パーティション [gpu-large]: >
  GPU数 [1]: > 4
  GPU種別 [A100]: >
  メモリ (GB) [256]: >
  時間制限 [12:00:00]: >
  ノードリスト (空=自動割り当て): >

  ── 実験実行ノード設定 ──
  パーティション [gpu]: >
  GPU数 [1]: >
  排他モード (ベンチマーク用) [N]: > y

  ── PPO学習ノード設定 ──
  パーティション [gpu-large]: >
  GPU数 [2]: >
  時間制限 [04:00:00]: >

  vLLMサーバーモード:
    [1] persistent — 研究中ずっと起動（推奨）
    [2] transient  — バッチごとに起動・停止
  > 1

  ┌──────────────────────────────────────────────────────┐
  │ distributed:                                          │
  │   enabled: true                                       │
  │   inference: {partition: gpu-large, gpu: 4×A100}      │
  │   experiment: {partition: gpu, exclusive: true}        │
  │   training: {partition: gpu-large, gpu: 2×A100}       │
  │   vllm_mode: persistent                               │
  └──────────────────────────────────────────────────────┘
  この設定でよろしいですか？ [Y/n/edit] >
```

分散モードが有効化された場合、`wizard_state["phase1_params"]` に以下のキーが追加される：

| キー | 型 | 説明 |
|------|-----|------|
| `distributed_enabled` | `bool` | 分散モード有効化フラグ |
| `distributed_inference_partition` | `str` | vLLM推論ノードのパーティション |
| `distributed_inference_gpu_count` | `int` | vLLM推論ノードのGPU数 |
| `distributed_inference_gpu_type` | `str` | vLLM推論ノードのGPU種別 |
| `distributed_inference_memory_gb` | `int` | vLLM推論ノードのメモリ |
| `distributed_inference_time_limit` | `str` | vLLM推論ノードの時間制限 |
| `distributed_inference_nodelist` | `str` | vLLM推論ノードのノードリスト |
| `distributed_experiment_partition` | `str` | 実験ノードのパーティション |
| `distributed_experiment_exclusive` | `bool` | 実験ノードの排他モード |
| `distributed_training_partition` | `str` | PPO学習ノードのパーティション |
| `distributed_training_gpu_count` | `int` | PPO学習ノードのGPU数 |
| `distributed_training_time_limit` | `str` | PPO学習ノードの時間制限 |
| `distributed_vllm_mode` | `str` | vLLMサーバーライフサイクル |

これらは `step11_freeze.py` で `cli_args` に展開され、`phase1_cmd.run_freeze_specs()` → `SpecBuilder` → `resource_spec.yaml` の `compute.distributed` セクションに反映される。

分散モードが無効（`[2] いいえ` 選択）の場合、これらのキーは追加されず、既存の §23.1 アーキテクチャが適用される。

**10c-2: ResourceSpec — MCP サーバー設定**

MCP（Model Context Protocol）サーバーを登録することで、外部ツールをエージェントに追加できる。

```text
  [3/5] ResourceSpec — MCP サーバー設定:

  MCP サーバーを追加しますか？ [y/N] > y

  MCP サーバー #1:
    名前: > local_jupyter
    接続URL: > http://localhost:8080
    APIキー環境変数名 (不要なら空): >
    タイムアウト (秒) [30]: >
    許可ツール (カンマ区切り、全許可なら空): > execute_notebook,read_dataframe

  MCP サーバーをさらに追加しますか？ [y/N] > n

  登録済み MCP サーバー:
  ┌──────────────────┬──────────────────────────┬──────────────────────────┐
  │ 名前             │ URL                       │ 許可ツール               │
  ├──────────────────┼──────────────────────────┼──────────────────────────┤
  │ local_jupyter    │ http://localhost:8080      │ execute_notebook, ...    │
  └──────────────────┴──────────────────────────┴──────────────────────────┘
```

MCP 設定は `ResourceSpec.mcp.servers` に保存される（§28.9.3 参照）。登録した MCP ツールは `PlanSpec.agent_commands.tools.available_tools` にも自動追加される。

**10d: PlanSpec — エージェント設定（reward / agent_commands）**

PlanSpec はLLM生成後、agent_commands セクションをユーザがレビュー・編集する。

```text
  [4/5] PlanSpec — エージェント設定:

  報酬計算方式を選択してください:
    [1] hiper     — HiPER 3層階層PPO (推奨、最高精度)
    [2] mt_grpo   — MT-GRPO ターンレベル報酬
    [3] tool_aware — ツール効率ボーナス付き MT-GRPO
    [4] outcome_rm — 結果ベース報酬のみ（シンプル）
  > 1

  ECHO（失敗知識再利用）を有効にしますか？ [Y/n] > Y

  ┌──────────────────────────────────────────────┐
  │ agent_commands:                              │
  │   tools:                                     │
  │     enabled:           true                  │
  │     rate_limit:        30/min                │
  │   functions:                                 │
  │     available:         全19関数 (有効)        │
  │     tool_bindings:     10関数にツール割当済み │
  │   loop_defaults:                             │
  │     max_steps:         10                    │
  │     tool_call_budget:  20                    │
  │     timeout_sec:       300                   │
  │   mcp_tools:           (MCP登録ツール数)      │
  └──────────────────────────────────────────────┘

  操作:
    [1] この設定で確定
    [2] 利用可能関数を編集する
    [3] ツールバインディングを編集する
    [4] ループパラメータを編集する
    [5] 全フィールドを表示する
  >
```

**関数編集（操作 2）**：

```text
  利用可能関数一覧（✓ = 有効、✗ = 無効）:

  [search]
    ✓ search_draft          ✓ search_debug         ✓ search_improve
  [execution]
    ✓ experiment_code_gen
  [spec]
    ✓ spec_generation_problem  ✓ spec_generation_plan
  [paper]
    ✓ paper_outline   ✓ paper_draft       ✓ paper_reflection
    ✓ aggregate_plot_generation  ✓ aggregate_plot_fix
    ✓ citation_identify  ✓ citation_select   ✓ citation_bibtex
  [evaluation]
    ✓ paper_review    ✓ paper_review_reflection  ✓ meta_review
  [phase0]
    ✓ query_generation  ✓ paper_clustering

  無効にする関数番号（カンマ区切り、なしならEnter）: >
```

> **デフォルト**: 全19関数が有効。ユーザーは必要に応じて関数を無効化できる。`function_tool_bindings` に定義された10関数は AgentLoop 経由で実行され、残り9関数は SINGLE_SHOT（1回の LLM 呼び出し）で実行される。

**10e: ExecutionSpec — 主要パラメータ確認**

全フィールドを表示するのではなく、影響の大きい主要パラメータのみ確認する：

```text
  [5/8] ExecutionSpec — 主要パラメータ (learning.enabled=true がデフォルト):
  ┌──────────────────────────────────────────────┐
  │ search.max_nodes:           100              │
  │ search.max_depth:           10               │
  │ search.branch_factor:       3                │
  │ evaluation.repeats:         3                │
  │ evaluation.lcb_coef:        1.96             │
  │ learning.enabled:           true             │
  │ learning.lr:                1e-4             │
  │ termination.plateau_patience: 10             │
  │ termination.max_wall_time_hours: null (無制限)│
  └──────────────────────────────────────────────┘

  この設定で確定しますか？
    [1] 確定
    [2] パラメータを変更する
    [3] 全フィールドを表示する
  >
```

**10f: LanguageConfig — 言語設定 + コンパイル型サポート + 複数ファイル**

```text
  [6/8] LanguageConfig — 言語設定:

  実験言語を選択してください:
    [1] python (Recommended)
    [2] r
    [3] julia
    [4] cpp
    [5] rust
    [6] go
  > 1

  複数ファイルプロジェクトを許可しますか？ [Y/n] > Y

  LanguageConfig Summary:
    language:    python
    compiled:    False
    multi_file:  True
```

コンパイル型言語（cpp, rust, go）選択時は追加で `compile_command`, `compile_flags`, `link_flags`, `binary_name`, `build_timeout_sec` を対話的に設定する。

**10g: DependencyConfig — 依存関係管理**

```text
  [7/8] DependencyConfig — 依存関係管理:

  依存関係管理を設定しますか？ [y/N] > y

  依存管理ツール:
    [1] pip (Recommended)
    [2] conda
    [3] cargo
    [4] cmake
    [5] go_mod
  > 1

  ビルドファイル名: > requirements.txt
  LLMにビルドファイルを生成させますか？ [Y/n] > Y
  インストールタイムアウト（秒）: > 300

  DependencyConfig Summary:
    manager:            pip
    build_file:         requirements.txt
    llm_generated_build: True
```

**10h: ContainerConfig — SLURM+コンテナ統合**（executor=slurm 時のみ表示）

```text
  [8/8] ContainerConfig — SLURM+コンテナ統合:

  SLURMでコンテナを使用しますか？ [y/N] > y

  コンテナランタイム:
    [1] singularity (Recommended)
    [2] apptainer
    [3] docker
  > 1

  コンテナイメージ（URIまたは.sifパス）: > /shared/images/sera_pytorch.sif
  GPU パススルーを有効にしますか？ [Y/n] > Y
  バインドマウント（カンマ区切り）: > /data:/data:ro,/scratch:/scratch

  ContainerConfig Summary:
    runtime:     singularity
    image:       /shared/images/sera_pytorch.sif
    GPU:         True
    bind_mounts: ['/data:/data:ro', '/scratch:/scratch']
```

#### 26.5.11 Step 11: 全 Spec サマリ + Freeze + ロック

**11a: 全 Spec サマリ表示**

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERA セットアップウィザード  [Step 11/11] 最終確認
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  生成された Spec 一覧:
  ┌──────────────────────┬─────────────────────────────────┐
  │ Spec                 │ ステータス                       │
  ├──────────────────────┼─────────────────────────────────┤
  │ input1.yaml          │ ✓ 確定済み                      │
  │ related_work_spec    │ ✓ Phase 0 で生成                │
  │ paper_spec           │ ✓ Phase 0 で生成                │
  │ paper_score_spec     │ ✓ Phase 0 で生成                │
  │ teacher_paper_set    │ ✓ Phase 0 で生成                │
  │ problem_spec         │ ✓ LLM生成 + レビュー済み        │
  │ model_spec           │ ✓ 環境検出 + レビュー済み       │
  │ resource_spec        │ ✓ 環境検出 + レビュー済み       │
  │ plan_spec            │ ✓ LLM生成 + レビュー済み        │
  │ execution_spec       │ ✓ 既定値 + レビュー済み         │
  └──────────────────────┴─────────────────────────────────┘

  全 Spec を確定し、ExecutionSpec をロックしますか？
  ※ ロック後は ExecutionSpec の変更はできません。
    [1] 確定・ロック
    [2] ステップに戻って修正 (goto N)
  > 1
```

**11b: Freeze 処理**

```python
# 内部処理フロー
all_specs = AllSpecs(
    input1=input1_model,
    related_work=related_work_spec,
    paper=paper_spec,
    paper_score=paper_score_spec,
    teacher_paper_set=teacher_paper_set,
    problem=problem_spec,
    model=model_spec,
    resource=resource_spec,
    plan=plan_spec,
    execution=execution_spec,
)
all_specs.save_to_dir(specs_dir)

freezer = SpecFreezer()
freezer.freeze(all_specs, specs_dir)
# → specs/execution_spec.yaml.lock に SHA-256 ハッシュを記録
```

**完了メッセージ**：

```text
  ✓ 全 Spec を specs/ に保存しました
  ✓ ExecutionSpec をロックしました (SHA-256: a3b2c1...)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    セットアップ完了！
    次のコマンドで探索を開始できます:

      sera research

    探索状態の確認:
      sera status
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 26.6 ナビゲーション

#### 26.6.1 コマンド一覧

ウィザード内の任意のプロンプトで以下のコマンドが使用可能：

| コマンド | 説明 |
|---------|------|
| `back` | 前のステップに戻る |
| `goto N` | ステップ N にジャンプする（例: `goto 3`） |
| `quit` | 現在の状態を `.wizard_state.json` に保存して終了 |
| `help` | ヘルプメッセージを表示 |
| `status` | 現在の進捗（完了ステップ / 全ステップ）を表示 |

**`goto` の制約**：完了済みステップにのみジャンプ可能。未到達のステップにはジャンプできない（依存データが未収集のため）。

#### 26.6.2 WizardState データクラス

```python
@dataclasses.dataclass
class WizardState:
    """ウィザードの中断・再開用状態。"""
    current_step: int = 1
    completed_steps: list[int] = dataclasses.field(default_factory=list)
    lang: str = "ja"

    # Phase A: Input-1 データ
    data: dict = dataclasses.field(default_factory=dict)       # Step 1
    domain: dict = dataclasses.field(default_factory=dict)     # Step 2
    task: dict = dataclasses.field(default_factory=dict)       # Step 3
    goal: dict = dataclasses.field(default_factory=dict)       # Step 4
    constraints: list[dict] = dataclasses.field(default_factory=list)  # Step 5
    notes: str = ""                                             # Step 6
    input1_confirmed: bool = False                             # Step 7

    # Phase B: Phase 0 結果
    phase0_params: dict = dataclasses.field(default_factory=dict)  # Step 8
    phase0_completed: bool = False                             # Step 8
    phase0_review_done: bool = False                           # Step 9
    excluded_paper_ids: list[str] = dataclasses.field(default_factory=list)  # Step 9
    added_papers: list[dict] = dataclasses.field(default_factory=list)       # Step 9

    # Phase C: Spec データ
    specs_reviewed: dict[str, bool] = dataclasses.field(
        default_factory=lambda: {
            "problem": False,
            "model": False,
            "resource": False,
            "plan": False,
            "execution": False,
        }
    )  # Step 10
    specs_frozen: bool = False                                 # Step 11

    # メタデータ
    work_dir: str = "./sera_workspace"
    created_at: str = ""          # ISO 8601
    updated_at: str = ""          # ISO 8601
    version: str = "1"

    def save(self, path: Path) -> None:
        """状態を JSON ファイルに保存する。"""
        self.updated_at = datetime.utcnow().isoformat() + "Z"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "WizardState":
        """JSON ファイルから状態を復元する。"""
        data = json.loads(path.read_text())
        return cls(**data)
```

#### 26.6.3 状態保存・復元

**保存先**: `{work_dir}/.wizard_state.json`

**保存タイミング**：
- 各ステップ完了時（自動保存）
- `quit` コマンド実行時
- `Ctrl+C`（SIGINT）受信時

**`--resume` の動作**：
1. `{work_dir}/.wizard_state.json` の存在を確認
2. 存在すれば `WizardState.load()` で復元
3. `current_step` のステップから再開
4. 存在しなければエラー: `"中断状態が見つかりません。--resume なしで再実行してください。"`

**`--from-input1` の動作**：
1. 指定 YAML パスを `Input1Model` でバリデーション
2. バリデーション成功 → `WizardState` の Phase A フィールドを埋め、`current_step = 8` に設定
3. バリデーション失敗 → エラー表示して終了

#### 26.6.4 状態クリーンアップ

`sera setup` が Step 11 で正常完了した場合、`.wizard_state.json` を削除する。

### 26.7 UI コンポーネント

#### 26.7.1 Rich ライブラリの使用

ウィザード UI は [Rich](https://rich.readthedocs.io/) ライブラリを使用する。SERA は既に Rich を依存関係に含んでいる（`cli.py` で `Console` を使用）。

| コンポーネント | Rich クラス | 用途 |
|--------------|------------|------|
| コンソール出力 | `Console` | 全テキスト出力、カラー表示 |
| テキスト入力 | `Prompt.ask()` | 自由テキスト入力（バリデーション付き） |
| 確認 | `Confirm.ask()` | Yes/No の確認 |
| 選択式入力 | `Prompt.ask(choices=[...])` | 列挙値の選択 |
| テーブル | `Table` | 論文一覧、Spec サマリ等 |
| パネル | `Panel` | YAML プレビュー、ステップヘッダ |
| シンタックスハイライト | `Syntax` | YAML コードの表示 |
| 進捗バー | `Progress` | Phase 0 実行の進捗表示 |
| スピナー | `Status` | LLM 生成中の待機表示 |

#### 26.7.2 ステップヘッダ

各ステップの先頭に統一フォーマットのヘッダを表示する：

```python
def print_step_header(console: Console, step: int, total: int, title: str) -> None:
    """ステップヘッダを表示する。"""
    console.print()
    console.rule(f"[bold cyan]SERA セットアップウィザード  [Step {step}/{total}] {title}[/]")
    console.print()
```

#### 26.7.3 入力ヘルパー

```python
def prompt_with_nav(
    console: Console,
    message: str,
    *,
    default: str | None = None,
    choices: list[str] | None = None,
    required: bool = True,
    validator: Callable[[str], bool] | None = None,
) -> str | NavigationCommand:
    """ナビゲーションコマンド対応のプロンプト。

    Returns:
        入力値（str）または NavigationCommand（back/goto/quit/help/status）。
    """
    while True:
        raw = Prompt.ask(message, default=default, choices=choices, console=console)

        # ナビゲーションコマンドチェック
        nav = parse_navigation(raw)
        if nav is not None:
            return nav

        # バリデーション
        if required and not raw.strip():
            console.print("[red]入力は必須です。[/red]")
            continue
        if validator and not validator(raw.strip()):
            console.print("[red]入力値が不正です。再入力してください。[/red]")
            continue

        return raw.strip()
```

### 26.8 モジュール構成

#### 26.8.1 ファイル構成

```text
src/sera/
  cli.py                          # ← setup コマンドを追加
  commands/
    setup_cmd.py                  # ← 新規: エントリポイント run_setup()
  wizard/                         # ← 新規パッケージ
    __init__.py
    state.py                      # WizardState データクラス
    runner.py                     # WizardRunner オーケストレータ
    steps/
      __init__.py
      base.py                     # WizardStep 抽象基底クラス
      step_data.py                # Step 1: data.*
      step_domain.py              # Step 2: domain.*
      step_task.py                # Step 3: task.*
      step_goal.py                # Step 4: goal.*
      step_constraints.py         # Step 5: constraints[]
      step_notes.py               # Step 6: notes
      step_input1_confirm.py      # Step 7: Input-1確認
      step_phase0_run.py          # Step 8: Phase 0実行
      step_phase0_review.py       # Step 9: Phase 0レビュー
      step_spec_review.py         # Step 10: Spec確定
      step_freeze.py              # Step 11: Freeze
    i18n.py                       # MESSAGES 辞書
    ui.py                         # UIヘルパー (print_step_header, prompt_with_nav 等)
    env_detect.py                 # 環境自動検出 (detect_environment)
```

#### 26.8.2 WizardStep 抽象基底クラス

```python
from abc import ABC, abstractmethod
from sera.wizard.state import WizardState

class WizardStep(ABC):
    """ウィザードの1ステップを表す抽象基底クラス。"""

    step_number: int          # 1-11
    title_key: str            # i18n キー（例: "step_data_title"）

    def __init__(self, console: Console, state: WizardState, messages: dict):
        self.console = console
        self.state = state
        self.messages = messages  # MESSAGES[lang]

    @abstractmethod
    def run(self) -> StepResult:
        """ステップを実行する。

        Returns:
            StepResult: completed / back / goto(N) / quit
        """
        ...

    @abstractmethod
    def is_complete(self) -> bool:
        """このステップが完了済みかどうか。"""
        ...

    def can_skip(self) -> bool:
        """このステップがスキップ可能かどうか（既定: False）。"""
        return False
```

```python
@dataclasses.dataclass
class StepResult:
    """ステップの実行結果。"""
    action: str  # "completed" | "back" | "goto" | "quit"
    goto_step: int | None = None  # action="goto" の場合のみ
```

#### 26.8.3 WizardRunner オーケストレータ

```python
class WizardRunner:
    """ウィザード全体のフロー制御。"""

    def __init__(
        self,
        work_dir: Path,
        lang: str = "ja",
        resume: bool = False,
        from_input1: str | None = None,
        skip_phase0: bool = False,
    ):
        self.work_dir = work_dir
        self.console = Console()
        self.messages = MESSAGES[lang]
        self.state = self._load_or_create_state(resume)
        self.steps = self._build_steps()

        if from_input1:
            self._apply_from_input1(from_input1)
        if skip_phase0:
            self._apply_skip_phase0()

    def run(self) -> None:
        """ウィザードのメインループ。"""
        try:
            while self.state.current_step <= len(self.steps):
                step = self.steps[self.state.current_step - 1]

                # ステップヘッダ表示
                print_step_header(
                    self.console,
                    step.step_number,
                    len(self.steps),
                    self.messages[step.title_key],
                )

                # スキップ判定
                if step.can_skip():
                    self.state.current_step += 1
                    continue

                # ステップ実行
                result = step.run()

                # 結果に基づくフロー制御
                match result.action:
                    case "completed":
                        self.state.completed_steps.append(step.step_number)
                        self.state.current_step += 1
                        self.state.save(self._state_path)
                    case "back":
                        if self.state.current_step > 1:
                            self.state.current_step -= 1
                    case "goto":
                        if result.goto_step in self.state.completed_steps or result.goto_step == 1:
                            self.state.current_step = result.goto_step
                        else:
                            self.console.print("[red]未到達のステップにはジャンプできません。[/red]")
                    case "quit":
                        self.state.save(self._state_path)
                        self.console.print("[yellow]状態を保存しました。--resume で再開できます。[/yellow]")
                        return

            # 正常完了 → 状態ファイル削除
            self._state_path.unlink(missing_ok=True)
            self._print_completion()

        except KeyboardInterrupt:
            self.state.save(self._state_path)
            self.console.print("\n[yellow]中断しました。--resume で再開できます。[/yellow]")

    def _build_steps(self) -> list[WizardStep]:
        """全ステップを構築する。"""
        return [
            StepData(self.console, self.state, self.messages),            # 1
            StepDomain(self.console, self.state, self.messages),          # 2
            StepTask(self.console, self.state, self.messages),            # 3
            StepGoal(self.console, self.state, self.messages),            # 4
            StepConstraints(self.console, self.state, self.messages),     # 5
            StepNotes(self.console, self.state, self.messages),           # 6
            StepInput1Confirm(self.console, self.state, self.messages),   # 7
            StepPhase0Run(self.console, self.state, self.messages),       # 8
            StepPhase0Review(self.console, self.state, self.messages),    # 9
            StepSpecReview(self.console, self.state, self.messages),      # 10
            StepFreeze(self.console, self.state, self.messages),          # 11
        ]

    @property
    def _state_path(self) -> Path:
        return self.work_dir / ".wizard_state.json"
```

### 26.9 既存コードとの統合

#### 26.9.1 再利用する既存モジュール

| 既存モジュール | 使用箇所 | 呼び出し方法 |
|--------------|---------|-------------|
| `commands/init_cmd.run_init()` | Step 7: workspace 初期化 | Input-1 データを一時 YAML に書き出し、`run_init()` に渡す |
| `commands/phase0_cmd.run_phase0()` | Step 8: Phase 0 実行 | パラメータを `run_phase0()` に渡す。進捗コールバックを注入 |
| `phase1/spec_builder.SpecBuilder` | Step 10: Spec LLM 生成 | `build_problem_spec()`, `build_plan_spec()` を呼び出す |
| `phase1/spec_freezer.SpecFreezer` | Step 11: Freeze | `freeze()` を呼び出す |
| `specs.AllSpecs` | Step 11: 全 Spec 集約・保存 | `AllSpecs(...)` + `save_to_dir()` |
| `specs.*Model` | 全ステップ: バリデーション | Pydantic モデルの `model_validate()` でバリデーション |
| `agent/agent_llm.AgentLLM` | Step 10: LLM 生成 | `SpecBuilder` 経由で使用 |

#### 26.9.2 既存コードへの変更（最小限）

既存コードへの侵襲的変更は最小限に抑える：

| ファイル | 変更内容 |
|---------|---------|
| `cli.py` | `setup` コマンドの追加（1関数追加のみ） |
| `commands/phase0_cmd.py` | 進捗コールバック引数の追加（オプション、既定 `None`） |

`phase0_cmd.run_phase0()` への進捗コールバック追加例：

```python
# 変更前
def run_phase0(work_dir: str, topk: int = 10, ...) -> None:

# 変更後
def run_phase0(
    work_dir: str,
    topk: int = 10,
    ...,
    progress_callback: Callable[[str, float], None] | None = None,  # 追加
) -> None:
    # progress_callback("semantic_scholar", 0.4) のように呼び出し
    # None の場合は従来通り（既存動作に影響なし）
```

#### 26.9.3 新規コードから既存コードへの依存方向

```text
wizard/          →  commands/init_cmd      (run_init)
wizard/          →  commands/phase0_cmd    (run_phase0)
wizard/          →  phase1/spec_builder    (SpecBuilder)
wizard/          →  phase1/spec_freezer    (SpecFreezer)
wizard/          →  specs/*                (Pydantic モデル)
wizard/          →  agent/agent_llm        (AgentLLM)
commands/setup_cmd → wizard/runner         (WizardRunner)
cli              →  commands/setup_cmd     (run_setup)
```

依存は **一方向**（wizard → 既存コード）。既存コードから wizard への依存は発生しない。

### 26.10 エラーハンドリング

#### 26.10.1 Phase 0 API エラー

| エラー | 対応 |
|--------|------|
| 特定 API の HTTP 429 (Rate Limit) | exponential backoff でリトライ（最大5回）。失敗時は警告を表示して他APIに切り替え |
| 特定 API のタイムアウト | 警告表示、他APIで補完 |
| 全 API 失敗 | ユーザに選択肢を提示: (1) リトライ、(2) Phase 0 スキップ（手動で論文リストを用意）、(3) 中断 |
| ネットワーク未接続 | 検出時に即座にエラー表示。Phase 0 スキップを提案 |

#### 26.10.2 LLM Spec 生成エラー

| エラー | 対応 |
|--------|------|
| LLM 応答が JSON でない | 3段階フォールバック（§ CLAUDE.md 記載の `_parse_json_response` パターン）。3回リトライ後、テンプレート既定値を使用 |
| Pydantic バリデーション失敗 | エラーメッセージを LLM に渡して再生成（最大3回）。失敗時はテンプレート既定値 |
| API キー未設定（provider=openai/anthropic） | Step 10 開始前に検出し、キー設定を促す or ローカルモデルへの切り替えを提案 |
| ローカルモデルのロード失敗 | エラー詳細を表示。API プロバイダへの切り替えを提案 |

#### 26.10.3 Ctrl+C（SIGINT）ハンドリング

```python
import signal

def _handle_sigint(signum, frame):
    """SIGINT ハンドラ: 状態保存後に終了。"""
    raise KeyboardInterrupt  # WizardRunner.run() の except で捕捉

# WizardRunner.__init__ 内で登録
signal.signal(signal.SIGINT, _handle_sigint)
```

`WizardRunner.run()` の `except KeyboardInterrupt` ブロックで `WizardState.save()` を呼び出し、終了コード 20 で終了する（既存の SERA 規約: §16.4 に準拠）。

#### 26.10.4 ワークスペース競合

既に `specs/execution_spec.yaml.lock` が存在する場合（過去の `sera freeze-specs` 完了済み）：

```text
  ⚠ このワークスペースには既にロック済みの ExecutionSpec があります。
  既存のセットアップを上書きしますか？
    [1] 上書きする (既存のロックを解除)
    [2] 別のワークスペースを指定する
    [3] 中断する
  >
```

### 26.11 テスト戦略

#### 26.11.1 テストファイル構成

```text
tests/
  test_wizard/
    __init__.py
    conftest.py                    # 共通フィクスチャ
    test_state.py                  # WizardState の保存/復元テスト
    test_runner.py                 # WizardRunner のフロー制御テスト
    test_steps/
      __init__.py
      test_step_data.py            # Step 1
      test_step_domain.py          # Step 2
      test_step_task.py            # Step 3
      test_step_goal.py            # Step 4
      test_step_constraints.py     # Step 5
      test_step_notes.py           # Step 6
      test_step_input1_confirm.py  # Step 7
      test_step_phase0_run.py      # Step 8
      test_step_phase0_review.py   # Step 9
      test_step_spec_review.py     # Step 10
      test_step_freeze.py          # Step 11
    test_i18n.py                   # i18n メッセージの網羅性テスト
    test_ui.py                     # UIヘルパーのテスト
    test_env_detect.py             # 環境検出のテスト
    test_nav.py                    # ナビゲーションコマンドのテスト
    test_integration.py            # E2E統合テスト（モック使用）
```

#### 26.11.2 テストパターン

**入力のモック**：`monkeypatch` で `rich.prompt.Prompt.ask` / `rich.prompt.Confirm.ask` をモックする。

```python
def test_step_data(monkeypatch, tmp_workspace):
    """Step 1: data.* の収集テスト。"""
    inputs = iter([
        "CIFAR-10 image dataset",   # description
        "/data/cifar10",             # location
        "1",                         # format: csv
        "1",                         # size_hint: small
    ])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: next(inputs))

    state = WizardState()
    step = StepData(Console(), state, MESSAGES["ja"])
    result = step.run()

    assert result.action == "completed"
    assert state.data["description"] == "CIFAR-10 image dataset"
    assert state.data["format"] == "csv"
```

**LLM のモック**：既存の `set_mock()` パターンを使用。

```python
def test_step_spec_review(tmp_workspace, mock_llm_response):
    """Step 10: Spec生成のテスト。"""
    agent_llm = AgentLLM(mock_model_spec, mock_resource_spec, log_path)
    agent_llm.set_mock(lambda prompt, purpose: json.dumps({
        "problem_spec": {"title": "Test", "objective": {"metric_name": "score", "direction": "maximize"}}
    }))
    # ... SpecBuilder 経由でテスト
```

**ナビゲーションのテスト**：

```python
def test_back_navigation(monkeypatch, tmp_workspace):
    """back コマンドで前ステップに戻れること。"""
    inputs = iter(["back"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: next(inputs))

    state = WizardState(current_step=3)
    step = StepTask(Console(), state, MESSAGES["ja"])
    result = step.run()

    assert result.action == "back"
```

**状態の保存/復元テスト**：

```python
def test_state_roundtrip(tmp_path):
    """WizardState の保存→復元で情報が失われないこと。"""
    state = WizardState(
        current_step=5,
        completed_steps=[1, 2, 3, 4],
        data={"description": "test", "location": "/tmp", "format": "csv", "size_hint": "small(<1GB)"},
    )
    path = tmp_path / ".wizard_state.json"
    state.save(path)

    loaded = WizardState.load(path)
    assert loaded.current_step == 5
    assert loaded.data["description"] == "test"
```

**統合テスト**（`test_integration.py`）：

```python
@pytest.mark.slow
def test_full_wizard_flow(monkeypatch, tmp_path):
    """ウィザードの全ステップを通しで実行するE2Eテスト。"""
    # 全ステップの入力をモック
    # Phase 0 の API 呼び出しをモック
    # LLM 呼び出しをモック
    # 最終的に specs/ に全 YAML が生成され、.lock が存在することを確認
```

#### 26.11.3 テストマーカー

| マーカー | 対象テスト |
|---------|-----------|
| なし（デフォルト） | ユニットテスト（モック使用、高速） |
| `@pytest.mark.slow` | E2E 統合テスト |
| `@pytest.mark.network` | 実 API を呼ぶテスト（CI では skip） |

### 26.12 i18n メッセージ辞書

#### 26.12.1 構造

```python
# src/sera/wizard/i18n.py

MESSAGES: dict[str, dict[str, str]] = {
    "ja": {
        # ステップタイトル
        "step_data_title": "データ情報",
        "step_domain_title": "分野情報",
        "step_task_title": "タスク情報",
        "step_goal_title": "目標情報",
        "step_constraints_title": "制約条件",
        "step_notes_title": "自由記述",
        "step_input1_confirm_title": "入力確認",
        "step_phase0_run_title": "Phase 0 実行",
        "step_phase0_review_title": "結果レビュー",
        "step_spec_review_title": "Spec確定",
        "step_freeze_title": "最終確認",

        # プロンプト（Step 1）
        "data_description_prompt": "データの説明を入力してください",
        "data_description_example": '例: "画像分類用のCIFAR-10データセット"',
        "data_location_prompt": "データの場所を入力してください（パス/URI/リポジトリURL）",
        "data_format_prompt": "データ形式を選択してください",
        "data_size_prompt": "データサイズの目安を選択してください",

        # ... 全ステップのプロンプト

        # ナビゲーション
        "nav_help": "back: 戻る / goto N: ジャンプ / quit: 中断保存 / help: ヘルプ / status: 進捗",
        "nav_back": "前のステップに戻ります",
        "nav_quit_saved": "状態を保存しました。--resume で再開できます。",
        "nav_goto_invalid": "未到達のステップにはジャンプできません。",

        # エラー
        "error_required": "入力は必須です。",
        "error_invalid": "入力値が不正です。再入力してください。",

        # 完了
        "setup_complete": "セットアップ完了！",
        "next_command": "次のコマンドで探索を開始できます:",
    },
    "en": {
        "step_data_title": "Data Information",
        "step_domain_title": "Domain Information",
        "step_task_title": "Task Information",
        "step_goal_title": "Goal Information",
        "step_constraints_title": "Constraints",
        "step_notes_title": "Notes",
        "step_input1_confirm_title": "Input Confirmation",
        "step_phase0_run_title": "Phase 0 Execution",
        "step_phase0_review_title": "Results Review",
        "step_spec_review_title": "Spec Configuration",
        "step_freeze_title": "Final Confirmation",

        "data_description_prompt": "Describe your data",
        "data_description_example": 'e.g., "CIFAR-10 image classification dataset"',
        "data_location_prompt": "Enter data location (path/URI/repository URL)",
        "data_format_prompt": "Select data format",
        "data_size_prompt": "Select approximate data size",

        # ...

        "nav_help": "back: go back / goto N: jump / quit: save & exit / help / status: progress",
        "nav_back": "Going back to previous step",
        "nav_quit_saved": "State saved. Use --resume to continue.",
        "nav_goto_invalid": "Cannot jump to an unreached step.",

        "error_required": "This field is required.",
        "error_invalid": "Invalid input. Please try again.",

        "setup_complete": "Setup complete!",
        "next_command": "Start research with:",
    },
}
```

#### 26.12.2 網羅性保証

テストで `ja` と `en` のキーセットが一致することを検証する：

```python
def test_i18n_key_coverage():
    """全言語で同じキーセットが存在すること。"""
    ja_keys = set(MESSAGES["ja"].keys())
    en_keys = set(MESSAGES["en"].keys())
    assert ja_keys == en_keys, f"Missing keys: ja-en={ja_keys - en_keys}, en-ja={en_keys - ja_keys}"
```

### 26.13 実装優先順位

| 優先度 | スコープ | 内容 | 依存 |
|--------|---------|------|------|
| **P0** | Phase A 基盤 | `WizardState`, `WizardRunner`, `WizardStep` 基底クラス, `i18n`, `ui` ヘルパー | なし |
| **P0** | Phase A ステップ | Step 1-7（Input-1 構築）の全ステップ実装 | P0 基盤 |
| **P1** | Phase B 統合 | Step 8-9（Phase 0 実行 + レビュー）。`phase0_cmd` への進捗コールバック追加含む | P0 |
| **P1** | Phase C 統合 | Step 10-11（Spec 生成 + Freeze）。`SpecBuilder`, `SpecFreezer` との統合 | P0 |
| **P2** | 中断再開 | `--resume` 機能の完全実装・テスト | P0 |
| **P2** | 環境検出 | `detect_environment()` + `ResourceSpec` / `ModelSpec` への自動反映 | P1 |
| **P2** | i18n | 英語メッセージの完全翻訳・テスト | P0 |

---

## 26.14 実装レビュー：仕様との差異一覧

> **レビュー日**: 2026-02-25
> **対象ソース**: `commands/wizard/` 配下全ファイル、`commands/setup_cmd.py`

以下は §26.5 で定義された仕様と、現在の実装（`step1_data.py` 〜 `step11_freeze.py`, `state.py`, `runner.py`, `env_detect.py`）を突き合わせた結果である。

---

### 26.14.1 【高】Step 10: Spec レビューがほぼ未実装

**仕様 (§26.5.10)**: 5段階のサブステップで各 Spec を個別にレビュー・編集する。

| サブステップ | 仕様の内容 | 実装状況 |
|---|---|---|
| 10a: ProblemSpec | LLM 生成 → フィールド単位編集 / YAML 直接編集 / 再生成 | **未実装** |
| 10b: ModelSpec | 環境検出結果表示 → provider/model_id 選択 → API キーチェック | **未実装** |
| 10c: ResourceSpec | 環境検出値を初期値に反映 → executor/GPU/CPU/メモリの確認 | **一部のみ**（executor, gpu_required のみ） |
| 10d: PlanSpec | LLM 生成 → サマリ確認 | **未実装** |
| 10e: ExecutionSpec | 主要パラメータ一覧表示 → 編集可能 | **一部のみ**（max_nodes, repeats のみ） |

**現状の `step10_specs.py`** は以下の4パラメータしか収集していない:
```python
params.setdefault("executor", "slurm" if env["slurm_available"] else "local")
params.setdefault("gpu_required", env["gpu_available"])
params.setdefault("max_nodes", 100)
params.setdefault("repeats", 3)
```

修正時にユーザが確認・編集可能にすべきパラメータも3つのみ（`executor`, `max_nodes`, `repeats`）。

**修正方針**: Step 10 を仕様通り 5 サブステップに分割し、各 Spec の LLM 生成→レビュー→編集フローを実装する。最低限、以下のパラメータは Step 10 で収集して `phase1_params` に格納する:

| Spec | 収集すべきパラメータ |
|------|-------------------|
| ModelSpec (10b) | `base_model`, `dtype`, `agent_llm` (provider:model_id), `rank`, `alpha` |
| ResourceSpec (10c) | `gpu_count`, `memory_gb`, `cpu_cores`, `gpu_type`, `timeout`, `no_web`, **`mcp.servers`** |
| PlanSpec (10d) | `reward.method`, `echo.enabled`, **`agent_commands.tools.enabled`**, **`agent_commands.functions.available_functions`**, **`agent_commands.functions.function_tool_bindings`**, `loop_defaults`, `function_loop_overrides` |
| ExecutionSpec (10e) | `max_depth`, `branch_factor`, `lambda_cost`, `beta`, `lcb_coef`, `lr`, `clip`, `ppo_steps`, `squash_depth`, `strategy` |

### 26.14.2 【高】Step 11: freeze へのフラグ欠落

**仕様 (§26.5.11)**: `AllSpecs` を構築し `SpecFreezer.freeze()` に渡す。

**現状の `step11_freeze.py`**: `run_freeze_specs()` に渡す `cli_args` が6キーしかないため、Step 10 で収集しても `spec_builder.py` に届かない。

```python
# 現状: 6キーのみ
cli_args = {
    "work_dir": str(work_dir),
    "auto": True,
    "max_nodes": params.get("max_nodes", 100),
    "repeats": params.get("repeats", 3),
    "executor": params.get("executor", "local"),
    "gpu_required": params.get("gpu_required", True),
}
```

**修正方針**: `phase1_params` の全キーを `cli_args` に展開する。Step 10 で収集しなかったパラメータは `cli.py freeze_specs` と同じデフォルト値を明示的にセットする。

```python
# 修正後: cli.py freeze_specs の全パラメータに対応
cli_args = {
    "work_dir": str(work_dir),
    "auto": True,
    # ResourceSpec
    "executor": params.get("executor", "local"),
    "gpu_required": params.get("gpu_required", True),
    "gpu_count": params.get("gpu_count", 1),
    "memory_gb": params.get("memory_gb", 32),
    "cpu_cores": params.get("cpu_cores", 8),
    "gpu_type": params.get("gpu_type", ""),
    "timeout": params.get("timeout", 3600),
    "no_web": params.get("no_web", False),
    # ModelSpec
    "base_model": params.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "dtype": params.get("dtype", "bf16"),
    "agent_llm": params.get("agent_llm", "local:same_as_base"),
    "rank": params.get("rank", 16),
    "alpha": params.get("alpha", 32),
    # ExecutionSpec
    "max_nodes": params.get("max_nodes", 100),
    "max_depth": params.get("max_depth", 10),
    "branch_factor": params.get("branch_factor", 3),
    "lambda_cost": params.get("lambda_cost", 0.1),
    "beta": params.get("beta", 0.05),
    "repeats": params.get("repeats", 3),
    "lcb_coef": params.get("lcb_coef", 1.96),
    "no_sequential": params.get("no_sequential", False),
    "seq_topk": params.get("seq_topk", 5),
    "lr": params.get("lr", 1e-4),
    "clip": params.get("clip", 0.2),
    "ppo_steps": params.get("ppo_steps", 128),
    "no_kl": params.get("no_kl", False),
    "squash_depth": params.get("squash_depth", 6),
    "no_snapshot_topk": params.get("no_snapshot_topk", False),
    "strategy": params.get("strategy", "best_first"),
    # Turn rewards
    "no_turn_rewards": params.get("no_turn_rewards", False),
    "turn_w_phase0": params.get("turn_w_phase0", 0.10),
    "turn_w_phase2": params.get("turn_w_phase2", 0.15),
    "turn_w_phase3": params.get("turn_w_phase3", 0.25),
    "turn_w_phase4": params.get("turn_w_phase4", 0.35),
    "turn_w_phase7": params.get("turn_w_phase7", 0.15),
    # PlanSpec — reward / agent_commands
    "reward_method": params.get("reward_method", "hiper"),
    "echo_enabled": params.get("echo_enabled", True),
    "tools_enabled": params.get("tools_enabled", True),
    "available_functions": params.get("available_functions", None),  # None = 全19関数
    "function_tool_bindings": params.get("function_tool_bindings", None),  # None = デフォルト
    "loop_max_steps": params.get("loop_max_steps", 10),
    "loop_tool_budget": params.get("loop_tool_budget", 20),
    "loop_timeout": params.get("loop_timeout", 300.0),
    # ResourceSpec — MCP
    "mcp_servers": params.get("mcp_servers", []),  # list[MCPServerConfig]
}
```

### 26.14.3 【高】環境検出が不十分

**仕様 (§26.5.10b)**: `detect_environment()` は以下を検出する:

| 検出項目 | 仕様の方法 | 実装状況 |
|---------|-----------|---------|
| GPU 有無 | `torch.cuda.is_available()` | `nvidia-smi` コマンド（torch 不要で動作する利点はある） |
| GPU 台数 | `torch.cuda.device_count()` | **未実装** — `nvidia-smi` 出力の先頭行のみ取得 |
| GPU 型番 | `torch.cuda.get_device_name(0)` | **未実装** — `gpu_info` に文字列として含まれるが未パース |
| GPU メモリ | `torch.cuda.get_device_properties(0).total_mem` | **未実装** — 同上 |
| SLURM 有無 | `shutil.which("sbatch")` | `shutil.which("sinfo")`（同等） |
| SLURM partition | `os.environ.get("SLURM_PARTITION")` | **未実装** |
| SLURM account | `os.environ.get("SLURM_ACCOUNT")` | **未実装** |
| システムメモリ | `psutil.virtual_memory().total` | **未実装** |
| CPU コア数 | `os.cpu_count()` | **未実装** |

**影響**: 仕様では検出結果を `ResourceSpec` の初期値に反映（§26.5.10c `build_resource_defaults()`）するが、検出情報がないため `gpu_count`, `memory_gb`, `cpu_cores`, `gpu_type` は常にデフォルト値になる。

**修正方針**: `env_detect.py` を拡張して全項目を検出する。`nvidia-smi` の `--query-gpu` 出力を全行パースして台数・型番・メモリを取得する。CPU/メモリは `os.cpu_count()` + `psutil`（または `/proc/meminfo` フォールバック）で取得する。

### 26.14.4 【中】Step 9: レビュー機能が不足

**仕様 (§26.5.9)**: 3つのサブセクションで結果をレビューする。

| 機能 | 仕様 | 実装状況 |
|------|------|---------|
| 9a: 収集論文一覧（被引用数・関連度付きテーブル） | Rich Table で表示 | **簡易表示のみ**（タイトルと年のみ、先頭10件） |
| 9b: クラスタ表示 | クラスタ名・説明・論文数のテーブル | **未実装** |
| 9c: 教師論文候補 | 候補一覧 + 確認 | **未実装** |
| 論文除外操作 | 番号指定で除外 → 再表示 | **未実装** |
| 論文追加操作 | DOI/arXiv ID で検索→追加 | **未実装** |
| 論文詳細表示 | 番号指定でアブストラクト等表示 | **未実装** |

**修正方針**: Rich `Table` を使って仕様通りの表形式表示を実装し、除外・追加・詳細表示の対話操作を追加する。

### 26.14.5 【中】Step 8: パラメータ編集が限定的

**仕様 (§26.5.8)**: 4つの Phase 0 パラメータ全てを編集可能にする。

| パラメータ | 仕様 | 実装状況 |
|-----------|------|---------|
| `topk` | 編集可能 | 編集可能 |
| `teacher_papers` | 編集可能 | 編集可能 |
| `citation_depth` | 編集可能 | **表示のみ、編集不可** |
| `years_bias` | 編集可能 | **表示のみ、編集不可** |

**修正方針**: `step8_phase0.py` の「Modify parameters?」ブランチに `citation_depth` と `years_bias` の `IntPrompt.ask` を追加する。

### 26.14.6 【中】WizardState が仕様より簡素

**仕様 (§26.6.2)**: `@dataclasses.dataclass` で多数のフィールドを持つ。

| フィールド | 仕様 | 実装状況 |
|-----------|------|---------|
| `current_step` | int | `current_step: int` |
| `completed_steps` | list[int] | **なし** |
| `lang` | str | **なし**（`runner.py` が保持） |
| `data`, `domain`, `task`, `goal` (個別フィールド) | 各 dict | `input1_data: dict` に統合（機能上は同等） |
| `constraints` | list[dict] | `input1_data["constraints"]` に統合 |
| `notes` | str | `input1_data["notes"]` に統合 |
| `input1_confirmed` | bool | **なし** |
| `phase0_completed` | bool | **なし** |
| `phase0_review_done` | bool | **なし** |
| `excluded_paper_ids` | list[str] | **なし**（Step 9 除外機能自体が未実装） |
| `added_papers` | list[dict] | **なし**（Step 9 追加機能自体が未実装） |
| `specs_reviewed` | dict[str, bool] | **なし**（Step 10 個別レビュー自体が未実装） |
| `specs_frozen` | bool | **なし** |
| `created_at`, `updated_at` | str (ISO 8601) | **なし** |
| `version` | str | **なし** |

**影響**: `completed_steps` がないため、仕様の「完了済みステップにのみ goto 可能」制約を検証できない。`specs_reviewed` がないため、どの Spec がレビュー済みか追跡できない。

**修正方針**: Input-1 フィールドの統合（`input1_data` dict）は実装上合理的なのでそのまま維持し、不足フィールド（`completed_steps`, `specs_reviewed`, `excluded_paper_ids`, `added_papers`, タイムスタンプ等）を追加する。

### 26.14.7 【中】Step 7: Input1Model バリデーション欠落

**仕様 (§26.5.7 確定時の処理)**: 「`Input1Model(**collected_data)` で Pydantic バリデーション」を行い、失敗なら再編集を促す。

**実装**: `step7_preview.py` は YAML プレビュー + 確認のみで、`Input1Model` によるバリデーションを行っていない。YAML 直接編集（選択肢 3）後の再バリデーションもない。

**修正方針**: 確認時に `Input1Model(**state.input1_data)` でバリデーションし、失敗時はエラーを表示して再編集を促す。

### 26.14.8 【低】WizardStep 抽象基底クラス未使用 — ✅ 解決済み

> **解決日**: 2026-02-25。`runtime_checkable` な `Protocol` クラスとして `WizardStep` / `WizardStepWithDir` を `commands/wizard/runner.py` に定義し、`commands/wizard/__init__.py` からエクスポート済み。

```python
@runtime_checkable
class WizardStep(Protocol):
    def __call__(self, state: WizardState, lang: str) -> None: ...

@runtime_checkable
class WizardStepWithDir(Protocol):
    def __call__(self, state: WizardState, lang: str, work_dir: Path) -> None: ...
```

**設計判断**: ABC ではなく `Protocol` を採用。関数ベースのステップ実装を維持しつつ、型チェッカーによる検証が可能。`is_complete()` / `can_skip()` 等の判定ロジックは現時点では不要（Step 10 のサブステップ化で必要になった場合に拡張）。

### 26.14.9 【低】ナビゲーション: help / status / quit コマンド未実装

**仕様 (§26.6.1)**: 5つのナビゲーションコマンドが使用可能。

| コマンド | 仕様 | 実装状況 |
|---------|------|---------|
| `back` | 前のステップに戻る | `NavigateBack` 例外で実装済み |
| `goto N` | ステップ N にジャンプ | `NavigateGoto` 例外で実装済み |
| `quit` | 状態保存して終了 | **未実装**（`Ctrl+C` / `EOFError` でカバー） |
| `help` | ヘルプ表示 | **未実装** |
| `status` | 進捗表示 | **未実装** |

**修正方針**: `prompt_with_nav` 相当のラッパーを `ui.py` に追加し、入力テキストに対するコマンド解析を行う。`quit` は `sys.exit(0)` の前に `state.save()`、`help` は利用可能コマンド一覧表示、`status` は `current_step / 11` 表示。

### 26.14.10 【低】Phase 0 進捗表示なし

**仕様 (§26.5.8c)**: `Rich.Progress` による進捗バーとリアルタイム状況表示。

**実装**: `step8_phase0.py` は `phase0_cmd.run_phase0()` を直接呼び出すのみで、進捗表示なし。仕様ではこのために `run_phase0()` に `progress_callback` 引数を追加する（§26.9.2）とされているが未実装。

**修正方針**: `phase0_cmd.run_phase0()` にオプショナルな `progress_callback: Callable | None = None` 引数を追加し、Step 8 から `Rich.Progress` でラップして渡す。

---

### 26.14.11 修正優先度まとめ

| # | 問題 | 優先度 | 影響 |
|---|------|-------|------|
| 27.14.1 | Step 10 の Spec レビュー5段階がほぼ未実装 | **高** | ProblemSpec/ModelSpec/PlanSpec/ExecutionSpec のレビュー・編集ができない |
| 27.14.2 | Step 11 の freeze へのフラグ欠落 | **高** | Step 10 で収集したパラメータが spec_builder に届かない |
| 27.14.3 | 環境検出が GPU 有無と SLURM 有無のみ | **高** | gpu_count/gpu_type/memory_gb/cpu_cores が常にデフォルト固定 |
| 27.14.4 | Step 9 のレビュー機能不足 | **中** | 論文の除外・追加・クラスタ表示・教師論文確認ができない |
| 27.14.5 | Step 8 の Phase 0 パラメータ編集が限定的 | **中** | citation_depth, years_bias が編集不可 |
| 27.14.6 | WizardState のフィールド不足 | **中** | completed_steps, specs_reviewed 等の追跡ができない |
| 27.14.7 | Step 7 の Input1Model バリデーション欠落 | **中** | 不正な Input-1 が後続フェーズに渡る可能性 |
| 27.14.8 | WizardStep ABC 未使用 | **低** ✅ | Protocol として実装済み（WizardStep / WizardStepWithDir） |
| 27.14.9 | help/status/quit コマンド未実装 | **低** | Ctrl+C で代替可能だが UX が仕様より劣る |
| 27.14.10 | Phase 0 進捗表示なし | **低** | 長時間処理の UX 改善 |
