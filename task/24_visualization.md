# SERA 要件定義書 — 探索木可視化ツール

> 本ファイルは TASK.md v13.0 の拡張仕様である。目次は [README.md](./README.md) を参照。

---

## 30. 探索木可視化ツール

### 30.1 目的

SERAの探索木（Best-First Search Tree）の状態をインタラクティブなHTML形式で可視化し、研究者がノード間の関係・性能推移・探索戦略を直感的に把握できるようにする。

### 30.2 CLI コマンド

```bash
# 最新チェックポイントから可視化HTML生成
sera visualize

# 特定ステップのチェックポイントを可視化
sera visualize --step 30

# 出力先を指定（デフォルト: sera_workspace/outputs/tree_visualization.html）
sera visualize --output path/to/output.html

# ライブモード: research実行中にファイル監視で自動更新
sera visualize --live --port 8080
```

**実装場所**: `src/sera/commands/visualize_cmd.py`（新規）、`cli.py` にサブコマンド追加。

---

### 30.3 データソース

可視化に使用するデータは以下のファイルから取得する：

| データ | ソース | 取得方法 |
|--------|--------|----------|
| ノード全体 | `checkpoints/search_state_step_N.json` → `all_nodes` | `load_latest_checkpoint()` or 指定step |
| オープンリスト | 同上 → `open_list` | heapq形式 `[(-priority, node_id), ...]` |
| クローズドセット | 同上 → `closed_set` | node_idリスト |
| ベストノード | 同上 → `best_node_id` | node_id |
| 現在ステップ | 同上 → `step` | int |
| 実験コード | `runs/<node_id>/experiment.py` | ファイル読み込み（存在する場合） |
| 実行ログ | `runs/<node_id>/stdout.log`, `stderr.log` | ファイル読み込み（存在する場合） |
| メトリクス | `runs/<node_id>/metrics.json` | ファイル読み込み（存在する場合） |

---

### 30.4 HTML出力仕様

単一の自己完結型HTMLファイル（外部依存なし）を生成する。CSS・JavaScriptは全てインライン埋め込み。

#### 30.4.1 全体レイアウト

```
┌─────────────────────────────────────────────────────────┐
│  Header: プロジェクト名 / ステップ数 / ノード総数        │
├──────────────────────┬──────────────────────────────────┤
│                      │                                  │
│   ツリービュー        │   ノード詳細パネル               │
│   (メイン可視化)      │   (選択ノードの情報)             │
│                      │                                  │
├──────────────────────┴──────────────────────────────────┤
│  統計サマリーパネル                                       │
│  (探索全体の統計情報)                                     │
└─────────────────────────────────────────────────────────┘
```

#### 30.4.2 ツリービュー（メイン領域）

**表示方式**: D3.js ベースのインタラクティブツリー図（d3.jsをインライン埋め込みまたはCDN）。

**ノード表現**:

| 要素 | マッピング |
|------|-----------|
| ノードの色 | ステータスに基づく配色（下表） |
| ノードのサイズ | LCB値に比例（大きい = 良い性能） |
| ノードのアイコン/形状 | `branching_op` に基づく: draft=丸、debug=三角、improve=四角 |
| ノードの枠線 | ベストノード = 金色太枠、feasible=false = 赤点線枠 |
| エッジ（辺） | `parent_id` → `node_id` の親子関係 |
| エッジのスタイル | debug=赤破線、improve=青実線、draft=灰色実線 |

**ステータス別配色**:

| ステータス | 色 | 意味 |
|-----------|-----|------|
| `pending` | `#E0E0E0` (灰色) | 未評価 |
| `running` | `#FFF176` (黄色) | 実行中 |
| `evaluated` | `#81C784` (緑色) | 評価完了 |
| `failed` | `#E57373` (赤色) | 失敗 |
| `timeout` | `#FFB74D` (橙色) | タイムアウト |
| `oom` | `#CE93D8` (紫色) | メモリ不足 |
| `pruned` | `#BDBDBD` (薄灰色) + 取消線 | 剪定済み |
| `expanded` | `#64B5F6` (青色) | 展開済み |

**インタラクション**:
- ノードクリック → 詳細パネルに情報表示
- ノードホバー → ツールチップ（hypothesis先頭50文字、μ±SE、status）
- ドラッグでパン、スクロールでズーム
- ダブルクリック → サブツリーの折りたたみ/展開

#### 30.4.3 ノード詳細パネル（右側）

選択したノードについて以下を表示する：

```
┌─ ノード詳細 ──────────────────────────┐
│ Node ID: 265b3a0a-72c9-...            │
│ Status:  evaluated  ●                  │
│ Operator: improve                      │
│ Depth: 1                               │
│ Created: 2026-02-25T10:30:00Z          │
│                                        │
│ ── 仮説 ──                             │
│ Integrating machine learning           │
│ algorithms to predict optimal...       │
│                                        │
│ ── 分岐理由 ──                          │
│ Parent node showed promise with...     │
│                                        │
│ ── 実験設定 (experiment_config) ──      │
│ ┌────────────────┬────────────┐        │
│ │ Key            │ Value      │        │
│ ├────────────────┼────────────┤        │
│ │ learning_rate  │ 0.001      │        │
│ │ batch_size     │ 64         │        │
│ │ method         │ "XGBoost"  │        │
│ └────────────────┴────────────┘        │
│                                        │
│ ── 評価メトリクス ──                    │
│ μ = 5387.32                            │
│ SE = 12.498                            │
│ LCB = 5362.82                          │
│ Eval Runs: 3 / 3                       │
│ Feasible: ✓                            │
│ Priority: 5362.82                      │
│                                        │
│ ── リソース ──                          │
│ Total Cost: 0.15                       │
│ Wall Time: 7.6s                        │
│                                        │
│ ── 失敗コンテキスト (ECHO) ──           │
│ [runtime] Avoid invalid memory...      │
│                                        │
│ ── 子ノード ──                          │
│ • node_2557b... (improve, evaluated)   │
│                                        │
│ ── エラーメッセージ ──                   │
│ (failedの場合のみ表示)                  │
│                                        │
│ [実験コード表示] [stdout] [stderr]      │
└────────────────────────────────────────┘
```

**「実験コード表示」「stdout」「stderr」ボタン**: クリックでモーダルを開き、`runs/<node_id>/` 配下のファイル内容をシンタックスハイライト付きで表示する。ファイルが存在しない場合はボタンを非活性にする。

#### 30.4.4 統計サマリーパネル（下部）

| 項目 | 内容 |
|------|------|
| 探索ステップ数 | 現在の `step` 値 |
| 総ノード数 | `len(all_nodes)` |
| ステータス別集計 | 各ステータスのノード数（棒グラフ） |
| オペレータ別集計 | draft / debug / improve のノード数（円グラフ） |
| ベストノード | best_node_id、μ±SE、LCB |
| 性能推移グラフ | X軸=ステップ、Y軸=best LCB値（折れ線グラフ）|
| 深度分布 | ツリーの深度ごとのノード数（ヒストグラム） |
| 成功率 | evaluated / (evaluated + failed + timeout + oom) |

---

### 30.5 可視化生成モジュール

#### 30.5.1 モジュール構成

```
src/sera/visualization/
├── __init__.py
├── tree_visualizer.py      # メインクラス: チェックポイント読込 → HTML生成
├── html_renderer.py        # HTMLテンプレート生成（Jinja2不使用、f-string）
├── node_formatter.py       # SearchNode → 表示用dict変換
└── stats_calculator.py     # 統計情報の集計
```

#### 30.5.2 `TreeVisualizer` クラス

```python
class TreeVisualizer:
    """探索木のHTML可視化を生成する。"""

    def __init__(self, workspace_dir: Path):
        """
        Parameters
        ----------
        workspace_dir : Path
            sera_workspace/ のパス
        """

    def load_checkpoint(self, step: int | None = None) -> dict:
        """
        チェックポイントを読み込む。

        Parameters
        ----------
        step : int | None
            指定stepのチェックポイントを読む。Noneなら最新。

        Returns
        -------
        dict
            チェックポイントデータ（all_nodes, open_list, closed_set, best_node_id, step）
        """

    def build_tree_data(self, checkpoint: dict) -> dict:
        """
        チェックポイントからD3.js用の階層データ構造を構築する。

        Returns
        -------
        dict
            D3.js hierarchy形式:
            {
                "id": "root",
                "children": [
                    {
                        "id": node_id,
                        "data": { ...SearchNode fields... },
                        "children": [ ... ]
                    },
                    ...
                ]
            }
        """

    def collect_run_artifacts(self, node_id: str) -> dict:
        """
        runs/<node_id>/ 配下の実験成果物を収集する。

        Returns
        -------
        dict
            {
                "experiment_code": str | None,
                "stdout": str | None,
                "stderr": str | None,
                "metrics": dict | None
            }
        """

    def compute_stats(self, checkpoint: dict) -> dict:
        """
        統計サマリー情報を計算する。

        Returns
        -------
        dict
            {
                "step": int,
                "total_nodes": int,
                "status_counts": {"evaluated": 5, "failed": 2, ...},
                "operator_counts": {"draft": 3, "debug": 1, "improve": 4},
                "best_node": {"node_id": str, "mu": float, "se": float, "lcb": float},
                "depth_distribution": {0: 5, 1: 3, 2: 1},
                "success_rate": float,
                "best_lcb_history": [{"step": int, "lcb": float}, ...]
            }
        """

    def generate_html(self, step: int | None = None, output_path: Path | None = None) -> Path:
        """
        HTML可視化ファイルを生成する。

        Parameters
        ----------
        step : int | None
            可視化するステップ。Noneなら最新。
        output_path : Path | None
            出力先。Noneなら sera_workspace/outputs/tree_visualization.html

        Returns
        -------
        Path
            生成されたHTMLファイルのパス
        """
```

#### 30.5.3 データフロー

```
checkpoint JSON
    │
    ▼
load_checkpoint()          ← checkpoints/search_state_step_N.json
    │
    ├─► build_tree_data()  ← all_nodes の parent_id/children_ids を再帰走査
    │       │
    │       ▼
    │   D3 hierarchy JSON（HTMLに埋め込み）
    │
    ├─► collect_run_artifacts()  ← runs/<node_id>/ のファイル読み込み
    │       │
    │       ▼
    │   各ノードの実験コード・ログ（HTMLに埋め込み）
    │
    ├─► compute_stats()    ← all_nodes を集計
    │       │
    │       ▼
    │   統計データ JSON（HTMLに埋め込み）
    │
    └─► generate_html()
            │
            ▼
        self-contained HTML file
```

---

### 30.6 HTML テンプレート構造

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>SERA Search Tree Visualization — Step {step}</title>
    <style>
        /* 全CSSをインライン埋め込み */
        /* レスポンシブレイアウト: flexbox */
        /* ノードステータス別の色定義 */
        /* ツールチップ・モーダルのスタイル */
    </style>
</head>
<body>
    <header>
        <!-- プロジェクト名、ステップ数、ノード総数 -->
    </header>

    <main>
        <div id="tree-container">
            <!-- D3.js SVGツリー描画領域 -->
        </div>
        <div id="detail-panel">
            <!-- 選択ノード詳細（JavaScript で動的更新） -->
        </div>
    </main>

    <section id="stats-panel">
        <!-- 統計サマリー: Chart.js or D3.jsで描画 -->
        <div id="status-chart"></div>      <!-- ステータス別棒グラフ -->
        <div id="operator-chart"></div>    <!-- オペレータ別円グラフ -->
        <div id="lcb-history"></div>       <!-- LCB推移折れ線グラフ -->
        <div id="depth-histogram"></div>   <!-- 深度分布ヒストグラム -->
    </section>

    <div id="code-modal" class="modal">
        <!-- 実験コード / stdout / stderr の表示モーダル -->
    </div>

    <script>
        // D3.js (v7) をインライン埋め込み（minified）
        // ※ ファイルサイズ制約により CDN fallback も可
    </script>
    <script>
        // ツリーデータ（Python側で生成してJSON埋め込み）
        const TREE_DATA = {tree_data_json};
        const STATS_DATA = {stats_data_json};
        const NODE_ARTIFACTS = {artifacts_json};

        // --- 定数 ---
        const STATUS_COLORS = {
            "pending":   "#E0E0E0",
            "running":   "#FFF176",
            "evaluated": "#81C784",
            "failed":    "#E57373",
            "timeout":   "#FFB74D",
            "oom":       "#CE93D8",
            "pruned":    "#BDBDBD",
            "expanded":  "#64B5F6"
        };

        const OP_SHAPES = {
            "draft":   "circle",
            "debug":   "triangle",
            "improve": "rect"
        };

        // --- ツリー描画 ---
        // D3.js tree layout
        // ノードクリック・ホバーイベント
        // パン・ズーム

        // --- 詳細パネル更新 ---
        function showNodeDetail(nodeId) { ... }

        // --- 統計チャート描画 ---
        // 棒グラフ・円グラフ・折れ線グラフ

        // --- モーダル ---
        function showCodeModal(nodeId, type) { ... }
    </script>
</body>
</html>
```

---

### 30.7 ライブモード（オプション）

`--live` オプション指定時、軽量HTTPサーバーを起動してリアルタイム更新を提供する。

```python
class LiveVisualizer:
    """ファイル監視 + WebSocket で自動更新するライブサーバー。"""

    def __init__(self, workspace_dir: Path, port: int = 8080):
        ...

    async def start(self):
        """
        1. checkpoints/ ディレクトリを watchdog で監視
        2. 新チェックポイント検出時に HTML を再生成
        3. WebSocket で接続中ブラウザに更新通知を送信
        4. ブラウザ側は通知受信時に差分データをfetchして再描画
        """
```

**依存パッケージ**: `watchdog`（ファイル監視）、`websockets` または `aiohttp`（WebSocket）。いずれも `[dev]` extras に追加。

**注意**: ライブモードは Phase 2（MVP後）の拡張機能とし、初期実装ではスタティックHTML生成のみを必須とする。

---

### 30.8 設計制約

| 制約 | 理由 |
|------|------|
| 単一HTML出力（self-contained） | ネットワーク不要環境（SLURM計算ノード等）で閲覧可能にするため |
| D3.js インライン埋め込み | 外部CDN依存を避ける。ただしファイルサイズが大きくなる場合はCDN fallback可 |
| Jinja2 不使用 | 依存を増やさない。f-string + `string.Template` で十分 |
| チェックポイントJSON直接読込 | 既存の `load_latest_checkpoint()` を再利用 |
| `runs/` 配下の読取は遅延 | ノード数が多い場合に全ファイル読込は重い。詳細パネル表示時にのみ埋め込む、またはノード選択時にJSで表示 |

---

### 30.9 ノード数が多い場合の対策

| ノード数 | 対策 |
|---------|------|
| ≤ 50 | 全ノード展開表示（デフォルト） |
| 51–200 | pruned ノードを初期状態で非表示（トグルで表示） |
| 201–500 | depth ≥ 3 のサブツリーを初期折りたたみ |
| > 500 | 上位50ノード（LCB順）のみ表示 + フィルタリングUI |

JavaScript側で `data-depth` / `data-status` 属性によるフィルタリングを実装する。

---

### 30.10 テスト仕様

#### 30.10.1 ユニットテスト

**ファイル**: `tests/test_visualization/`

```python
# test_tree_visualizer.py
class TestTreeVisualizer:
    def test_load_checkpoint_latest(self, tmp_workspace):
        """最新チェックポイントが正しく読み込まれること"""

    def test_load_checkpoint_specific_step(self, tmp_workspace):
        """指定stepのチェックポイントが読み込まれること"""

    def test_build_tree_data_single_root(self):
        """ルートノード1つのツリーデータが正しい階層構造になること"""

    def test_build_tree_data_multi_level(self):
        """3階層のツリーが正しく構築されること"""

    def test_build_tree_data_orphan_nodes(self):
        """parent_idが存在しないノードがルート直下に配置されること"""

    def test_collect_run_artifacts_exists(self, tmp_workspace):
        """runs/<node_id>/ にファイルがある場合に正しく収集されること"""

    def test_collect_run_artifacts_missing(self, tmp_workspace):
        """runs/<node_id>/ が存在しない場合にNoneが返ること"""

    def test_compute_stats(self):
        """ステータス別・オペレータ別の集計が正しいこと"""

    def test_compute_stats_empty_tree(self):
        """ノード0のとき空の統計が返ること"""


# test_node_formatter.py
class TestNodeFormatter:
    def test_format_node_evaluated(self):
        """評価済みノードのフォーマットが正しいこと"""

    def test_format_node_failed_with_error(self):
        """失敗ノードのエラーメッセージが含まれること"""

    def test_format_experiment_config_table(self):
        """experiment_configがテーブル形式に変換されること"""


# test_html_renderer.py
class TestHtmlRenderer:
    def test_generate_valid_html(self, tmp_workspace):
        """生成HTMLが構文的に正しいこと（html.parser でパース可能）"""

    def test_embedded_json_valid(self, tmp_workspace):
        """埋め込みJSONが有効であること"""

    def test_status_colors_complete(self):
        """全ステータスに対応する色が定義されていること"""

    def test_output_file_created(self, tmp_path):
        """指定パスにHTMLファイルが生成されること"""
```

#### 30.10.2 テスト用フィクスチャ

```python
@pytest.fixture
def sample_checkpoint():
    """テスト用の最小チェックポイントデータを返す。"""
    return {
        "step": 5,
        "all_nodes": {
            "root-001": {
                "node_id": "root-001",
                "parent_id": None,
                "depth": 0,
                "status": "evaluated",
                "branching_op": "draft",
                "hypothesis": "Baseline approach using default parameters",
                "experiment_config": {"learning_rate": 0.01, "method": "linear"},
                "mu": 0.85,
                "se": 0.02,
                "lcb": 0.81,
                "eval_runs": 3,
                "feasible": True,
                "priority": 0.81,
                "children_ids": ["child-001", "child-002"],
                "rationale": "Initial baseline",
                "total_cost": 0.1,
                "wall_time_sec": 5.0,
                "created_at": "2026-02-25T10:00:00Z",
                "adapter_node_id": None,
                "experiment_code": None,
                "debug_depth": 0,
                "error_message": None,
                "failure_context": [],
                "metrics_raw": [{"score": 0.84}, {"score": 0.85}, {"score": 0.86}],
                "tool_usage": {},
            },
            "child-001": {
                "node_id": "child-001",
                "parent_id": "root-001",
                "depth": 1,
                "status": "evaluated",
                "branching_op": "improve",
                "hypothesis": "Increase learning rate for faster convergence",
                "experiment_config": {"learning_rate": 0.05, "method": "linear"},
                "mu": 0.90,
                "se": 0.01,
                "lcb": 0.88,
                "eval_runs": 3,
                "feasible": True,
                "priority": 0.88,
                "children_ids": [],
                "rationale": "Higher LR may improve convergence",
                "total_cost": 0.1,
                "wall_time_sec": 4.8,
                "created_at": "2026-02-25T10:05:00Z",
                "adapter_node_id": None,
                "experiment_code": "import sklearn\\n...",
                "debug_depth": 0,
                "error_message": None,
                "failure_context": [],
                "metrics_raw": [{"score": 0.89}, {"score": 0.90}, {"score": 0.91}],
                "tool_usage": {},
            },
            "child-002": {
                "node_id": "child-002",
                "parent_id": "root-001",
                "depth": 1,
                "status": "failed",
                "branching_op": "debug",
                "hypothesis": "Fix runtime error in data preprocessing",
                "experiment_config": {"learning_rate": 0.01, "method": "svm"},
                "mu": None,
                "se": None,
                "lcb": None,
                "eval_runs": 0,
                "feasible": False,
                "priority": -float("inf"),
                "children_ids": [],
                "rationale": "Debug failed SVM experiment",
                "total_cost": 0.05,
                "wall_time_sec": 1.2,
                "created_at": "2026-02-25T10:03:00Z",
                "adapter_node_id": None,
                "experiment_code": "import sklearn\\n# buggy code",
                "debug_depth": 1,
                "error_message": "ValueError: could not convert string to float",
                "failure_context": [],
                "metrics_raw": [],
                "tool_usage": {},
            },
        },
        "open_list": [[-0.88, "child-001"]],
        "closed_set": ["root-001", "child-002"],
        "best_node_id": "child-001",
        "ppo_buffer": [],
    }
```

---

### 30.11 実装優先度

| 優先度 | 機能 | 備考 |
|--------|------|------|
| **P0（必須）** | チェックポイント読込 → ツリー構造構築 | `load_latest_checkpoint()` 再利用 |
| **P0（必須）** | ノードのステータス別色分け表示 | §30.4.2 配色テーブル |
| **P0（必須）** | ノードクリック → 詳細パネル表示 | §30.4.3 |
| **P0（必須）** | 統計サマリー（ノード数、ステータス集計、ベストノード） | §30.4.4 |
| **P0（必須）** | CLI `sera visualize` コマンド | §30.2 |
| **P1（推奨）** | オペレータ別ノード形状 | draft=丸、debug=三角、improve=四角 |
| **P1（推奨）** | パン・ズーム操作 | D3.js zoom behavior |
| **P1（推奨）** | 実験コード / stdout / stderr モーダル表示 | §30.4.3 下部ボタン |
| **P1（推奨）** | 性能推移グラフ（LCB折れ線） | §30.4.4 |
| **P1（推奨）** | 大規模ツリー対策（フィルタリング） | §30.9 |
| **P2（拡張）** | ライブモード（WebSocket自動更新） | §30.7 |
| **P2（拡張）** | 複数チェックポイント間の差分表示 | ステップ間のアニメーション遷移 |
| **P2（拡張）** | ノード検索・フィルタUI | hypothesis全文検索、config値フィルタ |

---

### 30.12 依存パッケージ

| パッケージ | 用途 | 追加先 |
|-----------|------|--------|
| なし（P0） | D3.js はHTMLインライン埋め込み | — |
| `watchdog` | ライブモードのファイル監視（P2） | `[dev]` extras |
| `aiohttp` | ライブモードのWebSocket（P2） | `[dev]` extras |

P0実装では**新規依存パッケージなし**。既存の `pathlib`, `json`, `html` 標準ライブラリのみ使用。
