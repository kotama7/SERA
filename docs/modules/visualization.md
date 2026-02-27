# TreeVisualizer / node_formatter / stats_calculator / html_renderer

探索木のインタラクティブ HTML 可視化モジュールのドキュメント。チェックポイントデータから D3.js ベースの可視化を生成する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `TreeVisualizer` | `src/sera/visualization/tree_visualizer.py` |
| `format_node` | `src/sera/visualization/node_formatter.py` |
| `compute_stats` | `src/sera/visualization/stats_calculator.py` |
| `render_html` | `src/sera/visualization/html_renderer.py` |
| `run_visualize` | `src/sera/commands/visualize_cmd.py` |

## 依存関係

- `json` -- チェックポイント/アーティファクトの読み書き
- `string.Template` -- HTML テンプレートエンジン（Jinja2 依存なし）
- `d3.js` v7 (CDN) -- インタラクティブツリー描画
- `sera.utils.checkpoint` (`load_latest_checkpoint`) -- 最新チェックポイントの検索
- `rich.console` -- CLI 出力フォーマット

---

## CLI コマンド

```bash
sera visualize [--work-dir PATH] [--step N] [--output PATH] [--live] [--port N]
```

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--work-dir` | `./sera_workspace` | SERA ワークスペースのパス |
| `--step` | `None`（最新） | 可視化するチェックポイントのステップ番号 |
| `--output` | `None`（自動） | 出力 HTML ファイルのパス |
| `--live` | `False` | チェックポイント変更を監視し自動更新する HTTP サーバーを起動 |
| `--port` | `8080` | ライブサーバーのポート番号 |

### run_visualize(work_dir, step=None, output=None, live=False, port=8080) -> None

`commands/visualize_cmd.py` の CLI ハンドラ。

**処理フロー:**

1. `TreeVisualizer(workspace)` を初期化
2. `visualizer.generate_html(step, output_path)` で HTML を生成
3. 成功時: 生成パスを表示
4. `FileNotFoundError`: チェックポイントが見つからない旨を表示し `exit(1)`
5. その他の例外: エラーメッセージを表示し `exit(1)`

### ライブサーバー機能

`live=True` の場合、`_run_live_server()` を呼び出し以下の動作を行う:

- ポーリングベースのファイルウォッチャー（5 秒間隔）でチェックポイント変更を検知
- 変更検知時に HTML を再生成
- HTML に `<meta http-equiv="refresh" content="10">` を注入してブラウザを自動リロード
- `http.server` を使用した HTTP サーバーを `port` で起動
- 外部依存なし（stdlib のみ）

---

## TreeVisualizer

探索木の HTML 可視化を生成するメインクラス。

### コンストラクタ

```python
def __init__(self, workspace_dir: Path) -> None
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `workspace_dir` | `Path` | `sera_workspace/` ディレクトリ |
| `checkpoint_dir` | `Path` | `sera_workspace/checkpoints/` |
| `runs_dir` | `Path` | `sera_workspace/runs/` |

---

### load_checkpoint(step=None) -> dict

チェックポイントをディスクからロードする。

| パラメータ | 動作 |
|-----------|------|
| `step` が指定 | `checkpoints/search_state_step_{step}.json` を読み込み。存在しない場合は `FileNotFoundError` |
| `step=None` | `load_latest_checkpoint()` で最新のチェックポイントを検索。見つからない場合は `FileNotFoundError` |

---

### build_tree_data(checkpoint) -> dict

チェックポイントデータから D3.js 階層構造を構築する。

**処理フロー:**

1. `checkpoint["all_nodes"]` から全ノードを取得
2. 各ノードを `format_node()` で表示用 dict に変換
3. `parent_id` に基づいて親子マッピングを構築
4. ルートノード（`parent_id=None`）を特定
5. 孤立ノード（親が `all_nodes` に存在しない）もルートとして追加
6. 再帰的にサブツリーを構築（循環参照防止のため `visited` セットを使用）

**出力形式:**

```python
{
    "id": "root",
    "data": {"node_id": "root", "status": "expanded", "branching_op": "draft"},
    "children": [
        {
            "id": "node-abc123",
            "data": {format_node() の出力},
            "children": [...]
        },
        ...
    ]
}
```

仮想ルートノード `"root"` が全ルートノードの親として配置される。

---

### collect_run_artifacts(node_id) -> dict

`runs/<node_id>/` から実験アーティファクトを収集する。

**収集対象:**

| キー | ファイル | 最大サイズ |
|------|---------|----------|
| `experiment_code` | `experiment.py` | 10,000 文字 |
| `stdout` | `stdout.log` | 10,000 文字 |
| `stderr` | `stderr.log` | 10,000 文字 |
| `metrics` | `metrics.json` | JSON 全体 |

ディレクトリまたはファイルが存在しない場合は対応するキーが `None` となる。読み込みエラーは無視される。

---

### compute_stats(checkpoint) -> dict

サマリー統計を計算する。内部で `stats_calculator.compute_stats()` に委譲する。

---

### generate_html(step=None, output_path=None) -> Path

完全な HTML 可視化を生成するメインメソッド。

**処理フロー:**

1. `load_checkpoint(step)` でチェックポイントをロード
2. `build_tree_data(checkpoint)` でツリーデータを構築
3. `compute_stats(checkpoint)` で統計データを計算
4. 全ノードのアーティファクトを収集:
   - ノード数 > 200 の場合: best ノード、evaluated/failed/oom/timeout ノードを優先的にロード（最大 200 件）
   - 実際のコンテンツが存在するノードのみ結果に含める
5. `render_html()` で HTML ファイルを生成

**デフォルト出力パス:** `sera_workspace/outputs/tree_visualization.html`

---

## format_node (node_formatter.py)

`SearchNode.to_dict()` の出力を表示用 dict に変換するモジュールレベル関数。

```python
def format_node(node_data: dict) -> dict
```

**変換内容:**

- `mu`, `se`, `lcb` を `round(..., 4)` で小数点以下 4 桁に丸める。`None` の場合はそのまま
- 全フィールドを dict として返す（未知のフィールドは無視）

**出力フィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `node_id` | `str` | ノード ID |
| `parent_id` | `str \| None` | 親ノード ID |
| `depth` | `int` | 木の深さ |
| `status` | `str` | ノード状態 |
| `branching_op` | `str` | 生成オペレータ |
| `hypothesis` | `str` | 仮説 |
| `rationale` | `str` | アプローチの理由 |
| `experiment_config` | `dict` | 実験設定 |
| `mu` | `float \| None` | 平均値（丸め済み） |
| `se` | `float \| None` | 標準誤差（丸め済み） |
| `lcb` | `float \| None` | 下側信頼限界（丸め済み） |
| `eval_runs` | `int` | 評価実行回数 |
| `feasible` | `bool` | 実行可能性 |
| `priority` | `float \| None` | 探索優先度 |
| `children_ids` | `list[str]` | 子ノード ID |
| `total_cost` | `float` | 累積コスト |
| `wall_time_sec` | `float` | 累積実行時間 |
| `created_at` | `str` | 作成日時 |
| `adapter_node_id` | `str \| None` | リネージツリーへのリンク |
| `debug_depth` | `int` | デバッグ深度 |
| `error_message` | `str \| None` | エラーメッセージ |
| `failure_context` | `list[dict]` | ECHO 失敗知識 |

### format_experiment_config_table(config) -> str

`experiment_config` を HTML テーブル文字列に変換するヘルパー関数。空の場合は `<em>No configuration</em>` を返す。

---

## compute_stats (stats_calculator.py)

チェックポイントデータからサマリー統計を計算するモジュールレベル関数。

```python
def compute_stats(checkpoint: dict) -> dict
```

**計算項目:**

| キー | 型 | 説明 |
|------|-----|------|
| `step` | `int` | 現在のステップ番号 |
| `total_nodes` | `int` | 全ノード数 |
| `status_counts` | `dict[str, int]` | ステータス別ノード数 |
| `operator_counts` | `dict[str, int]` | オペレータ別ノード数 |
| `best_node` | `dict` | 最良ノード情報（`node_id`, `mu`, `se`, `lcb`, `hypothesis[:100]`） |
| `depth_distribution` | `dict[int, int]` | 深さ別ノード数（深さでソート済み） |
| `success_rate` | `float` | 成功率（小数点以下 4 桁に丸め） |
| `best_lcb_history` | `list[dict]` | 最良 LCB の推移 |

**成功率の計算:**

```
n_evaluated = evaluated + expanded
n_terminal_fail = failed + timeout + oom
total_terminal = n_evaluated + n_terminal_fail
success_rate = n_evaluated / total_terminal    # total_terminal=0 の場合は 0.0
```

**最良 LCB 推移:**

LCB が非 None のノードを深さでソートし、累積最良 LCB を追跡する。

```python
[{"step": 1, "lcb": 0.45}, {"step": 2, "lcb": 0.52}, ...]
```

**dict / オブジェクトの二重対応:** 全フィールドアクセスは `isinstance(node, dict)` でチェックし、dict の場合は `.get()`、オブジェクトの場合は `getattr()` を使用する。

---

## render_html (html_renderer.py)

自己完結型の HTML 可視化ファイルを生成するモジュールレベル関数。

```python
def render_html(
    tree_data: dict,
    stats_data: dict,
    node_artifacts: dict[str, dict],
    step: int,
    output_path: Path,
) -> Path
```

**特徴:**

- Jinja2 不要: `string.Template` (`safe_substitute`) を使用
- D3.js v7 を CDN からロード
- CSS と JavaScript を全てインライン化した自己完結型 HTML
- 出力ディレクトリは自動作成（`parents=True`）

### HTML テンプレートの構成

**ヘッダ:** ステップ番号とノード数を表示

**メインエリア（2 カラム）:**

| 要素 | 説明 |
|------|------|
| `#tree-container` (左) | D3.js によるインタラクティブツリー描画。ズーム/パン対応 |
| `#detail-panel` (右, 380px) | クリックしたノードの詳細表示 |

**統計パネル（下部）:**

| カード | 表示内容 |
|--------|---------|
| Search Step | 現在のステップ番号 |
| Total Nodes | ノード数と成功率 |
| Best Node | 最良ノードの ID, LCB, mu +/- SE |
| Status Distribution | ステータス別の棒グラフ |
| Operator Distribution | オペレータ別の円グラフ（pie chart） |
| Depth Distribution | 深さ別の棒グラフ |
| Best LCB History | LCB 推移の折れ線グラフ（SVG） |

**モーダル:** ノードの実験コード、stdout、stderr をモーダルウィンドウで表示

### ノードの視覚的表現

**色（ステータス別）:**

| ステータス | 色 |
|-----------|-----|
| `pending` | `#E0E0E0`（グレー） |
| `running` | `#FFF176`（黄） |
| `evaluated` | `#81C784`（緑） |
| `failed` | `#E57373`（赤） |
| `timeout` | `#FFB74D`（オレンジ） |
| `oom` | `#CE93D8`（紫） |
| `pruned` | `#BDBDBD`（ライトグレー） |
| `expanded` | `#64B5F6`（青） |

**形（オペレータ別）:**

| オペレータ | 形 |
|-----------|-----|
| `draft` | 円 (circle) |
| `debug` | 三角形 (triangle) |
| `improve` | 矩形 (rect) |

**サイズ:** LCB 値に基づいて動的に変化（`max(8, min(24, 12 + lcb * 2))`）

**特殊表示:**

| 条件 | スタイル |
|------|---------|
| 最良ノード | 金色のストローク（`#FFD700`、太さ 3px） |
| infeasible ノード | 破線ストローク（`#E57373`） |

**リンク線:**

| オペレータ | 線のスタイル |
|-----------|------------|
| `debug` | 赤の破線（`#E57373`, dasharray `6,3`） |
| `improve` | 青の実線（`#64B5F6`） |
| `draft` | グレーの実線（`#666`） |

### インタラクション

- **ズーム/パン:** D3.js の `d3.zoom()` で対応。スケール範囲 `[0.1, 4]`
- **オートスケール:** ノード数 > 50 の場合、`max(0.3, 50/nodeCount)` で自動縮小
- **ツールチップ:** ノードホバー時にステータス、仮説（先頭 50 文字）、mu +/- SE を表示
- **ノードクリック:** 右パネルに詳細情報を表示（仮説全文、experiment_config テーブル、メトリクス、子ノードリスト、エラーメッセージ、ECHO 失敗コンテキスト）
- **コード/ログ表示:** ノード詳細の Code/stdout/stderr ボタンでモーダル表示

---

## ディレクトリ構成

```
src/sera/visualization/
  __init__.py              # TreeVisualizer をエクスポート
  tree_visualizer.py       # メインクラス（チェックポイント読み込み、ツリー構築、HTML 生成）
  node_formatter.py        # SearchNode dict の表示用フォーマット
  stats_calculator.py      # サマリー統計の計算
  html_renderer.py         # HTML テンプレートとファイル出力

src/sera/commands/
  visualize_cmd.py         # CLI コマンドハンドラ
```

---

## 出力ファイル

| ファイル | デフォルトパス | 内容 |
|---------|-------------|------|
| HTML 可視化 | `sera_workspace/outputs/tree_visualization.html` | インタラクティブな探索木可視化。ブラウザで開いて使用 |
