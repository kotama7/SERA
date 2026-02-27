# LocalExecutor

ローカルサブプロセスとして実験スクリプトを実行する `Executor` 実装。タイムアウト処理、OOM 検知、多言語サポート、コンパイル言語ビルド、依存関係インストール、ストリーミング出力をサポートする。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `LocalExecutor` | `src/sera/execution/local_executor.py` |

## 依存関係

- `sera.execution.executor` (`Executor`, `RunResult`) -- 基底クラスと結果型
- `sera.execution.streaming` (`StreamEvent`, `StreamEventType`) -- ストリーミングイベント型
- `subprocess` / `asyncio` -- プロセス管理

---

## LocalExecutor

`Executor` ABC を実装し、ローカルサブプロセスで実験スクリプトを実行するクラス。

### コンストラクタ

```python
def __init__(
    self,
    work_dir: str | Path = "./sera_workspace",
    python_executable: str = "python",
    interpreter_command: str | None = None,
    seed_arg_format: str | None = None,
    allow_internet: bool = True,
    language_config: object | None = None,
)
```

- `work_dir`: ワークスペースディレクトリ。`runs/<node_id>/` がこの下に作成される
- `python_executable`: デフォルトインタープリタ
- `interpreter_command`: インタープリタのオーバーライド（例: `"Rscript"`, `"julia"`）。指定時は `python_executable` より優先
- `seed_arg_format`: シード引数フォーマット（デフォルト: `"--seed {seed}"`）
- `allow_internet`: ネットワークアクセスの許可。`False` の場合、プロキシ環境変数を除去
- `language_config`: `LanguageConfig` オブジェクト（コンパイル言語/依存管理用）

---

### run(node_id, script_path, seed, timeout_sec) -> RunResult

実験スクリプトをサブプロセスとして実行する。

**処理フロー:**

1. `runs/<node_id>/` ディレクトリと `artifacts/` サブディレクトリを作成
2. コンパイル言語の場合:
   - `_install_dependencies()` で依存関係をインストール
   - `_run_build_step()` でコンパイル
   - いずれか失敗時: `RunResult(success=False, build_time_sec, build_exit_code)` で早期リターン
3. インタープリタ/バイナリコマンドを構築:
   - コンパイル済み: `run_dir/<binary_name>` を直接実行
   - スクリプト: `interpreter script_path --seed <seed>`
   - シェル構文検出: `interpreter_command` に空白や `&&`, `|`, `;` が含まれる場合は `shell=True`
4. `subprocess.Popen` でプロセスを起動（cwd は `artifacts_dir`）
5. タイムアウト処理: `proc.wait(timeout=timeout_sec)` → 超過時 `proc.kill()` → `exit_code=-9`
6. OOM 検知:
   - exit_code が `137` または `-9` + stderr に OOM パターン → `exit_code=-7`
   - stderr に `MemoryError` / `OutOfMemoryError` → `exit_code=-7`
7. `metrics.json` を `artifacts_dir` → `run_dir` にコピー（必要な場合）

**終了コード:**

| exit_code | 意味 |
|-----------|------|
| `0` | 成功 |
| `-9` | タイムアウト |
| `-7` | OOM (SERA 固有センチネル) |
| `137` | Linux OOM killer (SIGKILL) |
| `127` | スクリプト/インタープリタ未発見 |
| `126` | OS エラー |

---

### run_stream(node_id, script_path, seed, timeout_sec) -> AsyncIterator[StreamEvent] [async]

行単位の非同期ストリーミング実行。

**処理フロー:**

1. シェル構文のインタープリタの場合: 基底クラスの `run_stream()` にフォールバック
2. `asyncio.create_subprocess_exec` で PIPE 付きプロセスを起動
3. 2 つの reader タスク（stdout / stderr）が `asyncio.Queue` にイベントをプッシュ
4. メインループ:
   - キューからイベントをドレイン → `yield`
   - タイムアウトチェック
   - ハートビート送出（5 秒間隔）
5. プロセス終了後: ログファイルに出力を書き込み、OOM 検知、終端イベントを送出

**ストリーミングイベント型:**

| `StreamEventType` | 説明 |
|-------------------|------|
| `STDOUT` | stdout の 1 行 |
| `STDERR` | stderr の 1 行 |
| `HEARTBEAT` | 長時間実行の生存確認（5 秒間隔） |
| `COMPLETED` | 正常完了 |
| `TIMEOUT` | タイムアウト |
| `ERROR` | エラー終了 |

**バッファ:** `deque(maxlen=1000)` でメモリ使用量を制限。終端イベントの `metadata` に末尾 50 行を含む。

---

## コンパイル言語サポート

### _install_dependencies(run_dir) -> tuple[bool, int, float]

依存関係のインストール。`language_config.dependency` から設定を読み取る。

**処理フロー:**

1. `pre_install_commands` の実行
2. インストールコマンドの決定（`install_command` または `manager` からの自動生成）
3. オフラインモード対応（`allow_internet=False` 時のキャッシュ使用）
4. インストール実行（タイムアウト: `install_timeout_sec`、デフォルト 300 秒）
5. `post_install_commands` の実行

**自動コマンド生成:**

| `manager` | コマンド |
|-----------|---------|
| `pip` | `pip install -r <build_file>` |
| `conda` | `conda install --file <build_file> -y` |
| `cargo` | （なし — cargo build が自動解決） |
| `go_mod` | `go mod download` |

### _run_build_step(run_dir, script_path) -> tuple[bool, int, float]

コンパイル言語のビルドステップ。

- `language_config.compiled=True` かつ `compile_command` が設定されている場合のみ実行
- `compile_flags`, `link_flags`, `binary_name` を使用してコマンドを構築
- `build_timeout_sec`（デフォルト: 120 秒）のタイムアウト

### _build_subprocess_env() -> dict[str, str] | None

サブプロセスの環境変数を構築する。`allow_internet=False` の場合、プロキシ環境変数（`http_proxy`, `https_proxy` 等）を除去する。
