# DockerExecutor

Docker コンテナ内で実験スクリプトを実行する Executor 実装。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `DockerExecutor` | `src/sera/execution/docker_executor.py` |

## 依存関係

- `docker` -- Docker Python SDK（`pip install -e ".[docker]"`）
- `sera.execution.base` (`Executor`, `RunResult`) -- 基底クラスと結果型

---

## DockerExecutor

`Executor` ABC を実装し、Docker コンテナ内で実験スクリプトを実行するクラス。GPU パススルー、タイムアウト処理、OOM 検知をサポートする。

### コンストラクタ

```python
def __init__(
    self,
    work_dir: str | Path = "./sera_workspace",
    docker_config: object = None,
    interpreter_command: str | None = None,
    seed_arg_format: str | None = None,
    gpu_enabled: bool = True,
)
```

- `work_dir`: ワークスペースディレクトリ。`runs/<node_id>/` がこの下に作成される
- `docker_config`: Docker 設定オブジェクト（`image`, `volumes`, `env_vars`, `gpu_runtime` 等）
- `interpreter_command`: 実験スクリプトのインタープリタ（デフォルト: `"python"`）。多言語サポート用
- `seed_arg_format`: シード引数のフォーマット（デフォルト: `"--seed {seed}"`）
- `gpu_enabled`: GPU パススルーの有効化（デフォルト: `True`）

### run(node_id, script_path, seed, timeout_sec) -> RunResult

Docker コンテナ内で実験スクリプトを実行する。

**処理フロー:**

1. `runs/<node_id>/` ディレクトリを作成
2. スクリプトを run ディレクトリにコピー（必要な場合）
3. Docker イメージの確認・プル（`_ensure_image`）
4. コンテナを作成・起動:
   - ボリュームマウント: run ディレクトリ → `/workspace`
   - GPU 設定: `DeviceRequest(count=-1, capabilities=[["gpu"]])` または `runtime` パラメータ
   - 追加ボリューム: `docker_config.volumes` から `host:container[:mode]` 形式でパース
5. コンテナ終了を待機（タイムアウト付き）
6. ログ取得（`_capture_logs`）
7. 結果判定:
   - exit_code=0: `metrics.json` を読み取り → 成功
   - タイムアウト: exit_code=-9
   - OOM 検知: exit_code=-7
   - その他: 失敗として記録

**戻り値:** `RunResult(node_id, success, exit_code, stdout_path, stderr_path, metrics_path, artifacts_dir, wall_time_sec, seed)`

### 多言語サポート

`interpreter_command` と `seed_arg_format` を指定することで Python 以外の言語をサポート:

| 言語 | interpreter_command | seed_arg_format |
|------|-------------------|----------------|
| Python | `"python"` | `"--seed {seed}"` |
| R | `"Rscript"` | `"--seed {seed}"` |
| Julia | `"julia"` | `"--seed {seed}"` |

### GPU パススルー

`gpu_enabled=True` の場合、以下の順序で GPU アクセスを設定:

1. **DeviceRequest** 方式（推奨）: `docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])`
2. **runtime** 方式（フォールバック）: `runtime="nvidia"` パラメータ

### OOM 検知 (_detect_oom)

以下の条件のいずれかで OOM と判定:

1. コンテナ状態の `OOMKilled` フラグが `True`
2. exit_code が `137`（128 + SIGKILL = Docker OOM exit code）
3. stderr に OOM パターンが含まれる

**OOM stderr パターン:**

```python
_OOM_STDERR_PATTERNS = (
    "MemoryError",
    "OutOfMemoryError",
    "Killed",
    "Cannot allocate memory",
    "OOMKilled",
)
```

### 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `_OOM_STDERR_PATTERNS` | 上記参照 | OOM 検知用 stderr パターン |
| `_DOCKER_OOM_EXIT_CODE` | `137` | Docker OOM exit code |

### エラーハンドリング

| 状況 | 動作 |
|------|------|
| Docker イメージ不在 | 自動プル（`_ensure_image`） |
| タイムアウト | コンテナを graceful stop（10 秒待機）→ force kill |
| OOM | exit_code=-7 として RunResult を返す |
| docker パッケージ未インストール | `import docker` は `try/except` でラップされ、失敗時は `_DOCKER_AVAILABLE = False` に設定される。`__init__` で `_DOCKER_AVAILABLE` を確認し、`False` の場合は `ImportError` を送出（`"DockerExecutor requires the 'docker' Python SDK. Install it with: pip install docker"` メッセージ付き） |
| コンテナ起動失敗 | 例外をそのまま伝搬 |

### ヘルパー関数

| 関数 | 説明 |
|------|------|
| `_ensure_image(client, image)` | Docker イメージをローカルに確認し、なければプル |
| `_capture_logs(container, stdout_path, stderr_path)` | コンテナログをファイルに保存 |
| `_detect_oom(container, exit_code, stderr_path)` | OOM 判定（上記参照） |
| `_is_timeout_error(exc)` | 例外がタイムアウト系かどうかを判定 |
| `_stop_container(container)` | graceful stop (10s) → force kill |
| `_remove_container(container)` | コンテナを削除 |
