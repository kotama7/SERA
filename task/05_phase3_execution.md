# SERA 要件定義書 — Phase 3: 実験実行

> 本ファイルは TASK.md v13.2 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 7. Phase 3：実験実行（Executor）

### 7.1 実行器の要件（必須）
- ローカル実行（MVP必須）
- SLURM実行（ResourceSpecで利用可能なら切替）
- コンテナ実行（ResourceSpecで必須なら、指定イメージで実行）

### 7.2 実験コード生成（具体）

```python
@dataclass
class GeneratedFile:
    """LLM が生成した単一ファイルを表す"""
    relative_path: str       # runs/{node_id}/ からの相対パス（例: "experiment.py", "src/utils.py"）
    content: str             # ファイル内容

@dataclass
class GeneratedExperiment:
    """LLM が生成した実験プロジェクト一式を表す"""
    entry_point: str         # メインスクリプトの相対パス（例: "experiment.py", "src/main.rs"）
    files: list[GeneratedFile]  # 生成された全ファイル（entry_point を含む）

class ExperimentGenerator:
    """探索ノードの実験条件から実行可能な実験コードを生成する"""

    def generate(self, node: SearchNode, problem_spec: ProblemSpec, agent_llm: AgentLLM) -> GeneratedExperiment:
        """
        1. ProblemSpec.experiment_template をベースに、
           node.experiment_config の値で変数を埋める
        2. テンプレートが不十分な場合、LLM に以下を生成させる：
           - データ読み込み
           - 前処理
           - モデル構築/学習
           - 評価
           - metrics.json 出力
        3. 生成コードは必ず以下を満たす：
           - metrics.json を stdout ではなくファイルに出力
           - seed を受け取り np.random.seed / torch.manual_seed を設定
           - 例外時は stderr にトレースバックを出力し exit(1)
        4. 複数ファイルプロジェクトの場合（§7.2.1 参照）：
           - LLM にエントリポイント + 補助ファイルを一括生成させる
           - 全ファイルを runs/{node_id}/ 配下に書き出す
           - entry_point がメインスクリプトのパスを保持
        """
        pass
```

#### 7.2.1 複数ファイルプロジェクトサポート

実験が複雑化すると、単一の `experiment.py` では管理が困難になる。以下のケースでは LLM が複数ファイルを生成する：

- **Python**: ヘルパーモジュール（`utils.py`, `model.py`）を `experiment.py` から import
- **C++**: ヘッダ（`.h`）+ ソース（`.cpp`）の分割コンパイル
- **Rust**: `src/main.rs` + `src/lib.rs` のクレート構成
- **Go**: 複数 `.go` ファイルのパッケージ構成

**ExperimentGenerator の複数ファイル生成フロー**:

```text
ExperimentGenerator.generate(node, problem_spec, agent_llm)
  │
  ├─ language.multi_file == False の場合:
  │   └─ 単一ファイル生成（後方互換モード）
  │      GeneratedExperiment(
  │        entry_point="experiment.{ext}",
  │        files=[GeneratedFile("experiment.{ext}", code)]
  │      )
  │
  └─ language.multi_file == True の場合（デフォルト）:
      │
      ├─ 1. LLM に複数ファイル構成を依頼
      │     プロンプト:
      │       「以下の実験を実装せよ。複数ファイルに分割してよい。
      │        各ファイルを以下の JSON 形式で出力せよ:
      │        {"entry_point": "experiment.py",
      │         "files": [
      │           {"path": "experiment.py", "content": "..."},
      │           {"path": "utils.py", "content": "..."}
      │         ]}」
      │
      ├─ 2. JSON パース + バリデーション
      │     - entry_point が files 内に存在すること
      │     - パストラバーサル禁止（".." を含むパスは拒否）
      │     - ファイル数上限: max_files（デフォルト 10）
      │     - 合計サイズ上限: max_total_size_bytes（デフォルト 1MB）
      │
      ├─ 3. 全ファイルを runs/{node_id}/ 配下に書き出し
      │     サブディレクトリが必要な場合は自動作成（例: src/）
      │
      └─ 4. GeneratedExperiment を返却
```

**LanguageConfig 追加フィールド**:

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `multi_file` | `bool` | `True` | 複数ファイル生成を許可。`False` の場合は従来の単一ファイル |
| `max_files` | `int` | `10` | 生成ファイル数の上限 |
| `max_total_size_bytes` | `int` | `1048576` | 全ファイル合計サイズの上限（1MB） |

**言語別ディレクトリ構成例**:

```text
# Python（multi_file=True）
runs/<node_id>/
  ├─ experiment.py          # entry_point（import utils, model）
  ├─ utils.py               # ヘルパー関数
  ├─ model.py               # モデル定義
  └─ metrics.json

# C++（multi_file=True, compiled=True）
runs/<node_id>/
  ├─ main.cpp               # entry_point
  ├─ solver.h               # ヘッダ
  ├─ solver.cpp             # 実装
  ├─ CMakeLists.txt         # ビルドファイル（§7.3.3）
  └─ metrics.json

# Rust（multi_file=True, compiled=True）
runs/<node_id>/
  ├─ src/
  │   ├─ main.rs            # entry_point
  │   └─ lib.rs             # ライブラリ
  ├─ Cargo.toml             # ビルドファイル（§7.3.3）
  └─ metrics.json
```

**Executor への影響**:

`Executor.run()` に渡す `script_path` は `GeneratedExperiment.entry_point` を指す。`cwd` は `runs/{node_id}/` に設定されるため、相対 import / 相対 include が機能する：

```python
# LocalExecutor.run() の変更点
# 旧: cmd = [interpreter, script_path, seed_arg]
# 新: script_path = GeneratedExperiment.entry_point（runs/{node_id}/ からの相対パス）
cmd = [interpreter, str(run_dir / entry_point), seed_arg]
# cwd = run_dir（変更なし）
```

コンパイル型言語の場合、§7.3.2 のビルドステップで全ソースファイルがコンパイルされる：
- **C++ (g++)**: `g++ -O2 main.cpp solver.cpp -o experiment`（compile_flags で指定、またはCMakeLists.txtが管理）
- **Rust (cargo)**: `cargo build --release`（Cargo.toml が全クレート構成を管理）
- **Go**: `go build -o experiment ./...`（パッケージ内の全 .go をビルド）

**三層可変性モデルとの整合**: `multi_file`, `max_files`, `max_total_size_bytes` は全て Frozen 層（Phase 1 で固定）。生成されるファイルの**内容**と**構成**は Manipulated 層（ノード毎に LLM が決定）。

**テスト計画**:

| テストケース | 概要 |
|-------------|------|
| `test_generated_experiment_single_file` | `multi_file=False` で従来通り単一ファイルが生成されること |
| `test_generated_experiment_multi_file_python` | Python で複数ファイル（experiment.py + utils.py）が生成・書き出されること |
| `test_generated_experiment_multi_file_cpp` | C++ で main.cpp + solver.h + solver.cpp が生成されること |
| `test_generated_experiment_entry_point_validation` | entry_point が files 内に存在しない場合にエラー |
| `test_generated_experiment_path_traversal_blocked` | `../` を含むパスが拒否されること |
| `test_generated_experiment_max_files_exceeded` | ファイル数上限超過でエラー |
| `test_generated_experiment_max_size_exceeded` | 合計サイズ上限超過でエラー |
| `test_executor_multi_file_python_import` | 生成した複数 Python ファイル間の import が実行時に機能すること |
| `test_executor_multi_file_cpp_compile` | 複数 C++ ファイルのコンパイル + リンクが成功すること |

### 7.3 実行インターフェース（プラグイン抽象クラス）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    node_id: str
    success: bool
    exit_code: int
    stdout_path: Path        # runs/<node_id>/stdout.log
    stderr_path: Path        # runs/<node_id>/stderr.log
    metrics_path: Path | None  # runs/<node_id>/metrics.json（成功時のみ）
    artifacts_dir: Path      # runs/<node_id>/artifacts/
    wall_time_sec: float
    seed: int

class Executor(ABC):
    @abstractmethod
    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int) -> RunResult:
        """
        実験スクリプトを実行し、結果を返す。

        - 実行前: runs/<node_id>/ および runs/<node_id>/artifacts/ ディレクトリを作成
        - cwd を artifacts/ に設定（実験スクリプトの出力を隔離）
        - stdout/stderr をファイルにリダイレクト
        - タイムアウト超過は RunResult(success=False, exit_code=-9) を返す
        - OOM は RunResult(success=False, exit_code=-7) を返す
        """
        pass

class LocalExecutor(Executor):
    """subprocess.Popen でローカル実行。
    ProblemSpec.language の設定に応じて interpreter / seed 引数を切り替える。"""
    def __init__(
        self,
        work_dir: Path,
        python_executable: str = "python",
        interpreter_command: str | None = None,   # 例: "Rscript", "julia"
        seed_arg_format: str | None = None,        # 例: "-- --seed {seed}"
        allow_internet: bool = True,               # False → プロキシ環境変数を除去
    ): ...

class SlurmExecutor(Executor):
    """sbatch でジョブ投入、sacct で完了待ち"""
    pass

class DockerExecutor(Executor):
    """docker run で隔離実行"""
    pass
```

#### 7.3.1 多言語サポート（LanguageConfig）

`ProblemSpec.language` に `LanguageConfig` を設定すると、Python 以外の言語で実験を実行できる。Agent 実行時は Agent が言語選択・インタプリタ指定を動的に行うため、本設定は主に `StatisticalEvaluator` が `executor.run()` で反復実行する際に使用される。

```yaml
# ProblemSpec.language — デフォルト（未指定時）は Python
language:
  name: "python"                    # プロンプト生成に使用
  interpreter_command: "python"     # LocalExecutor.interpreter_command に渡る
  file_extension: ".py"             # 実験スクリプトの拡張子
  seed_arg_format: "--seed {seed}"  # LocalExecutor.seed_arg_format に渡る
  code_block_tag: "python"          # Markdown コードブロックタグ
```

**R の例**: `interpreter_command: "Rscript"`, `file_extension: ".R"`, `code_block_tag: "r"`
**Julia の例**: `interpreter_command: "julia"`, `file_extension: ".jl"`, `seed_arg_format: "-- --seed {seed}"`

`metrics.json` の出力スキーマ（§7.4）は言語に依存しない。全言語で同一の契約に準拠する。

#### 7.3.2 コンパイル型言語サポート（C++, Rust, Go）

§7.3.1 の `LanguageConfig` はインタプリタ型言語（Python, R, Julia）を想定している。C++, Rust, Go 等のコンパイル型言語では、実行前にビルドステップが必要となる。本節では `LanguageConfig` を拡張し、コンパイル・リンクフローを定義する。

##### 7.3.2.1 LanguageConfig 拡張フィールド

既存の `LanguageConfig` に以下のフィールドを追加する。全フィールドにデフォルト値があり、既存の YAML に影響しない：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `compiled` | `bool` | `False` | コンパイル型言語フラグ。`True` の場合ビルドステップを実行 |
| `compile_command` | `str` | `""` | コンパイラコマンド（例: `"g++"`, `"cargo build --release"`, `"go build"`） |
| `compile_flags` | `list[str]` | `[]` | コンパイルフラグ（例: `["-O2", "-std=c++17"]`） |
| `link_flags` | `list[str]` | `[]` | リンクフラグ（例: `["-lm", "-lpthread"]`） |
| `binary_name` | `str` | `"experiment"` | 出力バイナリ名 |
| `build_timeout_sec` | `int` | `120` | ビルドステップのタイムアウト（秒） |

##### 7.3.2.2 言語別 YAML 設定例

**C++ の例**:

```yaml
language:
  name: "cpp"
  compiled: true
  compile_command: "g++"
  compile_flags: ["-O2", "-std=c++17", "-Wall"]
  link_flags: ["-lm", "-lpthread"]
  binary_name: "experiment"
  build_timeout_sec: 120
  file_extension: ".cpp"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "cpp"
```

**Rust の例**:

```yaml
language:
  name: "rust"
  compiled: true
  compile_command: "cargo build --release"
  compile_flags: []    # Cargo が管理するため不要
  link_flags: []
  binary_name: "target/release/experiment"
  build_timeout_sec: 300
  file_extension: ".rs"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "rust"
```

**Go の例**:

```yaml
language:
  name: "go"
  compiled: true
  compile_command: "go build"
  compile_flags: ["-o", "experiment"]
  link_flags: []
  binary_name: "experiment"
  build_timeout_sec: 120
  file_extension: ".go"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "go"
```

##### 7.3.2.3 ビルドステップ実行フロー

`Executor.run()` の先頭にビルドステップを挿入する。既存のインタプリタ型パスは一切変更しない：

```text
Executor.run(node_id, script_path, seed, timeout_sec)
  │
  ├─ language.compiled == True の場合:
  │   │
  │   ├─ 1. 依存関係インストール（§7.3.3 DependencyConfig が存在する場合）
  │   │
  │   ├─ 2. コンパイル
  │   │     cwd = runs/{node_id}/
  │   │     cmd = [compile_command] + compile_flags + [script_path] + link_flags + ["-o", binary_name]
  │   │     ※ compile_command が複数語の場合（例: "cargo build --release"）は shell=True で実行
  │   │     timeout = build_timeout_sec
  │   │     stdout → runs/{node_id}/build_stdout.log
  │   │     stderr → runs/{node_id}/build_stderr.log
  │   │
  │   ├─ 3. ビルド失敗判定
  │   │     exit_code != 0 → RunResult(success=False, build_exit_code=exit_code, exit_code=exit_code)
  │   │     node.status = "failed"（ビルドエラーは実行エラーと同様に扱う）
  │   │
  │   └─ 4. バイナリ実行
  │         cmd = [runs/{node_id}/{binary_name}, seed_arg_format.format(seed=seed)]
  │         以降は既存の実行フローと同一（stdout/stderr リダイレクト、タイムアウト、OOM 検出）
  │
  └─ language.compiled == False の場合:
      └─ 既存のインタプリタ実行パス（変更なし）
```

##### 7.3.2.4 Executor 別の対応方針

| Executor | ビルド場所 | 備考 |
|----------|-----------|------|
| `LocalExecutor` | ローカルホスト上 | コンパイラがローカルにインストールされていること |
| `SlurmExecutor` | SLURM 計算ノード上 | `_run_experiment()` 内でビルド + 実行を一括実行。`modules` でコンパイラモジュールをロード |
| `DockerExecutor` | コンテナ内 | コンテナイメージにコンパイラが含まれていること |

`SlurmExecutor` の場合、`_run_experiment()` の処理フローが以下のように拡張される：

```python
def _run_experiment(interpreter_command, script_path, seed, run_dir, modules, seed_arg_format, language_config):
    # 1. 環境モジュールロード: module load <mod>
    # 2. コンパイル型の場合:
    #    a. 依存関係インストール（DependencyConfig あれば）
    #    b. コンパイル実行 → build_stdout.log / build_stderr.log
    #    c. ビルド失敗 → exit code 返却（実験は実行しない）
    #    d. バイナリ実行: [binary_name, seed_arg_format.format(seed=seed)]
    # 3. インタプリタ型の場合: 既存パス
```

##### 7.3.2.5 RunResult 拡張

ビルド情報を格納するため `RunResult` に以下のフィールドを追加する。既存フィールドは一切変更しない：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `build_time_sec` | `float \| None` | `None` | ビルドに要した時間（秒）。インタプリタ型は `None` |
| `build_exit_code` | `int \| None` | `None` | ビルドの終了コード。インタプリタ型は `None` |

ビルド失敗時の `RunResult`:
- `success = False`
- `exit_code` = ビルドプロセスの終了コード
- `build_exit_code` = 同上
- `metrics_path = None`（実験未実行のため）

##### 7.3.2.6 三層可変性モデルとの整合

| フィールド | 層 | Phase 1 以降 | 備考 |
|-----------|-----|-------------|------|
| `compiled` | Frozen | 不変 | 言語選択は研究開始前に決定 |
| `compile_command` | Frozen | 不変 | コンパイラは固定 |
| `compile_flags` | Frozen | 不変 | 最適化レベル等はユーザーが事前決定 |
| `link_flags` | Frozen | 不変 | リンク対象ライブラリは固定 |
| `binary_name` | Frozen | 不変 | |
| `build_timeout_sec` | Frozen | 不変 | |

全フィールドが Frozen 層に属する。実験スクリプト（`.cpp`, `.rs`, `.go`）自体は LLM が生成する Manipulated 層のアーティファクトであるが、コンパイラ設定は Frozen である。

##### 7.3.2.7 ToolPolicy 拡張

`ToolPolicy` のシェルコマンドホワイトリストに以下のビルドツールを追加する：

```python
# 既存: ["pip", "python", "ls", "cat", "wc"]
# 追加:
ALLOWED_BUILD_COMMANDS = [
    "g++", "gcc", "clang++", "clang",   # C/C++ コンパイラ
    "cargo", "rustc",                     # Rust
    "go",                                 # Go
    "make", "cmake",                      # ビルドシステム
]
```

ビルドコマンドは `language.compiled == True` の場合にのみ許可される。インタプリタ型言語の実行時にはこれらは引き続きブロックされる。

##### 7.3.2.8 テスト計画

| テストケース | 概要 |
|-------------|------|
| `test_language_config_compiled_defaults` | `compiled=False` のデフォルト値で既存動作に影響なし |
| `test_language_config_cpp` | C++ 設定の YAML パース・バリデーション |
| `test_language_config_rust` | Rust 設定の YAML パース・バリデーション |
| `test_language_config_go` | Go 設定の YAML パース・バリデーション |
| `test_local_executor_build_success` | ビルド成功 → バイナリ実行 → RunResult に `build_time_sec` 格納 |
| `test_local_executor_build_failure` | ビルド失敗 → `success=False`, `build_exit_code != 0`, 実験未実行 |
| `test_local_executor_build_timeout` | ビルドタイムアウト → `success=False`, `build_exit_code=-9` |
| `test_run_result_build_fields` | `build_time_sec`, `build_exit_code` のデフォルト `None` 検証 |
| `test_tool_policy_build_commands` | `compiled=True` 時にビルドコマンドが許可される |
| `test_tool_policy_build_commands_blocked` | `compiled=False` 時にビルドコマンドがブロックされる |

#### 7.3.3 動的依存関係管理と LLM 生成ビルドオプション

実験スクリプトが外部ライブラリに依存する場合、実行前に依存関係のインストールが必要となる。本節では `DependencyConfig` を定義し、言語別の依存管理フローを規定する。また、LLM がビルドファイル（`requirements.txt`, `CMakeLists.txt` 等）を動的に生成するフローを定義する。

##### 7.3.3.1 DependencyConfig 定義

`LanguageConfig` 内にネストする。未設定時は依存管理をスキップする：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `manager` | `str` | `"pip"` | 依存管理ツール（`"pip"`, `"conda"`, `"cargo"`, `"cmake"`, `"go_mod"`） |
| `install_command` | `str` | `""` | インストールコマンド（例: `"pip install -r requirements.txt"`）。空の場合は `manager` から自動推定 |
| `build_file` | `str` | `""` | ビルド/依存ファイル名（例: `"requirements.txt"`, `"CMakeLists.txt"`, `"Cargo.toml"`） |
| `llm_generated_build` | `bool` | `False` | `True` の場合、LLM が実験コードと共にビルドファイルを生成 |
| `pre_install_commands` | `list[str]` | `[]` | インストール前に実行するコマンド（例: `["apt-get update"]`） |
| `post_install_commands` | `list[str]` | `[]` | インストール後に実行するコマンド（例: `["ldconfig"]`） |
| `install_timeout_sec` | `int` | `300` | 依存インストールのタイムアウト（秒） |
| `cache_dir` | `str` | `""` | パッケージキャッシュディレクトリ（空の場合はデフォルトを使用） |

**YAML 設定例（Python + pip）**:

```yaml
language:
  name: "python"
  interpreter_command: "python"
  file_extension: ".py"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "python"
  dependency:
    manager: "pip"
    install_command: "pip install -r requirements.txt"
    build_file: "requirements.txt"
    llm_generated_build: true
    install_timeout_sec: 300
```

**YAML 設定例（C++ + CMake）**:

```yaml
language:
  name: "cpp"
  compiled: true
  compile_command: "cmake --build build --config Release"
  compile_flags: []
  binary_name: "build/experiment"
  build_timeout_sec: 300
  file_extension: ".cpp"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "cpp"
  dependency:
    manager: "cmake"
    install_command: "cmake -B build -DCMAKE_BUILD_TYPE=Release"
    build_file: "CMakeLists.txt"
    llm_generated_build: true
    pre_install_commands: []
    install_timeout_sec: 300
```

**YAML 設定例（Rust + Cargo）**:

```yaml
language:
  name: "rust"
  compiled: true
  compile_command: "cargo build --release"
  binary_name: "target/release/experiment"
  file_extension: ".rs"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "rust"
  dependency:
    manager: "cargo"
    build_file: "Cargo.toml"
    llm_generated_build: true
    install_timeout_sec: 600
```

**YAML 設定例（Go + go mod）**:

```yaml
language:
  name: "go"
  compiled: true
  compile_command: "go build"
  compile_flags: ["-o", "experiment"]
  binary_name: "experiment"
  file_extension: ".go"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "go"
  dependency:
    manager: "go_mod"
    build_file: "go.mod"
    llm_generated_build: true
    install_timeout_sec: 300
```

##### 7.3.3.2 LLM 生成ビルドファイルのフロー

`llm_generated_build: true` の場合、`ExperimentGenerator.generate()` が実験コードに加えてビルドファイルも生成する。§7.2.1 の複数ファイルサポートと統合され、ビルドファイルは `GeneratedExperiment.files` の一部として扱われる：

```text
ExperimentGenerator.generate(node, problem_spec, agent_llm)
  │
  ├─ 1. 実験コード生成（§7.2 / §7.2.1 のフロー）
  │     → GeneratedExperiment.files にソースファイル群を格納
  │
  ├─ 2. dependency.llm_generated_build == True の場合:
  │     │
  │     ├─ LLM にビルドファイル生成を依頼
  │     │   プロンプト:
  │     │     「以下の実験コードが依存するライブラリのビルドファイルを生成せよ。
  │     │      言語: {language.name}
  │     │      依存管理ツール: {dependency.manager}
  │     │      ビルドファイル形式: {dependency.build_file}
  │     │      実験コード: {experiment_files}」
  │     │
  │     └─ 生成結果を GeneratedExperiment.files に追加
  │         → GeneratedFile(dependency.build_file, content)
  │
  ├─ 3. 全ファイルを runs/{node_id}/ 配下に書き出し
  │
  └─ 4. 返却: GeneratedExperiment
```

`llm_generated_build: false` の場合、ビルドファイルはプロジェクトルートまたは `ProblemSpec` で事前指定されたものを使用する。

**複数ファイル + ビルドファイルの統合例**（C++ プロジェクト）:

```json
{
  "entry_point": "main.cpp",
  "files": [
    {"path": "main.cpp", "content": "#include \"solver.h\"\n..."},
    {"path": "solver.h", "content": "#pragma once\n..."},
    {"path": "solver.cpp", "content": "#include \"solver.h\"\n..."},
    {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n..."}
  ]
}
```

##### 7.3.3.3 依存関係インストール実行フロー

```text
_install_dependencies(run_dir, language_config, allow_internet)
  │
  ├─ dependency = language_config.dependency
  │   dependency が未設定 → スキップ（即座にreturn）
  │
  ├─ 0. ネットワークアクセス制御（§7.3.3.6 参照）
  │     allow_internet == False の場合:
  │       環境変数から http_proxy / https_proxy を除去
  │       manager 別オフラインフラグを設定（pip: --no-index, cargo: CARGO_NET_OFFLINE=true 等）
  │       cache_dir またはローカルミラーが必須（未設定 → 即座に失敗）
  │
  ├─ 1. pre_install_commands の実行
  │     for cmd in dependency.pre_install_commands:
  │         subprocess.run(cmd, shell=True, cwd=run_dir, timeout=60)
  │
  ├─ 2. install_command の決定
  │     install_command が明示指定されている → そのまま使用
  │     install_command が空 → manager から自動推定:
  │       pip     → "pip install -r {build_file}"
  │       conda   → "conda install --file {build_file} -y"
  │       cargo   → スキップ（cargo build が自動で依存解決）
  │       cmake   → "{install_command}"（CMake configure ステップ）
  │       go_mod  → "go mod download"
  │
  ├─ 2b. セキュリティチェック（llm_generated_build == True の場合）
  │     allowed_packages が非空 → ビルドファイル内のパッケージ名を照合、未許可は拒否
  │     require_pinned_versions == True → バージョン未指定の依存を拒否
  │     生成ビルドファイルの内容を tool_execution_log.jsonl に記録
  │
  ├─ 3. インストール実行
  │     subprocess.run(install_command, shell=True, cwd=run_dir, timeout=install_timeout_sec)
  │     stdout → runs/{node_id}/install_stdout.log
  │     stderr → runs/{node_id}/install_stderr.log
  │     exit_code != 0 → RunResult(success=False) で即座に返却
  │
  ├─ 4. post_install_commands の実行
  │     for cmd in dependency.post_install_commands:
  │         subprocess.run(cmd, shell=True, cwd=run_dir, timeout=60)
  │
  └─ 5. キャッシュ管理
        cache_dir が指定されている場合、manager 固有の環境変数を設定:
          pip   → PIP_CACHE_DIR={cache_dir}
          cargo → CARGO_HOME={cache_dir}
          go    → GOPATH={cache_dir}
```

##### 7.3.3.4 ディレクトリ構成の拡張

複数ファイルプロジェクト（§7.2.1）・コンパイル型言語・依存管理で拡張されるディレクトリ構成：

```text
runs/<node_id>/
  ├─ experiment.{ext}           # エントリポイント（entry_point）
  ├─ {supplementary_files}      # 補助ファイル（新規: §7.2.1, utils.py, model.py, solver.h 等）
  ├─ src/                       # サブディレクトリ（新規: §7.2.1, Rust/Go 等の言語規約に準拠）
  │   └─ ...
  ├─ stdout.log                 # 実行時 stdout（既存）
  ├─ stderr.log                 # 実行時 stderr（既存）
  ├─ metrics.json               # 実験結果（既存）
  ├─ artifacts/                 # 実験成果物（既存）
  ├─ {build_file}               # ビルドファイル（新規: §7.3.3, requirements.txt, CMakeLists.txt 等）
  ├─ build_stdout.log           # ビルド時 stdout（新規: compiled=True の場合）
  ├─ build_stderr.log           # ビルド時 stderr（新規: compiled=True の場合）
  ├─ install_stdout.log         # 依存インストール stdout（新規: dependency 設定時）
  ├─ install_stderr.log         # 依存インストール stderr（新規: dependency 設定時）
  └─ build/                     # ビルド成果物（新規: cmake 等が使用）
```

##### 7.3.3.5 三層可変性モデルとの整合

| フィールド | 層 | Phase 1 以降 | 備考 |
|-----------|-----|-------------|------|
| `dependency.manager` | Frozen | 不変 | 依存管理ツールは研究開始前に決定 |
| `dependency.install_command` | Frozen | 不変 | |
| `dependency.build_file` | Frozen | 不変 | ファイル名のみ。内容は LLM 生成時は Manipulated |
| `dependency.llm_generated_build` | Frozen | 不変 | |
| `dependency.pre_install_commands` | Frozen | 不変 | |
| `dependency.post_install_commands` | Frozen | 不変 | |
| `dependency.install_timeout_sec` | Frozen | 不変 | |
| `dependency.cache_dir` | Frozen | 不変 | |
| ビルドファイルの**内容** | Manipulated | ノード毎に変化 | `llm_generated_build=True` 時のみ |

##### 7.3.3.6 ネットワークアクセス制御とセキュリティ

依存関係のインストール（`pip install`, `cargo build`, `go mod download` 等）はインターネットからのパッケージダウンロードを伴う。既存の `ResourceSpec.network.allow_internet`（§18.6）および `LocalExecutor.allow_internet` と整合する制御が必要である。

**`allow_internet` との連携**:

```text
_install_dependencies(run_dir, language_config, allow_internet)
  │
  ├─ allow_internet == False の場合:
  │   ├─ ネットワーク環境変数を除去（http_proxy, https_proxy 等）
  │   ├─ pip: --no-index --find-links={cache_dir} を付与
  │   ├─ cargo: CARGO_NET_OFFLINE=true を設定
  │   ├─ go: GOFLAGS=-mod=vendor または GOPROXY=off を設定
  │   └─ キャッシュまたはローカルミラーが利用不可 → RunResult(success=False) で即座に返却
  │
  └─ allow_internet == True の場合:
      └─ 通常のインストール（ネットワークアクセス許可）
```

**HPC エアギャップ環境への対応**:

SLURM 計算ノードが外部ネットワークにアクセスできない場合の推奨構成：

| 方式 | 説明 | 設定例 |
|------|------|--------|
| **事前キャッシュ** | ログインノードで依存を事前ダウンロードし `cache_dir` を共有FS に配置 | `cache_dir: "/shared/pkg_cache"` |
| **ローカルミラー** | PyPI/crates.io のローカルミラーを組織内で運用 | `pre_install_commands: ["pip config set global.index-url http://mirror.local/pypi/simple"]` |
| **vendor 方式** | 全依存をリポジトリ内に同梱（Go vendor, pip wheel） | `install_command: "pip install --no-index --find-links=./wheels -r requirements.txt"` |
| **コンテナ事前ビルド** | 依存を含むコンテナイメージを事前作成（§23.6 参照） | `container.image: "/shared/images/sera_with_deps.sif"` |

**LLM 生成ビルドファイルのセキュリティ**:

`llm_generated_build: true` の場合、LLM が生成した依存リストに悪意あるパッケージや既知の脆弱性を持つバージョンが含まれるリスクがある。以下の対策を実施する：

| 対策 | 実装場所 | 説明 |
|------|---------|------|
| **パッケージ名ホワイトリスト** | `DependencyConfig.allowed_packages: list[str]` | 空の場合は制限なし。設定時は LLM 生成の依存に含まれるパッケージ名をホワイトリストと照合 |
| **バージョンピン強制** | `DependencyConfig.require_pinned_versions: bool = False` | `True` の場合、バージョン未指定の依存を拒否 |
| **ビルドファイルのログ記録** | `tool_execution_log.jsonl` | 生成されたビルドファイルの内容を全てログに記録（監査用） |
| **`--dry-run` プレチェック** | `_install_dependencies` 内 | pip: `--dry-run` で実際のインストール前にパッケージ一覧を確認（対応する manager のみ） |

`DependencyConfig` への追加フィールド：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `allowed_packages` | `list[str]` | `[]` | 許可パッケージ名リスト（空 = 制限なし） |
| `require_pinned_versions` | `bool` | `False` | バージョンピンを強制するか |

##### 7.3.3.7 テスト計画

| テストケース | 概要 |
|-------------|------|
| `test_dependency_config_defaults` | デフォルト値で依存管理がスキップされること |
| `test_dependency_config_pip` | pip 設定の YAML パース・バリデーション |
| `test_dependency_config_cmake` | cmake 設定の YAML パース・バリデーション |
| `test_dependency_config_cargo` | cargo 設定の YAML パース・バリデーション |
| `test_dependency_config_go_mod` | go_mod 設定の YAML パース・バリデーション |
| `test_install_dependencies_pip` | pip install が正しいコマンドで実行されること |
| `test_install_dependencies_auto_command` | `install_command` 空時の自動推定が正しいこと |
| `test_install_dependencies_failure` | インストール失敗時に `RunResult(success=False)` が返ること |
| `test_install_dependencies_timeout` | タイムアウト時の適切なハンドリング |
| `test_install_dependencies_cache_dir` | キャッシュディレクトリ環境変数が正しく設定されること |
| `test_llm_generated_build_file` | LLM がビルドファイルを生成し `runs/{node_id}/` に保存されること |
| `test_pre_post_install_commands` | pre/post コマンドが正しい順序で実行されること |
| `test_full_compiled_flow` | 依存インストール → コンパイル → バイナリ実行の統合テスト |
| `test_install_offline_pip` | `allow_internet=False` 時に `--no-index` が付与されること |
| `test_install_offline_cargo` | `allow_internet=False` 時に `CARGO_NET_OFFLINE=true` が設定されること |
| `test_install_offline_go` | `allow_internet=False` 時に `GOPROXY=off` が設定されること |
| `test_install_no_network_no_cache_fails` | オフライン + キャッシュなし → `success=False` |
| `test_allowed_packages_whitelist` | ホワイトリスト外パッケージが拒否されること |
| `test_require_pinned_versions` | バージョン未指定の依存が拒否されること |
| `test_build_file_logged` | 生成ビルドファイルがログに記録されること |

### 7.4 metrics.json の規約（必須）
Executorは `runs/<node_id>/metrics.json` を出力する。

```json
{
  "primary": {"name": "score", "value": 0.73, "higher_is_better": true},
  "constraints": [
    {"name": "format_valid", "value": 1, "type": "bool", "satisfied": true},
    {"name": "latency_ms", "value": 45.2, "type": "le", "threshold": 100, "satisfied": true}
  ],
  "secondary": [
    {"name": "cost_gpu_sec", "value": 12.3, "lower_is_better": true},
    {"name": "memory_mb", "value": 2048.0, "lower_is_better": true}
  ],
  "raw": {"epoch": 10, "train_loss": 0.32, "val_loss": 0.41},
  "environment": {
    "python": "3.11.5",
    "torch": "2.3.0",
    "cuda": "12.1",
    "gpu": "NVIDIA A100 80GB"
  },
  "seed": 42,
  "wall_time_sec": 125.3
}
```

### 7.5 ストリーミング実行プロトコル

現在の `LocalExecutor.run()` は同期的な `subprocess.Popen` + `proc.wait()` を使用しており、Agent は実行中の出力をリアルタイムに参照できない。本節では非同期の行単位出力キャプチャを追加する。既存の `run()` は変更しない。

**デフォルト実行パス**: `StatisticalEvaluator` はデフォルトで `run_stream()` を使用する（`use_streaming=True`）。`run()` は同期的な代替パスとして残り、テストやデバッグ時に `use_streaming=False` で明示的に切り替え可能。全 Executor タイプ（Local, SLURM, Docker）でストリーミングがデフォルトである。

#### 7.5.1 ストリーミング型定義（`src/sera/execution/streaming.py` 新規）

```python
class StreamEventType(enum.Enum):
    # Core types (spec §7.5)
    STDOUT = "stdout"
    STDERR = "stderr"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    # Extensions (実装済み: metrics polling + keepalive)
    METRICS_UPDATE = "metrics_update"
    HEARTBEAT = "heartbeat"

@dataclass
class StreamEvent:
    event_type: StreamEventType
    data: str = ""                          # 行内容 or 終端サマリ
    elapsed_sec: float = 0.0
    exit_code: int | None = None            # 終端イベントのみ
    metadata: dict[str, Any] = field(default_factory=dict)
    # 終端イベントの metadata:
    #   stdout_tail, stderr_tail, metrics_path, run_result

StreamIterator = AsyncIterator[StreamEvent]
```

`StreamEvent` と `StreamEventType` を `src/sera/execution/__init__.py` の `__all__` に追加する。

#### 7.5.2 Executor 基底クラスへの `run_stream()` 追加

`Executor` ABC に非抽象のデフォルト `run_stream()` を追加する。`SlurmExecutor` / `DockerExecutor` はこれを自動継承する。

```python
class Executor(ABC):
    # run() は変更なし

    async def run_stream(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """デフォルト実装: run() を run_in_executor で呼び、
        ファイルから tail を読み取り、COMPLETED/TIMEOUT/ERROR イベントを 1 つ yield する。"""
        ...
```

| ID | 要件 |
|----|------|
| S-5.2.1 | デフォルト実装は `asyncio.get_event_loop().run_in_executor` で既存 `run()` を呼ぶ |
| S-5.2.2 | 完了後 `stdout_path`/`stderr_path` 末尾 50 行を読み `metadata` に格納 |
| S-5.2.3 | `exit_code == -9` → `TIMEOUT`, `success == True` → `COMPLETED`, 他 → `ERROR` |
| S-5.2.4 | `run()` が例外を送出した場合は `ERROR` イベントを yield |

#### 7.5.3 LocalExecutor の `run_stream()` オーバーライド

`LocalExecutor` は `asyncio.create_subprocess_exec` + `PIPE` で真のストリーミングを実装する。

```python
class LocalExecutor(Executor):
    # run() は変更なし

    async def run_stream(self, node_id, script_path, seed, timeout_sec=None):
        """asyncio.create_subprocess_exec で行単位ストリーミング。
        asyncio.Queue + 2 reader タスクで stdout/stderr を時系列順にマージ。"""
        ...
```

| ID | 要件 |
|----|------|
| S-5.3.1 | `asyncio.create_subprocess_exec` + `PIPE` で stdout/stderr を非同期取得 |
| S-5.3.2 | `asyncio.Queue` + 2 reader タスクで stdout/stderr を時系列順にマージして yield |
| S-5.3.3 | メモリ内バッファは `deque(maxlen=1000)` で上限付き |
| S-5.3.4 | プロセス終了後 `stdout.log` / `stderr.log` にフル出力を書き込む（§7.3 RunResult 契約維持） |
| S-5.3.5 | OOM 検出ロジックは §7.3 `run()` と同一（exit_code 137/-9 + stderr パターン → `-7`） |
| S-5.3.6 | 終端イベントの `metadata` に `stdout_tail`, `stderr_tail`, `stdout_line_count`, `stderr_line_count`, `metrics_path`, `run_result` を格納 |
| S-5.3.7 | `run()` は一切変更しない |
| S-5.3.8 | シェル構文を含む `interpreter_command` は非対応（基底クラスのデフォルト実装にフォールバック可） |

#### 7.5.4 テスト（`tests/test_execution/test_streaming.py` 新規、`tests/test_execution/test_local_executor.py` 追記）

| テストケース | 概要 |
|-------------|------|
| `TestStreamEvent.test_event_types` | 7 つの `StreamEventType` 値を検証（core 5 + extension 2） |
| `TestStreamEvent.test_event_defaults` | デフォルト値（`data=""`, `exit_code=None`, `metadata={}` 等） |
| `TestStreamEvent.test_metadata_isolation` | 各イベントが独立した `metadata` dict を持つ |
| `TestLocalExecutorRunStream.test_run_stream_successful` | 成功スクリプト → STDOUT イベント + COMPLETED 終端 |
| `TestLocalExecutorRunStream.test_run_stream_timeout` | タイムアウト → TIMEOUT 終端、`exit_code == -9` |
| `TestLocalExecutorRunStream.test_run_stream_error` | エラースクリプト → ERROR 終端、`exit_code != 0` |
| `TestLocalExecutorRunStream.test_run_stream_preserves_artifacts` | `stdout.log`/`stderr.log` ファイルが書かれている |
| `TestLocalExecutorRunStream.test_run_stream_empty_output` | 出力なし → 終端イベント 1 つのみ |
| `TestLocalExecutorRunStream.test_run_stream_metadata_has_run_result` | 終端 `metadata` に `RunResult` が格納 |
| `TestExecutorBaseRunStream.test_default_wraps_run` | 基底クラスのデフォルト実装が `run()` をラップ |
| `TestExecutorBaseRunStream.test_default_handles_run_exception` | `run()` 例外時に ERROR イベント |
| `TestLocalExecutorRunStream.test_run_stream_oom` | stderr に MemoryError + exit 137 → `exit_code == -7` |

### 7.6 実験失敗時のハンドリング
```text
（§7.5 から番号繰り下げ）

1. exit_code != 0 の場合：
   - metrics.json が存在しない → node.status = "failed"、primary = -inf、feasible = false
   - metrics.json が部分的 → 読めるフィールドのみ使用、不足は NaN
2. タイムアウトの場合：
   - node.status = "timeout"、cost = timeout_sec（全消費扱い）
3. OOM の場合：
   - node.status = "oom"、cost = 最大予算（ペナルティ扱い）
4. リトライ：
   - 失敗ノードのリトライは行わない（別の seed で新ノードとして扱う）
```

---
