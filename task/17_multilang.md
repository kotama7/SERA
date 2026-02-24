# SERA 要件定義書 — 多言語実験サポート

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 23. 多言語実験サポート（Multi-Language Experiment Support）

### 23.1 概要

SERAは実験スクリプトをPython以外の言語（R, Julia, Go, C++, bash等）でも生成・実行できる。
言語設定は `ProblemSpec.language` で指定され、Phase 1で固定される。

### 23.2 LanguageConfig スキーマ

```yaml
# ProblemSpec 内
language:
  name: "python"              # 言語名（プロンプト生成に使用）
  interpreter_command: "python" # インタプリタコマンド
  file_extension: ".py"        # 実験スクリプトの拡張子
  seed_arg_format: "--seed {seed}" # シード引数のフォーマット文字列
  code_block_tag: "python"     # Markdownコードブロックのタグ
```

**例: R言語の設定**
```yaml
language:
  name: "R"
  interpreter_command: "Rscript"
  file_extension: ".R"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "r"
```

**例: Julia の設定**
```yaml
language:
  name: "julia"
  interpreter_command: "julia"
  file_extension: ".jl"
  seed_arg_format: "-- --seed {seed}"
  code_block_tag: "julia"
```

### 23.3 影響範囲

| コンポーネント | 変更内容 |
|---------------|---------|
| `ExperimentGenerator` | スクリプトファイル名を `experiment{file_extension}` に動的生成。プロンプトに言語名とコードブロックタグを使用 |
| `LocalExecutor` | `interpreter_command` と `seed_arg_format` に基づきサブプロセスを起動 |
| `SlurmExecutor` | 同上（SLURM ジョブ内部で使用） |
| `DockerExecutor` | 同上（コンテナ内で使用、スタブ） |
| `TreeOps.debug` | デバッグプロンプトのコードブロックタグを動的に設定 |
| `replay_cmd` | `experiment.*` パターンでスクリプトを検索、LanguageConfigからインタプリタを決定 |
| `export_cmd` | `experiment.*` パターンで全実験スクリプトをコピー |
| `status_cmd` | `experiment.*` パターンでスクリプトを表示 |
| `StatisticalEvaluator` | `experiment.*` パターンで既存スクリプトを検索 |

### 23.4 metrics.json 出力契約（言語非依存）

全言語の実験スクリプトは同一の `metrics.json` スキーマに準拠する：

```json
{
  "primary": {
    "name": "<metric_name>",
    "value": 0.95,
    "higher_is_better": true
  },
  "constraints": [],
  "secondary": [],
  "raw": {},
  "seed": 42,
  "wall_time_sec": 120.5,
  "<metric_name>": 0.95
}
```

`metrics.json` は実験スクリプトのカレントディレクトリにファイルとして出力する（stdoutではない）。
最上位に `"<metric_name>": <float>` を含めることで後方互換性を維持する。

### 23.5 デフォルト動作

`ProblemSpec.language` が未指定の場合、Python がデフォルト：
- `name: "python"`, `interpreter_command: "python"`, `file_extension: ".py"`
- `seed_arg_format: "--seed {seed}"`, `code_block_tag: "python"`

既存のワークスペースは変更なしで動作する（後方互換）。

---

---
