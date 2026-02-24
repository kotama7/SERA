# SERA 再現性ガイド

このドキュメントでは、SERA に実装されている再現性保証メカニズムについて説明します。

## シード管理

### グローバルシード設定

`sera.utils.seed.set_global_seed(seed)` は以下の 4 つの乱数生成器を一括設定します:

```python
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

ソースファイル: `src/sera/utils/seed.py`

### ノード別シード導出

各探索ノード・各繰り返しに対して決定論的なシードが導出されます:

```python
def get_seed_for_node(base_seed: int, node_id: str, repeat_idx: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{node_id}:{repeat_idx}".encode()).hexdigest()
    return int(h, 16) % (2**31)
```

この方式により、同じ `base_seed`、`node_id`、`repeat_idx` の組み合わせに対して、実行環境やタイミングによらず常に同一のシードが得られます。

### 非決定性の制限

LLM 呼び出しにおいて `temperature > 0` の場合、サンプリングの確率的性質により出力は完全には決定論的になりません。これは設計上の制約であり、固定シードを使用しても LLM 応答の完全な再現は保証されません。

## Spec 凍結

### ExecutionSpec のハッシュロック

Phase 1（`sera freeze-specs`）の実行時に、`ExecutionSpecModel` の内容が SHA-256 ハッシュとして `execution_spec.yaml.lock` に記録されます。

ハッシュ計算は `sera.utils.hashing.compute_spec_hash()` で実装されています:

```python
def compute_spec_hash(spec_dict: dict) -> str:
    canonical = json.dumps(spec_dict, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return f"sha256:{h}"
```

ハッシュ値のフォーマットは `sha256:<64文字の16進数>` です。

### 整合性検証

`sera research` の開始時に `SpecFreezer.verify()` が自動的に呼び出されます:

1. `specs/execution_spec.yaml` を読み込み、`compute_spec_hash()` でハッシュを計算
2. `specs/execution_spec.yaml.lock` のハッシュ値と比較
3. 不一致の場合、**終了コード 2** でプロセスを終了

```python
# sera/commands/research_cmd.py
freezer = SpecFreezer()
if not freezer.verify(specs_dir):
    console.print("[red]ExecutionSpec tampered! Aborting.[/red]")
    sys.exit(2)
```

### モデルメタデータの自動取得

Phase 1 の凍結処理中に、`SpecFreezer.freeze()` は以下のメタデータを自動取得して `model_spec.yaml` に記録します:

- **`base_model.revision`**: `transformers.AutoConfig.from_pretrained()` を呼び出し、`_commit_hash` を取得（取得失敗時は `"unknown"`）
- **`compatibility.adapter_spec_hash`**: `AdapterSpec` の内容から `compute_adapter_spec_hash()` でハッシュを計算

## LLM 呼び出しログ

すべての LLM 呼び出しは `logs/agent_llm_log.jsonl` に JSONL 形式で記録されます。

### ログエントリの形式

```json
{
  "event": "llm_call",
  "call_id": "uuid4形式の一意識別子",
  "purpose": "branch_generation",
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "adapter_node_id": "node_abc123",
  "prompt_tokens": 1523,
  "completion_tokens": 412,
  "temperature": 0.7,
  "prompt_hash": "sha256:abcdef0123456789",
  "response_hash": "sha256:fedcba9876543210",
  "latency_ms": 2340.5,
  "timestamp": "2026-02-22T10:30:00+00:00"
}
```

各フィールドの詳細:

| フィールド | 説明 |
|---|---|
| `call_id` | UUID v4 による一意の呼び出し識別子 |
| `purpose` | 呼び出しの目的（例: `"branch_generation"`, `"citation_identify"`, `"citation_select"`） |
| `model_id` | 使用されたモデルの識別子（ローカルの場合はベースモデル ID、API の場合はモデル名） |
| `adapter_node_id` | 適用されたアダプタのノード ID（アダプタなしの場合は `null`） |
| `prompt_tokens` | プロンプトのトークン数（単語分割による近似値） |
| `completion_tokens` | 応答のトークン数（単語分割による近似値） |
| `prompt_hash` | プロンプト内容の SHA-256 ハッシュ（先頭 16 文字） |
| `response_hash` | 応答内容の SHA-256 ハッシュ（先頭 16 文字） |
| `latency_ms` | 呼び出しのレイテンシ（ミリ秒） |
| `timestamp` | UTC タイムスタンプ（ISO 8601 形式） |

ソースファイル: `src/sera/agent/agent_llm.py`

## API クエリログ

Phase 0 の先行研究収集中に、`RelatedWorkEngine` は各クエリを JSONL 形式でログに記録します:

```json
{
  "event": "phase0_query",
  "query": "transformer scheduling optimization",
  "num_results": 20,
  "timestamp": "2026-02-22T10:30:00+00:00"
}
```

Phase 0 完了時には集計情報も記録されます:

```json
{
  "event": "phase0_complete",
  "total_searched": 150,
  "top_k": 10,
  "num_clusters": 3,
  "teacher_papers": 5,
  "timestamp": "2026-02-22T10:35:00+00:00"
}
```

## 実験リプレイ

### `sera replay` コマンド

特定ノードの実験を指定したシードで再実行できます:

```bash
sera replay <node_id> <seed>
```

### 動作の詳細

1. `runs/{node_id}/` ディレクトリ内で `experiment.*` のグロブパターンにより実験スクリプトを検索
2. `runs/{node_id}_replay_{seed}/` にスクリプトをコピー
3. `ProblemSpec` の `language` 設定から `interpreter_command` と `seed_arg_format` を取得
4. `LocalExecutor.run()` で実験を実行
5. 成功時には `metrics.json` の内容を表示、失敗時には stderr の末尾 500 文字を表示

ソースファイル: `src/sera/commands/replay_cmd.py`

## 探索状態チェックポイント

### 定期保存

`SearchManager` は **10 ステップごと**にチェックポイントを自動保存します:

```
checkpoints/search_state_step_10.json
checkpoints/search_state_step_20.json
...
```

チェックポイントには以下の状態が含まれます:
- 現在のステップ番号
- すべてのノードの状態（`all_nodes`）
- クローズドセット（処理済みノード集合）
- ベストノード ID
- オープンリスト（優先度キュー）
- PPO バッファ

### SIGINT ハンドリング

`SearchManager` は `SIGINT`（Ctrl+C）を受信すると、現在の状態をチェックポイントに保存してから**終了コード 20** で終了します:

```python
def _handler(signum, frame):
    logger.warning("SIGINT received, saving checkpoint and exiting")
    try:
        state = self.save_state()
        save_checkpoint(state, self.checkpoint_dir, self.step)
    except Exception as e:
        logger.error("Failed to save checkpoint on SIGINT: %s", e)
    sys.exit(20)
```

### 再開

`sera research --resume` で最新のチェックポイントから探索を再開できます:

```python
from sera.utils.checkpoint import load_latest_checkpoint
state = load_latest_checkpoint(checkpoint_dir)
if state:
    manager.load_state(state)
```

`load_latest_checkpoint()` は `checkpoints/` 内の `search_state_step_*.json` をソートし、最新のファイルを読み込みます。

## LoRA lineage トラッキング

### ディレクトリ構造

```
lineage/
  nodes/
    <adapter_node_id>/
      meta.json                    # メタデータ
      adapter_delta.safetensors    # 親からのデルタ重み
      adapter_snapshot.safetensors # 完全重み（スナップショット、オプション）
```

### meta.json の内容

各アダプタノードのメタデータは以下のフィールドを持ちます:

```json
{
  "adapter_node_id": "adapter_001",
  "parent_id": "adapter_000",
  "search_node_id": "node_abc",
  "depth": 3,
  "adapter_spec_hash": "sha256:...",
  "is_snapshot": false,
  "tensor_names": ["lora_A.weight", "lora_B.weight"],
  "tensor_shapes": {
    "lora_A.weight": [16, 4096],
    "lora_B.weight": [4096, 16]
  }
}
```

### 互換性検証

`LineageManager.validate_compatibility()` は子ノードと親ノードのメタデータを比較し、以下の条件を確認します:

1. **`adapter_spec_hash` の一致**: LoRA の rank、alpha、target_modules 等の設定が同一であること
2. **テンソル形状の一致**: 親のすべてのテンソルキーが子に存在し、形状が一致すること

これにより、設定変更によるデルタ合成の破綻を防止します。

### デルタ合成

重みの復元（materialization）は、ルートからノードまでのパスに沿ってデルタを累積加算することで行われます。途中にスナップショット（`is_snapshot: true`）が存在する場合、そのスナップショットを起点として加算を開始し、計算コストを抑制します。

## 再現性チェックリスト

以下は SERA の実行で自動的に記録される項目の一覧です:

| 項目 | 記録方法 | 格納場所 |
|---|---|---|
| シード | ノードごとにハッシュ導出 | 実行時に計算（`get_seed_for_node()`） |
| ベースモデル | `ModelSpec.base_model.id` + `revision` | `specs/model_spec.yaml` |
| LLM 呼び出し | prompt/response ハッシュ付きで全件記録 | `logs/agent_llm_log.jsonl` |
| API クエリ | クエリ内容と結果件数を記録 | `logs/` 内の JSONL ファイル |
| 実験成果物 | スクリプト + stdout + stderr + metrics | `runs/{node_id}/` |
| Spec ファイル | 10 個の YAML ファイル | `specs/` |
| チェックポイント | 10 ステップごとに探索状態を保存 | `checkpoints/search_state_step_N.json` |
| アダプタ lineage | デルタ重み + メタデータ | `lineage/nodes/{adapter_node_id}/` |
| ExecutionSpec ロック | SHA-256 ハッシュ | `specs/execution_spec.yaml.lock` |
