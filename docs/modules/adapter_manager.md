# LineageManager / Pruner / LRUCache

LoRA アダプタのリネージ（系譜）管理を担当するモジュール群のドキュメント。

**注意:** 実装上のモジュール名は `lineage_manager` であり、このファイル名 `adapter_manager.md` は慣例的なもの。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `LineageManager` | `src/sera/lineage/lineage_manager.py` |
| `Pruner` | `src/sera/lineage/pruner.py` |
| `LRUCache` | `src/sera/lineage/cache.py` |

## 依存関係

- `safetensors.torch` (`save_file`, `load_file`) -- テンソルの永続化
- `peft` (`get_peft_model_state_dict`) -- アダプタ重みの抽出
- `sera.utils.hashing` (`compute_adapter_spec_hash`) -- アダプタスペックのハッシュ

---

## LineageManager

LoRA アダプタのデルタ（差分）をディスク上でツリー構造として管理するクラス。PPO 更新ごとに親アダプタとの差分を保存し、必要に応じてデルタの累積和で完全な重みを復元（マテリアライズ）する。

### ディレクトリ構造

```
lineage/
  nodes/
    <adapter_node_id>/
      meta.json                      # メタデータ
      adapter_delta.safetensors      # 親からのデルタ
      adapter_snapshot.safetensors   # 完全な重み（スナップショット、任意）
```

### コンストラクタ

```python
def __init__(self, lineage_dir: Path, cache_size: int = 10)
```

- `lineage_dir`: リネージストレージのルートディレクトリ
- `cache_size`: LRU キャッシュの最大エントリ数（デフォルト 10）
- `nodes_dir = lineage_dir / "nodes"` が自動作成される

### save_delta(adapter_node_id, parent_id, delta_tensors, search_node_id, depth, adapter_spec_hash) -> Path

アダプタのデルタとメタデータを永続化する。

1. `adapter_delta.safetensors`: `contiguous().cpu()` に変換したテンソルを `safetensors.torch.save_file()` で保存
2. `meta.json`: 全メタデータを JSON で保存
3. 該当ノードの LRU キャッシュを無効化
4. 保存先ディレクトリのパスを返す

### meta.json スキーマ

```json
{
  "adapter_node_id": "...",
  "parent_id": "...",
  "search_node_id": "...",
  "depth": 3,
  "adapter_spec_hash": "sha256:...",
  "is_snapshot": false,
  "tensor_names": ["lora_A.weight", "lora_B.weight", ...],
  "tensor_shapes": {"lora_A.weight": [16, 4096], ...}
}
```

### materialize(adapter_node_id) -> dict[str, Tensor]

完全な LoRA 重みをデルタの累積和で復元する。

**処理フロー:**

1. LRU キャッシュを確認（ヒットすればそのまま返す）
2. `build_lineage_path()` でルートからノードまでのパスを構築
3. パス内で最も深いスナップショットを検索
4. スナップショットがある場合: そこから開始して残りのデルタを加算
5. スナップショットがない場合: ルートから全デルタを加算
6. 結果をキャッシュに格納して返す

### maybe_squash(exec_spec) -> list[str]

深いノードに対してスナップショットを作成する。

- 閾値: `exec_spec.search.squash_depth`、未設定の場合は `max_depth // 2`（max_depth のデフォルトは 10）
- `is_snapshot == False` かつ `depth >= squash_threshold` のノードが対象
- `_create_snapshot()`: `materialize()` で完全な重みを取得し、`adapter_snapshot.safetensors` として保存、`meta.json` の `is_snapshot` を `True` に更新
- スナップショットが作成されたノード ID のリストを返す

### build_lineage_path(adapter_node_id) -> list[str]

ルートから指定ノードまでの系譜パスを構築する。

- `parent_id` を辿ってルートまで逆順にトラバース
- サイクル検出あり（`visited` セット）
- 結果を反転して `[root, ..., adapter_node_id]` の順序で返す

### validate_compatibility(child_meta, parent_meta) -> bool

子と親のアダプタスペックの互換性を検証する。

1. `adapter_spec_hash` の一致を確認
2. 親の全テンソルキーが子に存在し、shape が一致することを確認

### export_for_vllm(adapter_node_id, output_dir, model_spec) -> Path

vLLM 用にアダプタを peft 形式でエクスポートする。

出力ファイル:
- `adapter_model.safetensors`: マテリアライズされた完全な重み
- `adapter_config.json`: peft 形式の設定（`peft_type`, `task_type`, `r`, `lora_alpha`, `target_modules`, `lora_dropout`, `bias`, `base_model_name_or_path`）

### extract_delta_from_model(model, parent_state=None) -> dict[str, Tensor]

静的メソッド。peft モデルからアダプタのデルタを抽出する。

- `peft.get_peft_model_state_dict()` を使用
- `AutoModelForCausalLMWithValueHead` の場合は `.pretrained_model` を経由
- `parent_state` が `None` の場合: 現在の重みをそのままクローンして返す（ルートノード用）
- `parent_state` がある場合: `current - parent` の差分を返す

### get_meta(adapter_node_id) -> dict | None

指定ノードのメタデータを読み込む。ファイルが存在しない場合は `None`。

---

## Pruner

探索木の枝刈りを行うクラス。全 public メソッドはステートレスで、実行時に `exec_spec` から設定を読み取る。

### prune(open_list, closed_set, all_nodes, exec_spec) -> list[str]

3 つの枝刈りパスを順に実行し、枝刈りされたノード ID のリストを返す。枝刈りされたノードの `status` は `"pruned"` に設定される。

**実行順序:**

1. **LCB 閾値枝刈り** (`_lcb_threshold_prune`)
2. **パレート枝刈り** (`_pareto_prune`)
3. **予算枝刈り** (`_budget_prune`)

重複は排除される（`dict.fromkeys()`）。

### 保護セット (_build_protection_set)

以下のノードは枝刈りから保護される:

- **最良ノード + その全祖先**: LCB が最高の評価済みノードと、ルートまでの全祖先
- **top-k + その全祖先**: LCB 上位 `keep_topk`（`exec_spec.pruning.keep_topk`、デフォルト 5）のノードと祖先
- **実行中ノード**: `status == "running"` の全ノード

### LCB 閾値枝刈り

- `exec_spec.pruning.reward_threshold` が 0 でない場合: その値を閾値として使用
- そうでない場合（自動モード）: `best_lcb * 0.5` を閾値とする
- `open_list` 内で保護されておらず `lcb < threshold` のノードを枝刈り

### パレート枝刈り

(LCB, コスト) 空間でのパレート支配に基づく枝刈り。

- ノード A がノード B に支配される条件: `B.lcb >= A.lcb` かつ `B.total_cost <= A.total_cost`（少なくとも一方が厳密不等式）
- 保護セットと既に枝刈り済みのノードは除外

### 予算枝刈り

全ノードの合計コストが予算上限を超えた場合、LCB が最低のノードから順に枝刈り。

- 予算上限: `termination.max_wall_time_hours * 3600` または `termination.max_wallclock_hours * 3600`（デフォルト 14400 秒 = 4 時間）
- 合計コストが予算内に収まるまで枝刈りを継続

### LoRA デルタの扱い

枝刈りされたノードの LoRA デルタは**保持される**。他のブランチが祖先として参照する可能性があるため、ディスク上のデータは削除しない。

---

## LRUCache

`OrderedDict` ベースの固定容量 LRU キャッシュ。`LineageManager` がマテリアライズしたアダプタ重みをメモリ上に保持するために使用する。

### コンストラクタ

```python
def __init__(self, max_entries: int = 10)
```

- `max_entries` は 1 以上（1 未満は `ValueError`）

### インターフェース

| メソッド/演算子 | 説明 |
|---------------|------|
| `__contains__(key)` | キーの存在確認 |
| `__getitem__(key)` | 取得（最近使用として末尾に移動） |
| `__setitem__(key, value)` | 挿入/更新（容量超過時は最古のエントリを削除） |
| `__len__()` | エントリ数 |
| `get(key, default=None)` | キーが存在すれば取得、なければ default |
| `clear()` | 全エントリ削除 |
| `keys()` | LRU 順（最古が先頭）のキーリストを返す |
