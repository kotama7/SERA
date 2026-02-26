# SERA 要件定義書 — Phase 6: 系譜管理・剪定

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 10. Phase 6：系譜管理・剪定

### 10.1 LoRA系譜（必須）
- 各探索ノードは LoRA系譜ノードを参照する
- `adapter_spec_hash` が一致しない差分の合成は禁止（互換性検証）
  ```python
  def validate_compatibility(child_meta: dict, parent_meta: dict) -> bool:
      return child_meta["adapter_spec_hash"] == parent_meta["adapter_spec_hash"]
  ```

### 10.2 materialize（復元：必須）
```python
def materialize(adapter_node_id: str, lineage_dir: Path, cache: LRUCache) -> dict[str, Tensor]:
    """
    adapter_node_id のフルLoRAウェイトを復元する。

    アルゴリズム:
    1. キャッシュにあればそのまま返す
    2. meta.json を読み、is_snapshot=true ならスナップショットをロードして返す
    3. 親→ルートまでの系譜パスを構築: [root, ..., parent, self]
    4. 系譜パス上で最も近いスナップショットまたはキャッシュを探す
    5. そこからΔを順次累積: Θ = Θ_snapshot + Σ Δ_i
    6. 結果をキャッシュに保存
    7. キャッシュが cache_max_entries を超えたらLRU追い出し
    """
    # 1. キャッシュ確認
    if adapter_node_id in cache:
        return cache[adapter_node_id]

    # 2. スナップショット確認
    meta = load_meta(lineage_dir / adapter_node_id / "meta.json")
    if meta["is_snapshot"]:
        weights = load_safetensors(lineage_dir / adapter_node_id / "snapshot.safetensors")
        cache[adapter_node_id] = weights
        return weights

    # 3-5. 系譜を辿って累積
    path = build_lineage_path(adapter_node_id, lineage_dir)  # [root, ..., self]
    base_weights = None
    start_idx = 0
    for i, nid in enumerate(reversed(path)):
        if nid in cache:
            base_weights = cache[nid].copy()
            start_idx = len(path) - i
            break
        m = load_meta(lineage_dir / nid / "meta.json")
        if m["is_snapshot"]:
            base_weights = load_safetensors(lineage_dir / nid / "snapshot.safetensors")
            start_idx = len(path) - i
            break

    if base_weights is None:
        base_weights = {k: torch.zeros_like(v) for k, v in get_lora_template(meta).items()}
        start_idx = 0

    for nid in path[start_idx:]:
        delta = load_safetensors(lineage_dir / nid / "adapter_delta.safetensors")
        for key in base_weights:
            delta_key = key.replace(".weight", ".delta")
            if delta_key in delta:
                base_weights[key] = base_weights[key] + delta[delta_key]

    # 6-7. キャッシュ
    cache[adapter_node_id] = base_weights
    return base_weights
```

### 10.3 squash（スナップショット化）
`ExecutionSpec.lora_runtime` に従い：

```python
def maybe_squash(lineage_manager, exec_spec):
    """
    スナップショット生成条件:
    1. depth >= squash_depth（既定6）のノード
    2. snapshot_on_topk=true かつ Top-k（pruning.keep_topk）に含まれるノード
    3. 既にスナップショット済みのノードはスキップ

    処理:
    1. materialize で全ウェイトを復元
    2. `snapshot.safetensors` として保存（ファイル名は `snapshot.safetensors` で統一）
    3. meta.json の is_snapshot = true に更新
    4. このノードより上の祖先のΔは不要にはしない（他の枝が参照する可能性）
    """
    pass
```

### 10.4 剪定（必須：具体アルゴリズム）
```python
def prune(open_list, closed_set, all_nodes, exec_spec) -> list[SearchNode]:
    """
    剪定アルゴリズム:

    1. 保護リスト構築:
       - best_node とその祖先（root まで）
       - LCB上位 keep_topk（既定5）ノードとその祖先
       - status="running" のノード

    2. Pareto剪定（pareto=true の場合）:
       - primary(LCB) と cost の2軸でPareto支配判定
       - 支配されるノード = 他のノードに primary も cost も負けるノード
       - 支配されかつ保護リストにないノードを候補に

    3. LCB閾値剪定:
       - lcb_threshold が null なら: threshold = best_lcb * 0.5
       - LCB < threshold のノード（保護リスト除外）を候補に

    4. 予算剪定:
       - 全ノードの累計コスト > budget_limit なら、LCBワースト順に削除

    5. 候補に対して:
       - node.status = "pruned"
       - open_list から除去
       - save_pruned=false なら runs/<node_id>/ を削除
       - LoRA delta は保持（他ノードの祖先の可能性）
    """
    pass
```

---
