import json

from sftmill.dataset import JsonlDatasetWriter, count_jsonl_dataset, iter_jsonl_dataset, split_by_group


def test_writer_shards_and_stream_read(tmp_path):
    root = tmp_path / "traces"
    with JsonlDatasetWriter(root, shard_size=2) as writer:
        for index in range(5):
            writer.write({"id": index, "kind": "trace", "messages": []})
    assert count_jsonl_dataset(root) == 5
    shards = sorted(root.glob("*.jsonl"))
    assert len(shards) == 3
    rows = list(iter_jsonl_dataset(root))
    assert [row["id"] for row in rows] == [0, 1, 2, 3, 4]


def test_single_file_append(tmp_path):
    path = tmp_path / "all.jsonl"
    with JsonlDatasetWriter(path, shard_size=100) as writer:
        writer.write({"n": 1})
        writer.write({"n": 2})
    with JsonlDatasetWriter(path, shard_size=100) as writer:
        writer.write({"n": 3})
    rows = list(iter_jsonl_dataset(path))
    assert [row["n"] for row in rows] == [1, 2, 3]


def test_split_keeps_harness_siblings_together():
    rows = []
    for group in ("search_edit-0001", "search_edit-0002"):
        for harness in ("alice", "openai", "claude_code", "novel"):
            rows.append({"id": f"{group}-{harness}", "group_id": group, "kind": "trajectory"})
    split = None
    for seed in range(20):
        train, val = split_by_group(rows, val_fraction=0.5, seed=seed)
        if train and val:
            split = (train, val)
            break
    assert split is not None
    train, val = split
    train_groups = {row["group_id"] for row in train}
    val_groups = {row["group_id"] for row in val}
    assert train_groups.isdisjoint(val_groups)
    assert len(train) + len(val) == len(rows)
    assert all(sum(1 for row in rows if row["group_id"] == group) == 4 for group in train_groups | val_groups)


def test_load_jsonl_still_reads_file(tmp_path):
    from sftmill.schema import load_jsonl

    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n")
    assert load_jsonl(path) == [{"a": 1}]
