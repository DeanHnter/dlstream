#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("restore_streamable_hf_dataset")


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Download a streamable Hugging Face Parquet dataset export and restore docs/, bench/, and the local Qdrant index.",
    )
    parser.add_argument(
        "repo_id",
        help="Source dataset repo id, for example 'your-org/enterprise-rag-bench-streamable'.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=settings.benchmark_root,
        help="Destination benchmark root for docs/ and bench/.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=settings.qdrant_path,
        help="Destination local Qdrant path.",
    )
    parser.add_argument(
        "--token",
        default=(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        ),
        help="Hugging Face token. Defaults to HF_TOKEN if set.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to restore. Defaults to main.",
    )
    parser.add_argument(
        "--index-repo-id",
        default="",
        help="Optional dataset repo id containing the original uploaded index/qdrant_data files to restore directly.",
    )
    parser.add_argument(
        "--index-revision",
        default="main",
        help="Revision for --index-repo-id. Defaults to main.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for reading parquet and upserting Qdrant points. Defaults to 64.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing docs/, question files, qdrant index, and embedding manifest.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Do not restore docs/ files.",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Do not restore bench question JSONL files.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not rebuild the local Qdrant index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be restored without writing files or downloading data.",
    )
    return parser.parse_args()


def resolve_destination_path(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path.resolve()

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    repo_candidate = (_REPO_ROOT / raw_path).resolve()
    script_candidate = (Path(__file__).resolve().parent / raw_path).resolve()

    for candidate in (cwd_candidate, repo_candidate, script_candidate):
        if candidate.exists():
            return candidate
    return cwd_candidate


def load_json_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def ensure_can_write(path: Path, *, force: bool, description: str) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"{description} already exists: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def download_snapshot(
    repo_id: str,
    *,
    token: str | None,
    revision: str,
    allow_patterns: list[str],
    dry_run: bool,
) -> Path | None:
    if dry_run:
        logger.info("Dry run requested; skipping snapshot download for %s", repo_id)
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Reinstall dependencies or run: pip install -r requirements.txt"
        ) from exc

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token or None,
            allow_patterns=allow_patterns,
        )
    )


def write_jsonl_from_parquet(parquet_path: Path, output_path: Path) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def write_document_file(docs_dir: Path, row: dict[str, object]) -> None:
    relative_path = str(row.get("relative_path") or "").strip()
    if not relative_path:
        raise ValueError("Missing relative_path in document row")

    title = str(row.get("title") or "").strip()
    body = str(row.get("body") or "").strip()
    text = title if not body else f"{title}\n\n{body}"

    output_path = docs_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def iter_parquet_rows(train_dir: Path, *, batch_size: int):
    import pyarrow.parquet as pq

    for parquet_path in sorted(train_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=max(1, batch_size)):
            for row in batch.to_pylist():
                yield row


def rebuild_qdrant_index(
    train_dir: Path,
    *,
    qdrant_path: Path,
    embedding_model: str,
    batch_size: int,
    expected_documents: int | None = None,
    progress_interval: int = 5_000,
) -> tuple[int, int]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Reinstall dependencies or run: pip install -r requirements.txt"
        ) from exc

    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    collection_name = "enterprise_docs"
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    first_row: dict[str, object] | None = None
    for row in iter_parquet_rows(train_dir, batch_size=1):
        first_row = row
        break
    if first_row is None:
        return 0, 0

    first_embedding = list(first_row.get("embedding") or [])
    vector_size = len(first_embedding)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    indexed_count = 0
    pending_rows: list[dict[str, object]] = [first_row]
    row_iterator = iter_parquet_rows(train_dir, batch_size=max(1, batch_size))
    next(row_iterator, None)

    for row in row_iterator:
        pending_rows.append(row)
        if len(pending_rows) >= max(1, batch_size):
            indexed_count += upsert_rows(client, collection_name, pending_rows)
            pending_rows = []
            if indexed_count % max(1, progress_interval) < max(1, batch_size):
                if expected_documents:
                    logger.info("Rebuilt Qdrant index for %s / %s documents", indexed_count, expected_documents)
                else:
                    logger.info("Rebuilt Qdrant index for %s documents", indexed_count)
            gc.collect()

    if pending_rows:
        indexed_count += upsert_rows(client, collection_name, pending_rows)
        if expected_documents:
            logger.info("Rebuilt Qdrant index for %s / %s documents", indexed_count, expected_documents)
        else:
            logger.info("Rebuilt Qdrant index for %s documents", indexed_count)
        gc.collect()

    manifest_path = qdrant_path.parent / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "document_count": indexed_count,
                "embedding_model": embedding_model,
                "qdrant_path": str(qdrant_path),
                "vector_size": vector_size,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return indexed_count, vector_size


def upsert_rows(client: object, collection_name: str, rows: list[dict[str, object]]) -> int:
    from qdrant_client.models import PointStruct

    points: list[PointStruct] = []
    for row in rows:
        metadata_json = str(row.get("metadata_json") or "")
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}

        source_type = str(row.get("source_type") or "unknown")
        payload_metadata = {"source_type": source_type, **metadata}
        text = str(row.get("text") or "")
        embedding = [float(value) for value in list(row.get("embedding") or [])]
        document_id = str(row.get("document_id") or "")

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "doc_id": document_id,
                    "text": text[:500],
                    "metadata": payload_metadata,
                },
            )
        )

    count = len(points)
    client.upsert(collection_name=collection_name, points=points)
    del points
    return count


def restore_index_snapshot(index_snapshot_path: Path, *, qdrant_path: Path, force: bool) -> None:
    source_qdrant_dir = index_snapshot_path / "index" / "qdrant_data"
    source_manifest = index_snapshot_path / "index" / "embedding_manifest.json"

    if not source_qdrant_dir.exists():
        raise FileNotFoundError(f"Snapshot has no index/qdrant_data directory: {index_snapshot_path}")

    ensure_can_write(qdrant_path, force=force, description="Qdrant path")
    ensure_can_write(qdrant_path.parent / "embedding_manifest.json", force=force, description="Embedding manifest")

    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_qdrant_dir, qdrant_path)
    if source_manifest.is_file():
        shutil.copy2(source_manifest, qdrant_path.parent / "embedding_manifest.json")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    benchmark_root = resolve_destination_path(args.benchmark_root)
    qdrant_path = resolve_destination_path(args.qdrant_path)
    docs_dir = benchmark_root / "docs"
    bench_dir = benchmark_root / "bench"
    questions_path = bench_dir / "questions.jsonl"
    extra_questions_path = bench_dir / "extra_questions.jsonl"
    embedding_manifest_path = qdrant_path.parent / "embedding_manifest.json"

    logger.info("Restore cwd: %s", Path.cwd())
    logger.info("Restore repo root: %s", _REPO_ROOT)
    logger.info("Restore benchmark root: %s", benchmark_root)
    logger.info("Restore qdrant path: %s", qdrant_path)

    if args.dry_run:
        logger.info("Would restore docs to %s", docs_dir)
        logger.info("Would restore questions to %s", bench_dir)
        if args.index_repo_id:
            logger.info("Would restore original Qdrant files from %s to %s", args.index_repo_id, qdrant_path)
        else:
            logger.info("Would rebuild local Qdrant index at %s", qdrant_path)
        download_snapshot(
            args.repo_id,
            token=args.token,
            revision=args.revision,
            allow_patterns=["train/*.parquet", "bench/*.parquet", "streamable_manifest.json", "README.md"],
            dry_run=True,
        )
        return 0

    if not args.skip_docs:
        ensure_can_write(docs_dir, force=args.force, description="Documents directory")
    if not args.skip_index:
        ensure_can_write(qdrant_path, force=args.force, description="Qdrant path")
        ensure_can_write(embedding_manifest_path, force=args.force, description="Embedding manifest")
    if not args.skip_questions:
        ensure_can_write(questions_path, force=args.force, description="Questions file")
        ensure_can_write(extra_questions_path, force=args.force, description="Extra questions file")

    snapshot_path = download_snapshot(
        args.repo_id,
        token=args.token,
        revision=args.revision,
        allow_patterns=["train/*.parquet", "bench/*.parquet", "streamable_manifest.json", "README.md"],
        dry_run=False,
    )
    if snapshot_path is None:
        return 1

    train_dir = snapshot_path / "train"
    snapshot_bench_dir = snapshot_path / "bench"
    if not train_dir.exists():
        logger.error("Downloaded snapshot has no train/ parquet shards: %s", snapshot_path)
        return 1

    manifest = load_json_file(snapshot_path / "streamable_manifest.json")
    embedding_model = str(manifest.get("embedding_model") or settings.embedding_model)
    expected_documents = int(manifest.get("document_count") or 0) or None

    if not args.skip_docs:
        docs_dir.mkdir(parents=True, exist_ok=True)
        restored_docs = 0
        for row in iter_parquet_rows(train_dir, batch_size=max(1, args.batch_size)):
            write_document_file(docs_dir, row)
            restored_docs += 1
        logger.info("Restored %s document files under %s", restored_docs, docs_dir)

    if not args.skip_questions:
        bench_dir.mkdir(parents=True, exist_ok=True)
        questions_restored = 0
        extra_questions_restored = 0
        questions_parquet = snapshot_bench_dir / "questions.parquet"
        extra_questions_parquet = snapshot_bench_dir / "extra_questions.parquet"
        if questions_parquet.is_file():
            questions_restored = write_jsonl_from_parquet(questions_parquet, questions_path)
        if extra_questions_parquet.is_file():
            extra_questions_restored = write_jsonl_from_parquet(extra_questions_parquet, extra_questions_path)
        logger.info(
            "Restored %s questions and %s extra questions under %s",
            questions_restored,
            extra_questions_restored,
            bench_dir,
        )

    if not args.skip_index:
        if args.index_repo_id:
            index_snapshot = download_snapshot(
                args.index_repo_id,
                token=args.token,
                revision=args.index_revision,
                allow_patterns=["index/qdrant_data/**", "index/embedding_manifest.json"],
                dry_run=False,
            )
            if index_snapshot is None:
                return 1
            restore_index_snapshot(index_snapshot, qdrant_path=qdrant_path, force=args.force)
            logger.info("Restored original Qdrant index files from %s to %s", args.index_repo_id, qdrant_path)
        else:
            indexed_count, vector_size = rebuild_qdrant_index(
                train_dir,
                qdrant_path=qdrant_path,
                embedding_model=embedding_model,
                batch_size=max(1, args.batch_size),
                expected_documents=expected_documents,
            )
            logger.info(
                "Restored Qdrant index with %s documents and vector size %s at %s",
                indexed_count,
                vector_size,
                qdrant_path,
            )

    logger.info("Restore complete from %s", args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
