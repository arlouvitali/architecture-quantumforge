#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def read_documents(input_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(
            {
                "path": path,
                "source": str(path),
                "title": path.stem,
                "text": text,
            }
        )
    return docs


def split_with_positions(text: str, splitter: RecursiveCharacterTextSplitter) -> list[dict[str, Any]]:
    chunks = splitter.split_text(text)
    out: list[dict[str, Any]] = []
    cursor = 0
    for idx, chunk in enumerate(chunks):
        start = text.find(chunk, cursor)
        if start == -1:
            start = text.find(chunk)
        end = start + len(chunk) if start != -1 else -1
        if start != -1:
            cursor = max(end - 120, 0)

        out.append(
            {
                "chunk_index": idx,
                "chunk_text": chunk,
                "start_char": start,
                "end_char": end,
            }
        )
    return out


def build_index(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str,
    local_files_only: bool,
) -> dict[str, Any]:
    started = time.perf_counter()

    docs = read_documents(input_dir)
    if not docs:
        raise RuntimeError(f"No documents found in: {input_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc in docs:
        chunks = split_with_positions(doc["text"], splitter)
        for chunk in chunks:
            texts.append(chunk["chunk_text"])
            metadatas.append(
                {
                    "source": doc["source"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['title']}_{chunk['chunk_index']}",
                    "chunk_index": chunk["chunk_index"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                }
            )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "local_files_only": local_files_only},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))

    duration_sec = time.perf_counter() - started
    embedding_size = len(embeddings.embed_query("index dimension probe"))

    stats = {
        "embedding_model": model_name,
        "embedding_size": embedding_size,
        "documents_count": len(docs),
        "chunks_count": len(texts),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "build_time_seconds": round(duration_sec, 2),
        "input_dir": str(input_dir),
        "index_dir": str(output_dir),
    }

    (output_dir / "index_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return stats


def run_sample_queries(
    index_dir: Path,
    model_name: str,
    device: str,
    local_files_only: bool,
    out_file: Path,
) -> None:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "local_files_only": local_files_only},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(index_dir), embeddings=embeddings, allow_dangerous_deserialization=True
    )

    queries = [
        "Who is Nano Banana and what is his role?",
        "What is Void Core and why was it important?",
        "What happened during Protocol 66?",
    ]

    lines: list[str] = ["# Sample Retrieval Results", ""]

    for q in queries:
        lines.append(f"## Query: {q}")
        results = vectorstore.similarity_search_with_score(q, k=3)
        for i, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata
            snippet = doc.page_content.replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            lines.append(
                f"{i}. score={score:.4f}; source={meta.get('source')}; chunk_id={meta.get('chunk_id')}"
            )
            lines.append(f"   text: {snippet}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG knowledge base.")
    parser.add_argument(
        "--input-dir",
        default="knowledge_base/final",
        help="Directory with source .md/.txt documents.",
    )
    parser.add_argument(
        "--output-dir",
        default="index/faiss_index",
        help="Directory to save FAISS index.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name.",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local model cache (no Hugging Face network requests).",
    )
    parser.add_argument(
        "--run-sample-queries",
        action="store_true",
        help="Run retrieval examples and save output markdown.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    input_dir = (root / args.input_dir).resolve()
    output_dir = (root / args.output_dir).resolve()

    try:
        stats = build_index(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    except OSError as exc:
        print(
            "Failed to load embedding model. "
            "If internet is restricted, run once with network access or use --local-files-only after caching the model.",
            file=sys.stderr,
        )
        raise

    if args.run_sample_queries:
        run_sample_queries(
            index_dir=output_dir,
            model_name=args.model_name,
            device=args.device,
            local_files_only=args.local_files_only,
            out_file=output_dir / "sample_query_results.md",
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
