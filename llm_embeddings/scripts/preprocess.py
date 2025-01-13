from pathlib import Path

from tiktoken import get_encoding

from definitions import ROOT_DIR

import json


def store_chunks_single_file(chunks, output_file):
    """Store all chunks in a single JSON file."""
    # Convert any Path objects to strings
    for chunk in chunks:
        if "source" in chunk and isinstance(chunk["source"], Path):
            chunk["source"] = str(chunk["source"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Stored {len(chunks)} chunks in {output_file}")


def collect_markdown_files(root_dir: str):
    """Collect all Markdown files under the root directory, ignoring _category.json files."""
    root_path = Path(root_dir)
    markdown_files = []

    # Iterate over all .md files, ignoring "_category.json"
    for file_path in root_path.rglob("*.md"):
        if file_path.name != "_category.json":
            markdown_files.append(file_path)

    return markdown_files


def preprocess_text(file_path, max_tokens=500):
    """Splits large text into smaller chunks."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


if __name__ == "__main__":
    # Example usage
    all_chunks = []  # Collect chunks from multiple files
    doc_root = f"{ROOT_DIR}/data/docs_raw"
    markdown_files = collect_markdown_files(doc_root)
    if not markdown_files:
        print("No Markdown files found.")
        exit(1)

    for file_path in markdown_files:
        chunks = preprocess_text(file_path)
        for chunk in chunks:
            all_chunks.append({"source": file_path, "chunk": chunk})

    # Store all chunks in a single file
    store_chunks_single_file(all_chunks, f"{ROOT_DIR}/data/chunks/all_chunks.json")
