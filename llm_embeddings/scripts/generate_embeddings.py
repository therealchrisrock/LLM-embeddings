import time
from typing import Dict, Any

import openai
from openai import RateLimitError, APIConnectionError

import json
from dotenv import load_dotenv

from definitions import ROOT_DIR

# Load API key from .env file
load_dotenv()
# ---------------------------
# Configuration
# ---------------------------

# Path to your JSON file
JSON_FILE_PATH = f'{ROOT_DIR}/data/chunks/all_chunks.json'  # Update this path if your file is located elsewhere

# Embedding Model Configuration
EMBEDDING_MODEL = 'text-embedding-ada-002'  # Update based on the model you intend to use

# ---------------------------
# Functions
# ---------------------------

def load_chunks(json_file_path):
    """
    Load chunks from a JSON file.

    :param json_file_path: Path to the JSON file.
    :return: List of chunks.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            chunks = json.load(file)
        if not isinstance(chunks, list):
            raise ValueError("JSON content must be a list of text chunks.")
        return chunks
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    except ValueError as ve:
        print(f"Value Error: {ve}")
        return []




def create_embedding(
        text:  Dict[str, Any],
        model: str = "text-embedding-ada-002",
        max_retries: int = 5,
        base_delay: float = 1.0
) -> list[float]:
    """
    Create an embedding for the provided text with exponential backoff on errors.
    """
    backoff = base_delay
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model=model,
                input=text['chunk']
            )
            data = response.data  # This is a list-like object
            if not data:
                print("No embedding data found.")
            # Access the 'embedding' attribute of the first item
            embedding = data[0].embedding
            return embedding
        except RateLimitError as e:
            # Handle 429 rate limit errors
            print(f"[RateLimitError] Attempt {attempt+1}/{max_retries}. "
                  f"Waiting {backoff:.2f} seconds before retry.")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
        except APIConnectionError as e:
            # Handle transient connection errors
            print(f"[ConnectionError] Attempt {attempt+1}/{max_retries}. "
                  f"Waiting {backoff:.2f} seconds before retry.")
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            # Some other error that we don't want to keep retrying
            print(f"[Error] {e}")
            raise e
    raise Exception(f"Failed to create embedding after {max_retries} retries.")

def generate_embeddings(chunks: list[str]) -> list[dict]:
    """
    Generate embeddings for all chunks using the create_embedding function.
    Includes optional rate-limiting logic between each request.
    """
    embeddings = []

    for i, c in enumerate(chunks):
        # --- Optional Rate Limiting Sleep ---
        # If you want to enforce a pause between every request (e.g., 200 ms),
        # you can uncomment the following line:
        # time.sleep(0.2)

        # Create the embedding with retry/backoff
        try:
            embedding = create_embedding(c)
            embeddings.append({"chunk": c["chunk"], "embedding": embedding})
            print(f"{i} / {len(chunks)} chunks embedded")
        except Exception as e:
            print(f"Failed to process chunk index {i}. Error: {e}")
            # Depending on your needs, you can break, continue, or re-raise
            raise e

    return embeddings

if __name__ == "__main__":
    chunks = load_chunks(JSON_FILE_PATH)

    embeddings = generate_embeddings(chunks)

    # Save embeddings as JSON
    with open(f"{ROOT_DIR}/data/embeddings/templ_embeddings.json", "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved for {len(embeddings)} chunks.")