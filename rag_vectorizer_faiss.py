import os
import json
import torch
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging
from logging.handlers import RotatingFileHandler
import argparse

# ------------------ ARGUMENTS ------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed JSON text chunks and build/append a FAISS index."
    )
    parser.add_argument(
        "--json-folder",
        type=str,
        default="./data/json_output",
        help="Folder containing JSON files with text chunks"
    )
    parser.add_argument(
        "--index-folder",
        type=str,
        default="./data/faiss_index",
        help="Folder to store FAISS index and mapping files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformer model name or local path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Initial batch size for encoding"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of prompts per encode chunk (for OOM safety)"
    )
    return parser.parse_args()

# ------------------ ENV & LOGGING ------------------ #

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_logger(index_folder: str):
    os.makedirs(index_folder, exist_ok=True)
    log_path = os.path.join(index_folder, "vectorization.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_path, maxBytes=10**7, backupCount=3),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ------------------ IO HELPERS ------------------ #

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_pickle(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# ------------------ SAFE ENCODING ------------------ #

def safe_encode_batchwise(model, prompts, batch_size=32, chunk_size=50, logger=None):
    all_embeddings = []
    for i in range(0, len(prompts), chunk_size):
        sub_prompts = prompts[i:i + chunk_size]
        current_batch_size = batch_size
        while current_batch_size >= 1:
            try:
                emb = model.encode(
                    sub_prompts,
                    batch_size=current_batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(emb)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if logger:
                        logger.warning(
                            f"CUDA OOM, chunk_size={len(sub_prompts)}, "
                            f"batch_size={current_batch_size}. Reducing batch size..."
                        )
                    torch.cuda.empty_cache()
                    current_batch_size = current_batch_size // 2
                else:
                    raise e
    return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, model.get_sentence_embedding_dimension()))

# ------------------ MAIN PIPELINE ------------------ #

def embed_and_store(args):
    logger = setup_logger(args.index_folder)
    ensure_dir(args.index_folder)

    processed_record_path = os.path.join(args.index_folder, "processed_files.json")
    pkl_mapping_path = os.path.join(args.index_folder, "text_mapping.pkl")
    faiss_index_path = os.path.join(args.index_folder, "index.faiss")
    failed_files_path = os.path.join(args.index_folder, "failed_files.json")

    processed_files = set(load_json(processed_record_path, []))
    failed_files = set(load_json(failed_files_path, []))
    text_mapping = load_pickle(pkl_mapping_path, default=[])

    logger.info(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    index = faiss.read_index(faiss_index_path) if os.path.exists(faiss_index_path) else None

    files = [f for f in os.listdir(args.json_folder) if f.endswith(".json")]
    new_files = [f for f in files if f not in processed_files]
    logger.info(f"Found {len(new_files)} new files to process in {args.json_folder}.")

    total_new = 0

    for filename in tqdm(new_files, desc="Embedding chunks"):
        file_path = os.path.join(args.json_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 兼容你前一个脚本的输出格式：
            # {
            #   "file_name": "...",
            #   "chunks": [{"id": 1, "text": "..."}, ...]
            # }
            chunk_items = data.get("chunks", [])
            chunks = [c["text"] for c in chunk_items if isinstance(c, dict) and "text" in c]

            if not chunks:
                logger.warning(f"Skip {filename}: no chunks found.")
                continue

            title = data.get("file_name", filename).replace(".json", "")

            # BGE-M3 官方建议使用的中文前缀，可以保留并在 README 中说明
            prompts = ["为这个句子生成表示以用于检索：" + chunk for chunk in chunks]
            entries = [
                {
                    "title": title,
                    "chunk_id": c.get("id", i + 1),
                    "text": c["text"]
                }
                for i, c in enumerate(chunk_items)
                if isinstance(c, dict) and "text" in c
            ]

            embeddings = safe_encode_batchwise(
                model,
                prompts,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                logger=logger
            )
            if embeddings.shape[0] == 0:
                logger.warning(f"Skip {filename}: no embeddings generated.")
                continue

            embeddings = normalize(embeddings)

            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                logger.info(f"Created new FAISS index with dim={dim}")

            index.add(embeddings)
            text_mapping.extend(entries)
            processed_files.add(filename)
            total_new += embeddings.shape[0]

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            failed_files.add(filename)
            torch.cuda.empty_cache()

    if index is not None and total_new > 0:
        faiss.write_index(index, faiss_index_path)
        save_pickle(text_mapping, pkl_mapping_path)
        save_json(list(processed_files), processed_record_path)
        logger.info(f"✅ Added {total_new} new vectors. Total index size: {index.ntotal}")
    else:
        logger.info("No new vectors added.")

    if failed_files:
        save_json(list(failed_files), failed_files_path)
        logger.warning(f"{len(failed_files)} files failed. See failed_files.json for details.")

# ------------------ ENTRYPOINT ------------------ #

if __name__ == "__main__":
    args = parse_args()
    embed_and_store(args)
