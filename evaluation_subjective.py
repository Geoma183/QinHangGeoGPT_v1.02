"""
generate_subjective_answers.py

Subjective question answering pipeline for QHGeoGPT.

This script:
- Parses questions to infer geological entities and relation types (entity+relation parser).
- Queries a geological knowledge graph (Neo4j) for single-hop and multi-hop triples.
- Retrieves relevant passages from a FAISS-based RAG index (e.g., BGE-M3 + text_mapping).
- Builds a Chinese, high-academic-style prompt for subjective geological questions.
- Uses a LoRA-augmented base model (e.g., DeepSeek-R1-7B) to generate detailed answers.
- Saves generated answers and context traces to an Excel file.

Author: QHGeoGPT team
License: MIT
"""

import os
import sys
import re
import json
import pickle
import logging
import argparse

import faiss
import torch
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ------------------ Logging ------------------ #

def setup_logger(output_dir: str, log_name: str = "subjective_eval.log") -> logging.Logger:
    """Configure simple UTF-8 logging to console + file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


# ------------------ Question Parsing & KG ------------------ #

class QuestionParser:
    """
    Entity + relation parser combining:
    - Exact match (name and aliases)
    - Fuzzy match (Levenshtein similarity via fuzzywuzzy)
    - Vector-based match (SentenceTransformer + FAISS index)

    entity_structured: list of dicts, e.g.:
        [{"name": "Qin-Hang belt", "aliases": ["QH belt", "Qin–Hang"]}, ...]
    entity_index: FAISS index for entity texts
    entity_mapping: list of canonical entity names, aligned with FAISS index IDs

    relation_structured / relation_index / relation_mapping: analogous for relations.
    """

    def __init__(
        self,
        entity_structured,
        relation_structured,
        vec_model: SentenceTransformer,
        entity_index: faiss.Index,
        entity_mapping,
        relation_index: faiss.Index,
        relation_mapping,
    ):
        self.entity_structured = entity_structured
        self.relation_structured = relation_structured
        self.vec_model = vec_model
        self.entity_index = entity_index
        self.entity_mapping = entity_mapping
        self.relation_index = relation_index
        self.relation_mapping = relation_mapping

    @staticmethod
    def fuzzy_match(name, aliases, text, threshold=80):
        candidates = [name] + (aliases or [])
        best_match = process.extractOne(text, candidates)
        if best_match and best_match[1] >= threshold:
            return best_match[0]
        return None

    def match_entity_combined(self, text: str):
        # 1) Exact / alias match
        for item in self.entity_structured:
            name = item.get("name", "")
            aliases = item.get("aliases", [])
            if name and name in text:
                return name
            if any(alias in text for alias in aliases):
                return name

        # 2) Fuzzy match
        for item in self.entity_structured:
            name = item.get("name", "")
            aliases = item.get("aliases", [])
            matched = self.fuzzy_match(name, aliases, text)
            if matched:
                return name

        # 3) Vector match
        vec = self.vec_model.encode([text], normalize_embeddings=True)
        D, I = self.entity_index.search(np.array(vec, dtype=np.float32), 1)
        idx = int(I[0][0])
        if 0 <= idx < len(self.entity_mapping):
            return self.entity_mapping[idx]
        return None

    def match_relation_combined(self, text: str):
        # 1) Exact / alias match
        for item in self.relation_structured:
            name = item.get("name", "")
            aliases = item.get("aliases", [])
            if name and name in text:
                return name
            if any(alias in text for alias in aliases):
                return name

        # 2) Fuzzy match
        for item in self.relation_structured:
            name = item.get("name", "")
            aliases = item.get("aliases", [])
            matched = self.fuzzy_match(name, aliases, text)
            if matched:
                return name

        # 3) Vector match
        vec = self.vec_model.encode([text], normalize_embeddings=True)
        D, I = self.relation_index.search(np.array(vec, dtype=np.float32), 1)
        idx = int(I[0][0])
        if 0 <= idx < len(self.relation_mapping):
            return self.relation_mapping[idx]
        return None

    def parse(self, question: str):
        # Remove whitespace (especially for Chinese)
        text = re.sub(r"\s+", "", question)
        entity = self.match_entity_combined(text)
        relation = self.match_relation_combined(text)
        return entity, relation


class KGQuerier:
    """
    Graph database wrapper for Neo4j.

    Supports:
    - query_by_relation_type: single-hop triples
    - query_multi_hop: multi-hop patterns up to N hops
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_by_relation_type(self, entity_name: str, relation_type: str):
        cypher = """
        MATCH (n {name: $name})-[r]->(m)
        WHERE type(r) = $rel
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name, rel=relation_type)
            return [dict(record) for record in result]

    def query_multi_hop(self, entity_name: str, relation_keyword: str, max_hops: int = 2, limit: int = 10):
        """
        Multi-hop query: paths with length 1..max_hops where at least one relation
        along the path contains the given keyword.
        """
        cypher = f"""
        MATCH p = (n)-[r*1..{max_hops}]->(m)
        WHERE n.name CONTAINS $name
          AND any(rel IN relationships(p) WHERE type(rel) CONTAINS $rel)
        RETURN n.name AS source, type(relationships(p)[0]) AS relation, m.name AS target
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name, rel=relation_keyword, limit=limit)
            return [dict(record) for record in result]


# ------------------ RAG Utilities ------------------ #

def load_text_mapping(mapping_path: str):
    """
    Load RAG mapping (list of dicts) from a pickle file.
    Expected structure similar to:
        [{"title": "...", "text": "..."}, ...]
    """
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return mapping


def build_rag_context(
    question: str,
    embedder: SentenceTransformer,
    faiss_index: faiss.Index,
    text_mapping,
    top_k: int = 4,
    max_docs: int = 3,
    max_text_len: int = 200,
):
    """
    Retrieve top-k most similar chunks and build a textual RAG context.
    """
    query = "为这个句子生成表示以用于检索：" + question
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(vec, dtype=np.float32), top_k)

    rag_context_lines = []
    ids = I[0][:max_docs]
    for idx in ids:
        if 0 <= idx < len(text_mapping):
            item = text_mapping[idx]
            title = item.get("title", "Unknown Source")
            text = item.get("text", "")
            snippet = text[:max_text_len].replace("\n", " ")
            rag_context_lines.append(f"{title}: {snippet}...")
    return "\n".join(rag_context_lines), len(ids)


# ------------------ Prompt & Answer Processing ------------------ #

def build_subjective_prompt(question, graph_context=None, rag_context=None):
    graph_info = f"以下是从知识图谱中查询到的相关信息：\n{graph_context}\n" if graph_context else ""
    rag_info = f"以下是RAG检索到的相关文献：\n{rag_context}\n" if rag_context else ""

    prompt = f"""请针对以下主观题提供一个详细、准确、具备高学术水准和专业深度的答案。严禁随意编造年代或数值。确保回答完全、详细，不遗漏任何关键信息。

问题：{question}

{graph_info}{rag_info}
要求：
1. 答案需以一个完整连续的文本段落呈现，不分段、不使用列举或序号。
2. 使用严谨的学术语句和专业术语，逻辑严密，条理清晰。
3. 不在答案中包含参考文献编号或显式引用标记。
4. 避免使用诸如“根据研究”“文献指出”等字样，直接陈述事实和结论。
5. 尽可能完整地覆盖与问题相关的关键地质过程、构造背景和成矿机制。

答案：
"""
    return prompt


def extract_pure_answer(full_text: str) -> str:
    """
    Extract answer segment after the first occurrence of '答案:' / '答案：'.
    If no such marker is found, return the whole text (trimmed).
    """
    pattern = r"答案[:：]\s*(.*)"
    match = re.search(pattern, full_text, re.DOTALL)
    if match:
        answer_part = match.group(1).strip()
    else:
        answer_part = full_text.strip()

    # Remove common special tags / tokens
    answer_part = re.sub(r"</think>\s*", "", answer_part)
    answer_part = re.sub(r"<\|endoftext\|>|<\｜end▁of▁sentence\｜>|</s>", "", answer_part)
    # Normalize whitespace
    answer_part = re.sub(r"\s+", " ", answer_part)
    return answer_part.strip()


def refine_answer(answer: str) -> str:
    """
    Light post-processing for readability, without changing the semantic content.
    - Remove obvious duplicated segments.
    - Remove meta-phrases about "this question".
    - Normalize whitespace.
    """
    # Remove repeated consecutive segments (very rough heuristic)
    answer = re.sub(r"(.*?)(\1)+", r"\1", answer)

    # Remove meta phrases about "this question"
    answer = re.sub(r"这个问题的答案|这个问题", "", answer)

    # Collapse whitespace
    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()


# ------------------ LLM Loader ------------------ #

def load_lora_model(base_model_path: str, lora_path: str):
    """
    Load a 4-bit quantized base model (e.g., DeepSeek-R1-7B) and
    attach a LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_use_nested_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer


def generate_answer(
    question: str,
    model,
    tokenizer,
    graph_context: str = None,
    rag_context: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.9,
    logger: logging.Logger = None,
):
    """
    Build the subjective prompt, call the model, and post-process the answer.
    """
    prompt = build_subjective_prompt(question, graph_context, rag_context)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if logger:
        logger.info(f"Raw model output:\n{full_output}")

    pure_answer = extract_pure_answer(full_output)
    refined_answer = refine_answer(pure_answer)
    return prompt, refined_answer, full_output


# ------------------ Main Evaluation Loop ------------------ #

def process_questions_and_generate_answers(
    excel_path: str,
    model,
    tokenizer,
    kg: KGQuerier,
    faiss_index: faiss.Index,
    text_mapping,
    parser: QuestionParser,
    embedder: SentenceTransformer,
    logger: logging.Logger,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """
    Read subjective questions from Excel and generate answers.

    Expected columns:
    - 'Question' : question text
    Optional columns:
    - 'ID' or '编号' : question identifier
    """
    df = pd.read_excel(excel_path)
    results = []

    logger.info(f"Loaded {len(df)} questions from {excel_path}")

    for _, row in df.iterrows():
        try:
            question = row["Question"]
            qid = row.get("ID", row.get("编号", ""))

            # 1) Parse entity & relation
            entity, relation = parser.parse(question)
            graph_context = ""
            graph_process = f"Entity: {entity}, Relation: {relation}"

            triples = []
            if entity and relation:
                try:
                    triples = kg.query_by_relation_type(entity, relation)
                    if not triples:
                        # fallback to multi-hop if direct relation not found
                        triples = kg.query_multi_hop(entity, relation, max_hops=2, limit=10)
                        graph_process += f" | Multi-hop query used."
                except Exception as e:
                    logger.warning(f"KG query failed for QID={qid}: {e}")

            if triples:
                triples = triples[:5]
                graph_context = "\n".join(
                    [f"{t['source']} --{t['relation']}--> {t['target']}" for t in triples]
                )

            # 2) RAG context
            rag_context, retrieved_docs = build_rag_context(
                question,
                embedder=embedder,
                faiss_index=faiss_index,
                text_mapping=text_mapping,
                top_k=4,
                max_docs=3,
                max_text_len=200,
            )
            rag_process = f"RAG retrieval: {retrieved_docs} candidate chunks."

            # 3) Generate answer
            prompt, generated_answer, raw_output = generate_answer(
                question,
                model=model,
                tokenizer=tokenizer,
                graph_context=graph_context,
                rag_context=rag_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logger=logger,
            )

            results.append(
                {
                    "ID": qid,
                    "Question": question,
                    "Generated Answer": generated_answer,
                    "Generation Trace": f"{graph_process}\n{rag_process}\n\n[Prompt]\n{prompt}\n\n[KG]\n{graph_context}\n\n[RAG]\n{rag_context}",
                    "Raw Model Output": raw_output,
                }
            )

        except Exception as e:
            logger.error(f"Error processing question '{row.get('Question', '')}': {e}")
            results.append(
                {
                    "ID": row.get("ID", row.get("编号", "")),
                    "Question": row.get("Question", ""),
                    "Generated Answer": "Error",
                    "Generation Trace": f"Error: {e}",
                    "Raw Model Output": "",
                }
            )

    return results


def save_results_to_excel(results, output_path: str, logger: logging.Logger):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")


# ------------------ CLI & Main ------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate subjective answers using QHGeoGPT (LoRA) + KG + RAG."
    )

    # Questions & output
    parser.add_argument(
        "--questions-xlsx",
        type=str,
        required=True,
        help="Path to the Excel file containing subjective questions.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default="./subjective_answers_qhgeogpt.xlsx",
        help="Path to save the generated answers (Excel).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory for logs and intermediate outputs.",
    )

    # KG & parsing
    parser.add_argument(
        "--entity-structured",
        type=str,
        required=True,
        help="Path to JSON file with structured entity list (name + aliases).",
    )
    parser.add_argument(
        "--relation-structured",
        type=str,
        required=True,
        help="Path to JSON file with structured relation list (name + aliases).",
    )
    parser.add_argument(
        "--entity-index",
        type=str,
        required=True,
        help="Path to FAISS index for entities.",
    )
    parser.add_argument(
        "--relation-index",
        type=str,
        required=True,
        help="Path to FAISS index for relations.",
    )
    parser.add_argument(
        "--entity-mapping",
        type=str,
        required=True,
        help="Path to JSON or pickle file mapping entity index -> entity name.",
    )
    parser.add_argument(
        "--relation-mapping",
        type=str,
        required=True,
        help="Path to JSON or pickle file mapping relation index -> relation name.",
    )

    # Neo4j
    parser.add_argument(
        "--kg-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j bolt URI.",
    )
    parser.add_argument(
        "--kg-user",
        type=str,
        default="neo4j",
        help="Neo4j username.",
    )
    parser.add_argument(
        "--kg-password",
        type=str,
        default=None,
        help="Neo4j password. If not provided, NEO4J_PASSWORD env var will be used.",
    )

    # RAG
    parser.add_argument(
        "--faiss-index",
        type=str,
        required=True,
        help="Path to the FAISS index file for RAG (e.g., index.faiss).",
    )
    parser.add_argument(
        "--text-mapping",
        type=str,
        required=True,
        help="Path to text_mapping.pkl used for RAG (title + text).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformer model name or local path for retrieval embeddings.",
    )

    # LLM + LoRA
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base causal LM path or HF ID (e.g., DeepSeek-R1-7B).",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory.",
    )

    # Generation hyperparameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling for generation.",
    )

    return parser.parse_args()


def load_mapping_auto(path: str):
    """
    Load mapping from JSON or pickle automatically.
    - JSON: expect list or dict
    - PKL : expect list
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)

    # Normalize to list of names (index -> name)
    if isinstance(data, dict):
        # assume {"0": "name0", "1": "name1", ...}
        # sort by key to get index order
        items = sorted(data.items(), key=lambda x: int(x[0]))
        return [v for _, v in items]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported mapping format in {path}")


def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)

    logger.info("Starting subjective question answering pipeline for QHGeoGPT.")

    # 1. Neo4j password
    kg_password = args.kg_password or os.getenv("NEO4J_PASSWORD")
    if not kg_password:
        raise ValueError("Neo4j password not provided. Use --kg-password or NEO4J_PASSWORD env var.")

    # 2. Load structured entity / relation definitions
    with open(args.entity_structured, "r", encoding="utf-8") as f:
        entity_structured = json.load(f)
    with open(args.relation_structured, "r", encoding="utf-8") as f:
        relation_structured = json.load(f)

    # 3. Load FAISS indexes and mappings for entity / relation parsing
    logger.info(f"Loading entity FAISS index from {args.entity_index}")
    entity_index = faiss.read_index(args.entity_index)
    logger.info(f"Loading relation FAISS index from {args.relation_index}")
    relation_index = faiss.read_index(args.relation_index)

    logger.info(f"Loading entity mapping from {args.entity_mapping}")
    entity_mapping = load_mapping_auto(args.entity_mapping)
    logger.info(f"Loading relation mapping from {args.relation_mapping}")
    relation_mapping = load_mapping_auto(args.relation_mapping)

    # 4. SentenceTransformer for vector operations
    logger.info(f"Loading embedding model for parsing and RAG: {args.embed_model}")
    embedder = SentenceTransformer(
        args.embed_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 5. Build QuestionParser
    parser_obj = QuestionParser(
        entity_structured=entity_structured,
        relation_structured=relation_structured,
        vec_model=embedder,
        entity_index=entity_index,
        entity_mapping=entity_mapping,
        relation_index=relation_index,
        relation_mapping=relation_mapping,
    )

    # 6. Connect to Neo4j
    kg = KGQuerier(args.kg_uri, args.kg_user, kg_password)
    logger.info(f"Connected to Neo4j at {args.kg_uri}")

    # 7. Load RAG index and mapping
    logger.info(f"Loading RAG FAISS index from {args.faiss_index}")
    rag_index = faiss.read_index(args.faiss_index)

    logger.info(f"Loading RAG text mapping from {args.text_mapping}")
    text_mapping = load_text_mapping(args.text_mapping)

    # 8. Load base LM + LoRA adapter
    logger.info(f"Loading base model + LoRA: {args.base_model}, {args.lora_path}")
    model, tokenizer = load_lora_model(args.base_model, args.lora_path)

    # 9. Run evaluation
    results = process_questions_and_generate_answers(
        excel_path=args.questions_xlsx,
        model=model,
        tokenizer=tokenizer,
        kg=kg,
        faiss_index=rag_index,
        text_mapping=text_mapping,
        parser=parser_obj,
        embedder=embedder,
        logger=logger,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # 10. Save results
    save_results_to_excel(results, args.output_xlsx, logger)

    # 11. Cleanup
    kg.close()
    logger.info("All done.")


if __name__ == "__main__":
    main()
