"""
evaluate_mc_questions.py

Multiple-choice question (MCQ) evaluation pipeline for QHGeoGPT.

This script:
- Parses questions to extract entities and relation types.
- Queries a geological knowledge graph (Neo4j) for structured triples.
- Retrieves relevant passages from a FAISS-based vector index.
- Builds a context-rich prompt (KG + passages + MCQ) for the LLM.
- Uses a LoRA-augmented base model (e.g., DeepSeek-R1-7B) to answer.
- Compares predicted answers with ground truth and exports results.

Author: (QHGeoGPT team)
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
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ------------------ Logging ------------------ #

def setup_logger(log_file: str = "evaluate_mc_questions.log") -> logging.Logger:
    """Configure simple UTF-8 logging to console + file."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


# ------------------ Question Parsing & KG ------------------ #

class QuestionParser:
    """
    Simple rule-based parser to detect geological entities and relation types
    from a question string, using a predefined entity list and relation dictionary.
    """

    def __init__(self, entity_list, relation_dict):
        self.entity_list = entity_list
        self.relation_dict = relation_dict

    def parse(self, question: str):
        # Remove whitespace to simplify matching (for Chinese text)
        text = re.sub(r"\s+", "", question)
        entity = next((e for e in self.entity_list if e in text), None)
        relation = self.match_relation(text)
        return entity, relation

    def match_relation(self, text: str):
        for key, synonyms in self.relation_dict.items():
            if key in text:
                return key
            for syn in synonyms:
                if syn in text:
                    return key
        return None


class KGQuerier:
    """
    Wrapper around Neo4j driver to query the geological knowledge graph.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_by_relation_type(self, entity_name: str, relation_type: str):
        """
        Returns triples of the form:
        (source) -[relation_type]-> (target)
        """
        cypher = """
        MATCH (n {name: $name})-[r]->(m)
        WHERE type(r) = $rel
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name, rel=relation_type)
            return [dict(record) for record in result]


# ------------------ Vector Retrieval (FAISS + BGE-M3) ------------------ #

def load_text_mapping(mapping_path: str):
    """Load text mapping (list of dicts) from a pickle file."""
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return mapping


def retrieve_passages(
    question: str,
    embedder: SentenceTransformer,
    faiss_index: faiss.Index,
    text_mapping,
    top_k: int = 5,
):
    """
    Encode the question using the same embedding model used to build the FAISS index
    (e.g., BGE-M3) and retrieve top-k most similar text chunks.
    """
    # BGE-M3 recommendation: use a Chinese retrieval prefix
    query = "为这个句子生成表示以用于检索：" + question
    emb = embedder.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(emb, top_k)
    indices = I[0]
    passages = []
    for idx in indices:
        if 0 <= idx < len(text_mapping):
            passages.append(text_mapping[idx]["text"])
    return passages


# ------------------ Prompt Builder ------------------ #

def build_prompt_glm_style(question, options, triples, passages):
    """
    Build a Chinese instruction prompt for the geological MCQ evaluation,
    combining knowledge graph triples and retrieved passages.

    NOTE: The prompt is intentionally kept in Chinese to match the original
    experimental setting of QHGeoGPT.
    """
    option_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

    triple_str = "\n".join(
        [f"{t['source']} --{t['relation']}--> {t['target']}" for t in triples]
    ) or "（无相关图谱知识）"

    passage_str = "\n".join([f"- {p}" for p in passages]) or "（无检索到的文献信息）"

    prompt = f"""你是一个矿床地质专家，请结合以下参考资料判断选择题的正确答案：

-------------------------
【图谱知识】
{triple_str}

【文献信息】
{passage_str}
-------------------------

题目：{question}

选项：
{option_text}

要求：
1. 必须严格基于参考资料回答。
2. 最终只输出一个选项对应的字母（A、B、C、D），不要输出其他文字。
3. 不要输出“解析”或说明理由。
4. 不允许编造答案，如果资料不足，请根据最有可能的信息选择。

请直接输出答案字母（A、B、C或D）：
答案："""
    return prompt


# ------------------ LLM Loader (Base + LoRA) ------------------ #

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


# ------------------ Answer Extraction ------------------ #

def extract_answer(answer_text: str) -> str:
    """
    Extract the final answer letter A-D from model output.
    If multiple matches, return the last one; otherwise return '?'.
    """
    all_matches = re.findall(r"[答題题案][:：]?\s*([A-D])", answer_text, flags=re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()
    # Fallback: pure A-D search
    all_matches = re.findall(r"\b([A-D])\b", answer_text, flags=re.IGNORECASE)
    return all_matches[-1].upper() if all_matches else "?"


# ------------------ MCQ Evaluation ------------------ #

def evaluate_mc_questions(
    excel_path: str,
    output_path: str,
    parser: QuestionParser,
    kg: KGQuerier,
    embedder: SentenceTransformer,
    faiss_index: faiss.Index,
    text_mapping,
    model,
    tokenizer,
    logger: logging.Logger,
):
    """
    Evaluate multiple-choice questions stored in an Excel file.

    Expected columns:
    - '编号'       : Question ID
    - 'Question' : Question text
    - 'Option A' : Option A
    - 'Option B' : Option B
    - 'Option C' : Option C
    - 'Option D' : Option D
    - 'Answer'   : Ground truth answer letter (A/B/C/D)
    """
    df = pd.read_excel(excel_path)
    results = []

    logger.info(f"Loaded {len(df)} questions from {excel_path}")

    for _, row in df.iterrows():
        qid = row.get("编号", "")
        question = row["Question"]
        options = [
            row["Option A"],
            row["Option B"],
            row["Option C"],
            row["Option D"],
        ]
        correct = str(row.get("Answer", "")).strip().upper()

        # 1. Parse entity & relation
        entity, relation = parser.parse(question)
        triples = []
        if entity and relation:
            try:
                triples = kg.query_by_relation_type(entity, relation)
            except Exception as e:
                logger.warning(f"KG query failed for QID={qid}: {e}")

        # 2. Retrieve passages
        passages = retrieve_passages(
            question,
            embedder=embedder,
            faiss_index=faiss_index,
            text_mapping=text_mapping,
            top_k=5,
        )

        # 3. Build prompt
        prompt = build_prompt_glm_style(question, options, triples, passages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 4. Deterministic generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.1,
            )

        answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_answer(answer_text)
        is_correct = (predicted == correct)

        results.append(
            {
                "编号": qid,
                "题目": question,
                "预测答案": predicted,
                "标准答案": correct,
                "是否正确": is_correct,
                "模型原始输出": answer_text,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_excel(output_path, index=False)
    logger.info(f"Evaluation finished. Results saved to: {output_path}")


# ------------------ CLI & Main ------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MCQs using QHGeoGPT (LoRA) + KG + RAG."
    )

    # Question file & output
    parser.add_argument(
        "--questions-xlsx",
        type=str,
        required=True,
        help="Path to the Excel file containing MCQ questions.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default="./mcq_evaluation_results.xlsx",
        help="Path to save the evaluation results (Excel).",
    )

    # KG dictionaries & Neo4j
    parser.add_argument(
        "--entity-list",
        type=str,
        required=True,
        help="Path to JSON file containing entity list.",
    )
    parser.add_argument(
        "--relation-dict",
        type=str,
        required=True,
        help="Path to JSON file containing relation synonym dictionary.",
    )
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
        default="password",
        help="Neo4j password (for local dev only; use env vars in production).",
    )

    # Vector index & embedding model
    parser.add_argument(
        "--faiss-index",
        type=str,
        required=True,
        help="Path to the FAISS index file (e.g., index.faiss).",
    )
    parser.add_argument(
        "--text-mapping",
        type=str,
        required=True,
        help="Path to text_mapping.pkl (list of dicts with 'text' field).",
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
        help="Base model path or HF ID for causal LM (e.g., DeepSeek-R1-7B).",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("Starting MCQ evaluation pipeline for QHGeoGPT.")

    # 1. Load entity list & relation dict
    with open(args.entity_list, "r", encoding="utf-8") as f:
        entity_list = json.load(f)
    with open(args.relation_dict, "r", encoding="utf-8") as f:
        relation_dict = json.load(f)

    parser = QuestionParser(entity_list, relation_dict)

    # 2. Connect to Neo4j
    kg = KGQuerier(args.kg_uri, args.kg_user, args.kg_password)
    logger.info(f"Connected to Neo4j at {args.kg_uri}")

    # 3. Load FAISS index & text mapping
    logger.info(f"Loading FAISS index from {args.faiss_index}")
    faiss_index = faiss.read_index(args.faiss_index)

    logger.info(f"Loading text mapping from {args.text_mapping}")
    text_mapping = load_text_mapping(args.text_mapping)

    # 4. Load embedding model
    logger.info(f"Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model, device="cuda" if torch.cuda.is_available() else "cpu")

    # 5. Load base model + LoRA
    logger.info(f"Loading base model + LoRA: {args.base_model}, {args.lora_path}")
    model, tokenizer = load_lora_model(args.base_model, args.lora_path)

    # 6. Run evaluation
    evaluate_mc_questions(
        excel_path=args.questions_xlsx,
        output_path=args.output_xlsx,
        parser=parser,
        kg=kg,
        embedder=embedder,
        faiss_index=faiss_index,
        text_mapping=text_mapping,
        model=model,
        tokenizer=tokenizer,
        logger=logger,
    )

    # 7. Cleanup
    kg.close()
    logger.info("All done.")


if __name__ == "__main__":
    main()
