QinHangGeoGPT: A Domain-Specific Geological Language Model for the Qin–Hang Metallogenic Belt

This repository provides the complete reproducible pipeline used to build QinHangGeoGPT, a domain-specific geological large language model integrating:

PDF preprocessing & chunking

FAISS-based semantic vector retrieval (RAG)

Neo4j knowledge graph construction

LoRA-based instruction fine-tuning

Objective & subjective QA evaluation

All code is fully open-source, self-contained, and written to international reproducibility standards for peer-review.

📘 1. Project Overview

QinHangGeoGPT is a knowledge-enhanced language model designed for:

Geological document understanding

Knowledge graph reasoning

RAG-augmented question answering

Objective (multiple-choice) and subjective (long-form) evaluation

This repository includes the full computational pipeline from:

PDFs → text chunks → vector retrieval → knowledge graph → model fine-tuning → evaluation

📁 2. Repository Structure
QinHangGeoGPT_v1.02/
│
├── preprocess_pdf_chunks.py           # PDF → JSON text chunks
├── rag_vectorizer_faiss.py            # JSON chunks → FAISS vector index
├── build_qh_kg_from_excel.py          # Excel → Neo4j knowledge graph
├── train_qhgeogpt_lora.py             # LoRA instruction fine-tuning
├── evaluation_objective.py            # Multiple-choice evaluation
├── evaluation_subjective.py           # Subjective QA evaluation
│
├── data/
│   ├── eval/
│   │   ├── objective_questions_v1.00.xlsx
│   │   └── subjective_questions_v1.00.xlsx
│   └── kg/
│       └── qh_kg_demo.xlsx
│
├── requirements.txt
├── LICENSE
└── README.md

⚙️ 3. Installation

Install Python dependencies:

pip install -r requirements.txt


Required libraries include:

torch

transformers

peft

bitsandbytes

sentence-transformers

faiss-cpu / faiss-gpu

pandas

openpyxl

neo4j

pdfplumber / PyMuPDF / pdfminer.six

🚀 4. End-to-End Pipeline
Step 1 — Convert PDFs into JSON text chunks
python preprocess_pdf_chunks.py \
  --input ./data/pdf_input \
  --output ./data/json_output

Step 2 — Build or update FAISS vector index
python rag_vectorizer_faiss.py \
  --json-folder ./data/json_output \
  --index-folder ./data/faiss_index \
  --model BAAI/bge-m3 \
  --batch-size 32 \
  --chunk-size 50


Outputs:

index.faiss

text_mapping.pkl

processed_files.json

failed_files.json

Step 3 — Build Neo4j Knowledge Graph
python build_qh_kg_from_excel.py \
  --xlsx ./data/kg/qh_kg_demo.xlsx \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password your_password

Step 4 — LoRA Instruction Fine-Tuning
python train_qhgeogpt_lora.py \
  --data ./data/final_unified_instruction_finetune.jsonl \
  --base-model DeepSeek-R1-Distill-Qwen-7B \
  --output ./models/qhgeogpt_lora

Step 5 — Objective QA Evaluation
python evaluation_objective.py \
  --questions-xlsx ./data/eval/objective_questions_v1.00.xlsx \
  --faiss-index ./data/faiss_index/index.faiss \
  --text-mapping ./data/faiss_index/text_mapping.pkl \
  --entity-list ./data/kg/entity_list.json \
  --relation-dict ./data/kg/relation_dict.json

Step 6 — Subjective QA Evaluation
python evaluation_subjective.py \
  --questions-xlsx ./data/eval/subjective_questions_v1.00.xlsx

📊 5. External Data

Due to licensing restrictions, original geological PDFs and the full KG database cannot be redistributed.

This repository provides:

Example KG Excel

Evaluation datasets

Complete scripts for reconstruction

📜 6. License

Released under the MIT License.

📚 7. Citation
Cai, B. et al. (2025).
QinHangGeoGPT: A Domain-Specific Large Language Model for Metallogenic Belt Knowledge Reasoning.
Under Review.

👨‍💻 8. Maintainer

Author: Dr. [Your Name]
Sun Yat-sen University
Email: your_email@sysu.edu.cn
