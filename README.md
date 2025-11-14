QinHangGeoGPT: A Domain-Specific Geological Language Model for the Qin–Hang Metallogenic Belt

This repository provides the complete reproducible pipeline used to build QinHangGeoGPT, a domain-specific geological large language model integrating:

PDF preprocessing & chunking

FAISS-based semantic vector retrieval (RAG)

Neo4j knowledge graph construction

LoRA-based instruction fine-tuning

Objective & subjective QA evaluation

All code is fully open-source, self-contained, and written to international reproducibility standards to satisfy peer-review requirements.

**🔍 1. Project Overview**

QinHangGeoGPT is a knowledge-enhanced language model designed for:

Geological document understanding

Knowledge graph reasoning

RAG-augmented question answering

Objective (multiple-choice) and subjective (long-form) evaluation

This repository includes the full computational pipeline from PDFs → vectors/KG → fine-tuning → evaluation.

**📁 2. Repository Structure**

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
│      ├── objective_questions_v1.00.xlsx
│      └── subjective_questions_v1.00.xlsx
│
├── requirements.txt
├── LICENSE (MIT)
└── README.md

**⚙️ 3. Installation**
Install Python dependencies
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
PyMuPDF, pdfplumber, pdfminer.six (PDF parsing)
🚀 4. End-to-End Pipeline

This section shows how to reproduce the entire system.

Step 1 — Convert PDFs into JSON text chunks
