\# QHGeoGPT: Geological Domain LLM for the Qin–Hang Metallogenic Belt



\## Overview

\- One paragraph about the goal of the project.

\- High-level description of the pipeline (PDF → Text → Vectors/KG → LoRA → Evaluation).



\## Repository Structure

\- Briefly list each script:

&nbsp; - `pdf\_to\_chunks.py` – PDF preprocessing and text chunking

&nbsp; - `vectorize\_faiss.py` – Build FAISS index from JSON chunks

&nbsp; - `build\_kg\_from\_excel.py` – Construct Neo4j KG from Excel triples

&nbsp; - `train\_lora.py` – LoRA fine-tuning pipeline

&nbsp; - `evaluate\_objective.py` – Objective MCQ evaluation

&nbsp; - `evaluate\_subjective.py` – Subjective question evaluation



\## Installation

\- `git clone ...`

\- `pip install -r requirements.txt`



\## End-to-End Usage

1\. Run `pdf\_to\_chunks.py` to create JSON text chunks.

2\. Run `vectorize\_faiss.py` to build the FAISS index.

3\. Run `build\_kg\_from\_excel.py` to construct the KG in Neo4j.

4\. Run `train\_lora.py` to fine-tune the model (optional if you release weights).

5\. Run `evaluate\_objective.py` and `evaluate\_subjective.py` to reproduce the experiments.



\## Data Format

\- JSON chunk format

\- Excel KG format

\- Evaluation Excel format (columns)



\## Citation

\- Your paper citation.



\## License

\- MIT License note.



