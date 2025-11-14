import os
import json
import fitz  # PyMuPDF
import hashlib
import pytesseract
import re
from pdfminer.high_level import extract_text
import pdfplumber
from langdetect import detect
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from PIL import Image
import argparse

# ------------------ CONFIGURATION ------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PDF files into cleaned JSON text chunks.")
    parser.add_argument("--input", type=str, default="./data/pdf_input", help="Path to input folder containing PDF files")
    parser.add_argument("--output", type=str, default="./data/json_output", help="Folder to save JSON output files")
    parser.add_argument("--tesseract", type=str, default="/usr/bin/tesseract", help="Path to Tesseract executable")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=256, help="Maximum chunk size (characters)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Number of overlapping characters between chunks")
    return parser.parse_args()

# ------------------ LOGGING ------------------ #

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ------------------ STRATEGY SELECTOR ------------------ #

class StrategySelector:
    def detect_scanned_pdf(self, text: str, threshold: int = 1000) -> bool:
        return len(text.strip()) < threshold

    def is_complex_layout(self, pdf_path: str) -> bool:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:2]:
                    if len(page.extract_words()) < 5:
                        return True
            return False
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            return True

    def select_strategy(self, pdf_path: str) -> str:
        try:
            text = extract_text(pdf_path)
            if self.detect_scanned_pdf(text):
                return "ocr"
            if self.is_complex_layout(pdf_path):
                return "plumber"
            return "pdfminer"
        except Exception as e:
            logger.error(f"Strategy selection error: {e}")
            return "ocr"

# ------------------ PDF PROCESSOR ------------------ #

class PDFProcessor:
    def __init__(self, chunk_size, overlap):
        self.strategy_selector = StrategySelector()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_file_hash(self, pdf_path):
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def extract_text_pdfminer(self, pdf_path):
        try:
            return extract_text(pdf_path)
        except Exception as e:
            logger.warning(f"PDFMiner failed: {e}")
            return ""

    def extract_text_pdfplumber(self, pdf_path):
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.warning(f"PDFPlumber failed: {e}")
            return ""

    def extract_text_ocr(self, pdf_path):
        text = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                text.append(page_text)
        return "\n".join(text)

    def clean_text(self, text):
        text = re.sub(r'Page\s*\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def split_chunks(self, text):
        paragraphs = text.split("\n\n")
        chunks = []
        sentence_splitter = re.compile(r'(?<=[。！？；.!?])')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                sentences = sentence_splitter.split(para)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= self.chunk_size:
                        temp_chunk += sentence
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = temp_chunk[-self.overlap:] + sentence
                if temp_chunk:
                    chunks.append(temp_chunk.strip())

        return chunks

# ------------------ FILE PROCESSING ------------------ #

def process_pdf(pdf_path, output_dir, chunk_size, overlap):
    processor = PDFProcessor(chunk_size, overlap)
    selector = processor.strategy_selector
    file_name = os.path.basename(pdf_path)
    file_hash = processor.get_file_hash(pdf_path)

    strategy = selector.select_strategy(pdf_path)
    if strategy == "pdfminer":
        text = processor.extract_text_pdfminer(pdf_path)
    elif strategy == "plumber":
        text = processor.extract_text_pdfplumber(pdf_path)
    else:
        text = processor.extract_text_ocr(pdf_path)

    text = processor.clean_text(text)
    language = processor.detect_language(text)
    chunks = processor.split_chunks(text)

    output_json = {
        "file_name": file_name,
        "file_hash": file_hash,
        "language": language,
        "extraction_method": strategy,
        "num_chunks": len(chunks),
        "avg_chunk_length": round(sum(len(c) for c in chunks) / len(chunks), 2) if chunks else 0,
        "chunks": [{"id": i+1, "text": c} for i, c in enumerate(chunks)]
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_name[:-4]}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    logger.info(f"Processed {file_name} with strategy: {strategy}")
    return file_name

# ------------------ BATCH PROCESSING ------------------ #

def run_pdf_to_json_conversion(args):
    pytesseract.pytesseract.tesseract_cmd = args.tesseract
    pdf_files = [os.path.join(root, f)
                 for root, _, files in os.walk(args.input)
                 for f in files if f.lower().endswith('.pdf')]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf, args.output, args.chunk_size, args.chunk_overlap): pdf for pdf in pdf_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing file: {e}")

if __name__ == "__main__":
    args = parse_args()
    run_pdf_to_json_conversion(args)
