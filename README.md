
# LayoutLMv3 Token Classification for Data Prep + Visualizer

End-to-end pipeline to go from **PDF** → **OCR (tokens + 2D boxes)** → **rule-based labels** → **overlapping chunks (512, stride 128)** using **LayoutLMv3Processor** → **visual highlighter** over labeled tokens.


---

## 1) Quickstart

```bash
# 1) Create venv
python -m venv .venv && .venv\Scripts\activate   # Windows

# 2) Install system deps
# - Tesseract OCR:
#   - Windows: 
#       - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
#       - Install Poppler:  https://github.com/oschwartz10612/poppler-windows/releases/
#       - Add both to PATH.

# 3) Install Python deps
pip install -r requirements.txt

# 4) Run pipeline
python -m src.process --pdf data/input/your.pdf --out_dir data/processed
# add more PDFs by repeating --pdf your2.pdf your3.pdf ...

# Disable visuals if you just want JSONL:
python -m src.process --pdf data/input/your.pdf --out_dir data/processed --no_visuals
```

Outputs (under `data/processed`):
- `images/` — PNGs per PDF page  
- `intermediate/*_pages.jsonl` — debug: OCR words + bboxes + labels (per page)  
- `visualizations/*_highlight.png` — highlighted boxes for labels  
- `dataset_chunks.jsonl` — final **chunked** dataset for LayoutLMv3  

---

## 2) What the pipeline does (step-by-step)

1. **PDF → Images** (pdf2image)  
   Converts each page to a PNG @ configurable DPI (default 200).

2. **OCR + Layout** (pytesseract)  
   Extracts **words** and **pixel boxes**, normalizes to **0–1000** (LayoutLM format), and filters by confidence.

3. **Rule-based labels**  
   - Section headers, subsection headers (see below).  
   - Everything else → `O`.  

4. **Chunking + Alignment** (LayoutLMv3Processor)  
   - Uses Hugging Face **overflow** + **stride** so every chunk is ≤ **512 tokens** with **128 overlap**.  
   - Subword→label alignment is handled via `word_labels` and `word_ids()`.  
   - Adds `[CLS]/[SEP]` with `labels = -100` automatically.

5. **Visualization**  
   - Overlays translucent rectangles on the original page image.  
   - **Sections = green, Subsections = orange.**  
   - Great for QA/Debug.

---

## 3) Labeling Strategy  

The labeling is **rule-based** and captures common academic PDF structures:

- **Section headers**  
  - Direct match with common names (*Abstract, Introduction, Conclusion, References…*).  
  - **Roman numerals** (`I.`, `II.`, `III.`).   
  → Tagged as `B-SEC` / `I-SEC`.

- **Subsection headers**  
  - **Roman numeral + number** (`II.6.`, `III.1.`).  
  → Tagged as `B-SUBSEC` / `I-SUBSEC`.

- **Cutoff rule**  
  Headers are capped at **15 tokens max**. Labeling stops when sentence-style lowercase words begin.

- **Everything else** is `O`.  

This keeps labeling clean and avoids over-tagging normal text.

---

## 4) Data format (JSONL)

**Intermediate (per page):**

```json
{
  "image_path": "data/processed/images/paper_page_001.png",
  "image_size": [1654, 2339],
  "words": ["Abstract", "This", "paper", "..."],
  "bboxes": [[60,72,260,110], "... normalized 0-1000 ..."],
  "confs": [95, 92, 88, "..."],
  "labels": [1, 0, 0, "..."],  // e.g., 1 = B-SEC
  "page_index": 0
}
```

**Final chunks (ready for HF Dataset):**

```json
{
  "doc_id": "paper",
  "page_index": 0,
  "chunk_index": 0,
  "image_path": "data/processed/images/paper_page_001.png",
  "input_ids": [101, 1332, "... up to 512 ..."],
  "attention_mask": [1, 1, 1, "..."],
  "bbox": [[0,0,0,0], [60,72,260,110], "... length == max_length ..."],
  "labels": [-100, 1, 0, "... aligned to tokens ..."]
}
```

---

## 5) Training tip (sketch)

```python
from datasets import load_dataset
from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer

id2label = {0:'O',1:'B-SEC',2:'I-SEC',3:'B-SUBSEC',4:'I-SUBSEC'}
label2id = {v:k for k,v in id2label.items()}

ds = load_dataset('json', data_files='data/processed/dataset_chunks.jsonl', split='train')

model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir='runs/layoutlmv3_tokcls',
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=50
)

def collate(batch):
    keys = ['input_ids','attention_mask','bbox','labels']
    out = {k: [b[k] for b in batch] for k in keys}
    out = {k: torch.tensor(v) for k,v in out.items()}
    return out

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
```
---

## 6) Results

![App Screenshot](https://drive.google.com/file/d/1f6oAo_Qdtiqss3qiPRO-t106fO3MZzmW/view?usp=drive_link)

---

## 7) Current Challenges

- **OCR Issue:** Sub-section headers not able to detect due to Roman Initials being absent from the OCR output.

  **Solution that might help** : Fine-tune OCR (Increasing DPI/Lowering Confidence Thresholds).

- Sub-section headers not being ended by period (.)

  **Solution that might help** : Use OCR output to create a txt file and then use regex to find end of section headers (a newline character)
  


---

## 8 Notes & Troubleshooting

- **Tesseract not found**: install & ensure its binary is in your PATH. On Windows, set  
  `pytesseract.pytesseract.tesseract_cmd` to the installed path.  
- **pdf2image errors**: you need Poppler (`pdftoppm`). Install as above.  

---
## Authors

- [@Sarvesh](https://github.com/Sarvesh326)
