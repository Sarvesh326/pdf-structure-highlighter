import argparse
from pathlib import Path

from .ocr import process_pdf
from .labeling import label_tokens
from .chunker import build_processor, chunk_with_processor
from .highlight_labels import highlight_tokens
from .utils import save_jsonl

def run_pipeline(pdf_paths, out_dir, dpi=200, conf_threshold=40, model_name="microsoft/layoutlmv3-base",
                 max_length=512, stride=128, make_visuals=True):
    out_dir = Path(out_dir)
    imgs_dir = out_dir / "images"
    inter_dir = out_dir / "intermediate"
    vis_dir = out_dir / "visualizations"
    for d in [imgs_dir, inter_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    processor = build_processor(model_name=model_name)
    all_rows = []

    for pdf in pdf_paths:
        doc_id = Path(pdf).stem
        pages = process_pdf(pdf, imgs_dir, conf_threshold=conf_threshold)
        # Label at word level
        for p in pages:
            p["labels"] = label_tokens(p["words"])

        # Save page-level intermediate (debugging)
        inter_path = inter_dir / f"{doc_id}_pages.jsonl"
        save_jsonl(inter_path, pages)

        # Chunk + tokenize with HF
        chunks = chunk_with_processor(pages, processor, max_length=max_length, stride=stride)

        # Save chunks in JSONL
        for i, ch in enumerate(chunks):
            row = {
                "doc_id": doc_id,
                "page_index": ch["page_index"],
                "chunk_index": i,
                "image_path": ch["image_path"],
                "input_ids": ch["input_ids"],
                "attention_mask": ch["attention_mask"],
                "bbox": ch["bbox"],
                "labels": ch["labels"],
            }
            all_rows.append(row)

        # Visuals: highlight tokens
        if make_visuals:
            for p in pages:
                out_img = vis_dir / f"{doc_id}_page_{p['page_index']+1:03d}_highlight.png"
                try:
                    highlight_tokens(p["image_path"], p["bboxes"], p["labels"], str(out_img))
                except Exception as e:
                    print("Highlight failed:", e)

    # Write dataset JSONL
    save_jsonl(out_dir / "dataset_chunks.jsonl", all_rows)
    return out_dir

def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline for LayoutLMv3 token classification dataset prep.")
    ap.add_argument("--pdf", nargs="+", required=True, help="One or more PDF file paths.")
    ap.add_argument("--out_dir", required=True, help="Output directory for artifacts.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--conf_threshold", type=int, default=40)
    ap.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--no_visuals", action="store_true", help="Disable highlight image generation.")
    args = ap.parse_args()

    run_pipeline(
        pdf_paths=args.pdf,
        out_dir=args.out_dir,
        dpi=args.dpi,
        conf_threshold=args.conf_threshold,
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride,
        make_visuals=not args.no_visuals
    )

if __name__ == "__main__":
    main()
