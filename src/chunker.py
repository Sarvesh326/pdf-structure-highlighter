from typing import List, Dict, Any
from PIL import Image
from transformers import LayoutLMv3Processor

def build_processor(model_name: str = "microsoft/layoutlmv3-base") -> LayoutLMv3Processor:
    return LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)

def chunk_with_processor(pages: List[Dict[str, Any]], processor: LayoutLMv3Processor,
                         max_length: int = 512, stride: int = 128):
    """Creates overlapping token chunks using HF overflow/stride mechanics.

    Each element in `pages` must have: words, bboxes, image_path.
    Returns a list of encodings dicts (ready for model) plus metadata for traceability.
    """
    all_chunks = []
    for i, page in enumerate(pages):
        print("PAGE : ", i+1)
        image = Image.open(page["image_path"]).convert("RGB")
        words = page["words"]
        # print("WORDS : " , words)
        boxes = page["bboxes"]
        word_labels = page.get("labels", [0]*len(words))

        enc = processor(
            images=image,
            text=[words],         
            boxes=[boxes],
            word_labels=[word_labels],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_offsets_mapping=False
        )

        n_overflows = len(enc["input_ids"])
        for i in range(n_overflows):
            chunk = {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "bbox": enc["bbox"][i],
                "labels": enc["labels"][i],
                "image_path": page["image_path"],
                "page_index": page.get("page_index", 0),
                "overflow_to_sample_mapping": int(enc["overflow_to_sample_mapping"][i]) if "overflow_to_sample_mapping" in enc else 0,
                "sequence_ids": enc.sequence_ids(i) if hasattr(enc, "sequence_ids") else None,
            }
            all_chunks.append(chunk)
    return all_chunks
