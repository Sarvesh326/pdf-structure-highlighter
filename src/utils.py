import json
from pathlib import Path

def normalize_bbox(left, top, width, height, img_w, img_h):
    # normalize to 0..1000 int as required by LayoutLM series
    x0 = max(0, min(1000, int(1000 * left / img_w)))
    y0 = max(0, min(1000, int(1000 * top / img_h)))
    x1 = max(0, min(1000, int(1000 * (left + width) / img_w)))
    y1 = max(0, min(1000, int(1000 * (top + height) / img_h)))

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [x0, y0, x1, y1]

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

LABEL2ID = {
    "O": 0,
    "B-SEC": 1,      
    "I-SEC": 2,
    "B-SUBSEC": 3,   
    "I-SUBSEC": 4    
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
