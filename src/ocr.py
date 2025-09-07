from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

from .utils import normalize_bbox
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\tesseract-ocr\tesseract.exe"

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"C:\Users\Sri Lakshmi Prasanna\OneDrive\Documents\Sarvesh\Release-25.07.0-0\poppler-25.07.0\Library\bin")
    image_paths = []
    for i, img in enumerate(pages):
        p = out_dir / f"{Path(pdf_path).stem}_page_{i+1:03d}.png"
        img.save(p)
        image_paths.append(str(p))
    return image_paths

def ocr_image(image_path: str, conf_threshold: int = 40) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words, bboxes, confidences = [], [], []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i] != '-1' else -1
        if text and conf >= conf_threshold:
            left, top, width, height = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = normalize_bbox(left, top, width, height, W, H)
            words.append(text)
            bboxes.append(bbox)
            confidences.append(conf)
    return {
        "image_path": image_path,
        "image_size": [W, H],
        "words": words,
        "bboxes": bboxes,
        "confs": confidences,
    }

def process_pdf(pdf_path: str, out_img_dir: str, conf_threshold: int = 40):
    image_paths = pdf_to_images(pdf_path, out_img_dir)
    pages = []
    for idx, ip in enumerate(tqdm(image_paths, desc=f"OCR {Path(pdf_path).name}")):
        page = ocr_image(ip, conf_threshold=conf_threshold)
        page["page_index"] = idx
        pages.append(page)
    return pages
