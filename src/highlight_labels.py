from typing import List
from PIL import Image, ImageDraw

def highlight_tokens(image_path: str, boxes: List[List[int]], labels: List[int],
                     out_path: str, label_id_exclude: int = 0, opacity: int = 64):
    """
    Highlight tokens with different colors depending on label type.
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = img.size
    to_px = lambda v, L: int(v * L / 1000)  # reverse-normalize

    # Define colors by label type
    COLORS = {
        1: (0, 0, 255),     # Purple
        2: (0, 0, 255),     # Purple
        3: (0, 200, 0),     # GReen
        4: (0, 200, 0),     # Green
    }

    for bbox, lab in zip(boxes, labels):
        if lab == label_id_exclude or lab not in COLORS:
            continue

        color = COLORS[lab]
        x0, y0 = to_px(bbox[0], W), to_px(bbox[1], H)
        x1, y1 = to_px(bbox[2], W), to_px(bbox[3], H)

        # Draw rectangle with translucent fill
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(color[0], color[1], color[2], opacity),
            outline=(color[0], color[1], color[2], 255),
            width=2,
        )

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_path)
    return out_path
