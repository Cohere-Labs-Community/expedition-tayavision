import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from evaluation.utils import load_imagenette_images, load_model


def _is_readable(text: str) -> bool:
    """True for clean ASCII word-like tokens (no BPE noise, no foreign script)."""
    if not text or not text.isascii():
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count >= 2 and alpha_count / len(text) >= 0.6


def get_top_k_tokens(image_embeddings, embedding_matrix, tokenizer, k) -> list[list[tuple[str, float]]]:
    pooled = image_embeddings.mean(dim=1).float()
    emb = embedding_matrix.float()
    sims = F.normalize(pooled, dim=-1) @ F.normalize(emb, dim=-1).T

    # Fetch many more candidates so filtering still yields k results
    top = sims.topk(k * 20, dim=-1)
    candidates, scores = top.indices, top.values

    results = []
    for row_ids, row_scores in zip(candidates, scores):
        tokens = []
        seen = set()
        for tid, score in zip(row_ids.tolist(), row_scores.tolist()):
            text = tokenizer.decode([tid]).strip()
            if _is_readable(text) and text not in seen:
                tokens.append((text, score))
                seen.add(text)
            if len(tokens) == k:
                break
        results.append(tokens)
    return results


def main(top_k=10, num_per_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor = load_model(device)

    images = load_imagenette_images(num_per_class=num_per_class)
    print(f"Loaded {len(images)} image(s): {[label for label, _ in images]}")

    pixel_values = processor.image_processor(
        images=[img for _, img in images], return_tensors="pt"
    )["pixel_values"].to(device).to(torch.bfloat16)

    with torch.no_grad():
        image_embeddings = model.get_image_features(pixel_values)
        embedding_matrix = model.language_model.get_input_embeddings().weight.detach()

    top_tokens = get_top_k_tokens(image_embeddings, embedding_matrix, processor.tokenizer, top_k)

    results = []
    for (label, img), tokens in zip(images, top_tokens):
        print(f"  {label}: {[t for t, _ in tokens]}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results.append({
            "label": label,
            "image_bytes": buf.getvalue(),
            "tokens": tokens,
        })

    return results


def build_composite(label: str, img: Image.Image, tokens: list[tuple[str, float]]) -> Image.Image:
    img = img.copy()
    img.thumbnail((384, 384))
    img_w, img_h = img.size

    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=14)

    label_h = 32
    token_line_h = 22
    panel_w = 260
    total_h = label_h + max(img_h, len(tokens) * token_line_h)
    total_w = img_w + panel_w

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([(0, 0), (total_w, label_h)], fill="#2c3e50")
    draw.text((8, 6), label, fill="white", font=font)

    canvas.paste(img, (0, label_h))

    x0 = img_w + 12
    for i, (token, sim) in enumerate(tokens):
        y = label_h + 6 + i * token_line_h
        draw.text((x0, y), token, fill="#2c3e50", font=small_font)
        draw.text((x0 + 140, y), f"{sim:.3f}", fill="#7f8c8d", font=small_font)

    return canvas


def save_assets(results: list[dict], output_dir: Path = Path("assets/eval_aligned_tokens")):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in results:
        label = r["label"]
        idx = counts.get(label, 0)
        counts[label] = idx + 1
        suffix = f"_{idx}" if idx > 0 else ""
        img = Image.open(io.BytesIO(r["image_bytes"]))
        composite = build_composite(label, img, r["tokens"])
        out_path = output_dir / f"{label.replace(' ', '_')}{suffix}.png"
        composite.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-per-class", type=int, default=1)
    args = parser.parse_args()

    results = main(top_k=args.top_k, num_per_class=args.num_per_class)
    save_assets(results)
