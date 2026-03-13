import torch
import torch.nn.functional as F

from config.model_config import TinyAyaVisionConfig
from evaluation.utils import load_imagenette_images, load_model


def main(num_per_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model_config = TinyAyaVisionConfig.for_global()
    model, processor = load_model(model_config, device)
    images = load_imagenette_images(num_per_class=num_per_class)
    labels = [label for label, _ in images]

    pixel_values = processor.image_processor(
        images=[img for _, img in images], return_tensors="pt"
    )["pixel_values"].to(device).to(torch.bfloat16)

    with torch.no_grad():
        mlp_out = model.get_image_features(pixel_values)
        emb_matrix = model.language_model.get_input_embeddings().weight.detach()

    token_norms = mlp_out.float().norm(dim=-1)
    emb_norms = emb_matrix.float().norm(dim=-1)

    print("=== MLP output L2 norms (per image, across 196 tokens) ===")
    print(f"  Token embedding norms  — mean: {emb_norms.mean():.3f}  std: {emb_norms.std():.3f}")
    print()
    for i, label in enumerate(labels):
        norms_i = token_norms[i]
        print(
            f"  {label:<20} mean={norms_i.mean():.3f}  std={norms_i.std():.3f}"
            f"  min={norms_i.min():.3f}  max={norms_i.max():.3f}"
        )

    pooled = mlp_out.float().mean(dim=1)
    pooled_norm = F.normalize(pooled, dim=-1)
    sim_matrix = pooled_norm @ pooled_norm.T

    print("\n=== Inter-image cosine similarity (mean-pooled MLP outputs) ===")
    off_diag = sim_matrix[~torch.eye(len(labels), dtype=torch.bool)].mean().item()
    print(f"\n  Mean off-diagonal similarity: {off_diag:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-per-class", type=int, default=1)
    args = parser.parse_args()

    main(num_per_class=args.num_per_class)
