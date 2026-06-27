"""Generate prompt bank embeddings for water segmentation.

Usage:
    python tools/generate_water_prompt_bank.py --output tools/prompt_bank_water.pt

This script encodes category-aware prompts using CLIP Text Encoder
and saves the embeddings as a .pt file for training with CLIPEncoderDecoder.
"""

import argparse
import os
import sys

import torch

# Water segmentation prompt bank
PROMPT_BANK = {
    'water': [
        'river',
        'lake',
        'sea',
        'ocean',
        'wave',
        'water surface',
        'water reflection',
        'flood',
        'stream',
        'reservoir',
    ],
    'ship': [
        'boat',
        'vessel',
        'cargo ship',
        'fishing boat',
        'yacht',
        'sailboat',
        'canoe',
        'barge',
        'ship',
        'small boat',
    ],
    'land': [
        'shore',
        'coast',
        'vegetation',
        'road',
        'bridge',
        'building',
        'sky',
        'tree',
        'sand',
        'grass',
    ],
}


def load_clip_model(model_name='ViT-B/32', device='cpu'):
    """Load CLIP model."""
    try:
        import clip
        model, _ = clip.load(model_name, device=device)
        return model
    except ImportError:
        print('OpenAI CLIP not found. Trying transformers...')
        try:
            from transformers import CLIPModel, CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(
                f'openai/clip-{model_name.replace("/", "-").lower()}')
            model = CLIPModel.from_pretrained(
                f'openai/clip-{model_name.replace("/", "-").lower()}')
            return model, tokenizer
        except ImportError:
            print('Neither clip nor transformers available.')
            print('Using random embeddings for testing.')
            return None


def encode_prompts_clip(prompts, model, device='cpu'):
    """Encode prompts using OpenAI CLIP."""
    import clip
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


def encode_prompts_transformers(prompts, model, tokenizer, device='cpu'):
    """Encode prompts using HuggingFace transformers."""
    inputs = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


def encode_prompts_random(prompts, embed_dim=512):
    """Generate random embeddings for testing (no CLIP available)."""
    print(f'WARNING: Using random embeddings ({len(prompts)} prompts, dim={embed_dim})')
    embeddings = torch.randn(len(prompts), embed_dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description='Generate water segmentation prompt bank')
    parser.add_argument('--output', type=str,
                        default='tools/prompt_bank_water.pt',
                        help='Output .pt file path')
    parser.add_argument('--model', type=str, default='ViT-B/32',
                        help='CLIP model name')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for encoding')
    parser.add_argument('--embed-dim', type=int, default=512,
                        help='Embedding dimension (for random fallback)')
    args = parser.parse_args()

    # Collect all prompts
    categories = list(PROMPT_BANK.keys())
    all_prompts = []
    category_indices = []
    for cat_idx, cat in enumerate(categories):
        prompts = PROMPT_BANK[cat]
        all_prompts.extend(prompts)
        category_indices.extend([cat_idx] * len(prompts))

    print(f'Categories: {categories}')
    print(f'Total prompts: {len(all_prompts)}')

    # Load CLIP model
    clip_model = load_clip_model(args.model, args.device)

    # Encode prompts
    if clip_model is None:
        embeddings = encode_prompts_random(all_prompts, args.embed_dim)
    elif isinstance(clip_model, tuple):
        model, tokenizer = clip_model
        embeddings = encode_prompts_transformers(
            all_prompts, model, tokenizer, args.device)
    else:
        embeddings = encode_prompts_clip(
            all_prompts, clip_model, args.device)

    # Organize by category
    num_categories = len(categories)
    prompts_per_category = len(PROMPT_BANK[categories[0]])
    embed_dim = embeddings.shape[-1]

    # Reshape: [num_categories, prompts_per_category, embed_dim]
    organized = torch.zeros(num_categories, prompts_per_category, embed_dim)
    for i, emb in enumerate(embeddings):
        cat_idx = category_indices[i]
        prompt_idx = i - sum(len(PROMPT_BANK[c]) for c in categories[:cat_idx])
        organized[cat_idx, prompt_idx] = emb

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'embeddings': organized,
        'categories': categories,
        'prompts': PROMPT_BANK,
        'embed_dim': embed_dim,
        'model_name': args.model,
    }, args.output)

    print(f'Saved prompt bank to {args.output}')
    print(f'Shape: {organized.shape} (categories x prompts x dim)')


if __name__ == '__main__':
    main()
