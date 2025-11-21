"""
Simple structured pruning utilities.

⚠️ Layer names are for BERT-like models and may need adaptation for other architectures.
"""
import torch


def prune_attention_heads_bert_layer(model, layer_idx: int, heads_to_prune):
    """
    Zeroes out selected attention heads in a BERT-like encoder layer.
    """
    encoder_layer = model.base_model.encoder.layer[layer_idx]
    attn = encoder_layer.attention.self
    num_heads = attn.num_attention_heads
    head_dim = attn.attention_head_size  # out_features = num_heads * head_dim

    mask = torch.ones(num_heads, head_dim, device=attn.query.weight.device)
    for h in heads_to_prune:
        if 0 <= h < num_heads:
            mask[h, :] = 0.0
    mask = mask.view(-1)  # (num_heads * head_dim,)

    with torch.no_grad():
        for proj in [attn.query, attn.key, attn.value]:
            w = proj.weight.data
            proj.weight.data = w * mask.unsqueeze(1)


def magnitude_prune_linear(layer: torch.nn.Linear, keep_fraction: float = 0.8):
    """
    Prune smallest-magnitude weights in a Linear layer.
    """
    with torch.no_grad():
        w = layer.weight.data
        flat = w.abs().view(-1)
        k = int(flat.numel() * keep_fraction)
        if k <= 0:
            return
        threshold = flat.kthvalue(k).values
        mask = (w.abs() >= threshold).float()
        layer.weight.data = w * mask


def example_prune_student(model):
    """
    Example: prune last encoder layer's some heads and compress FFN.
    """
    # prune heads 2 and 5 of last layer
    last_idx = len(model.base_model.encoder.layer) - 1
    prune_attention_heads_bert_layer(model, last_idx, heads_to_prune=[2, 5])

    # magnitude prune intermediate dense layer in first encoder layer
    first_layer = model.base_model.encoder.layer[0]
    magnitude_prune_linear(first_layer.intermediate.dense, keep_fraction=0.7)

    return model
