"""AIM video encoder used by TF-CoVR (NeurIPS 2025), copied for feature extraction only.

Source: github.com/UCF-CRCV/TF-CoVR, AIM_Embeddings/src/model/aim/aim_model.py (MIT licence).
It is a CLIP ViT-B/16 image encoder with temporal, spatial and MLP adapters, fine-tuned by the
TF-CoVR authors to classify 306 FineGym + FineDiving sub-actions from 12-frame clips.

Changes from the original, none of which affect the weights:
  * DropPath replaced by Identity (it does nothing at inference) and no `clip`/`timm` imports
  * `features()` returns the pooled 768-d vector (the input of the classifier, i.e. what the
    TF-CoVR `aim_emb.py` script saves), the per-frame CLS tokens, and the 306 class logits
"""
from collections import OrderedDict

import torch
from einops import rearrange
from torch import nn


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc2(self.act(self.D_fc1(x)))
        return x + xs if self.skip_connect else xs


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, scale=1.0, num_frames=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.num_frames = num_frames

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        n, bt, d = x.shape
        xt = rearrange(x, "n (b t) d -> t (b n) d", t=self.num_frames)
        xt = self.T_Adapter(self.attention(self.ln_1(xt)))           # temporal adaptation
        x = x + rearrange(xt, "t (b n) d -> n (b t) d", n=n)
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))          # spatial adaptation
        xn = self.ln_2(x)
        return x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)  # joint adaptation


class Transformer(nn.Module):
    def __init__(self, num_frames, width, layers, heads, scale=1.0):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, scale, num_frames)
                                         for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class ViT_CLIP(nn.Module):
    def __init__(self, input_resolution=224, num_frames=12, patch_size=16, width=768, layers=12,
                 heads=12, adapter_scale=0.5, num_classes=306):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(num_frames, width, layers, heads, scale=adapter_scale)
        self.ln_post = LayerNorm(width)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(768, num_classes)

    def features(self, x):
        """x: [B, T, 3, 224, 224] -> pooled [B, 768], per-frame CLS [B, T, 768], logits [B, 306]"""
        B, T = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv1(x).flatten(2).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                             dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frames) + self.temporal_embedding
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2)
        x = self.ln_post(x)[:, 0]                                       # CLS token per frame
        per_frame = rearrange(x, "(b t) d -> b t d", b=B, t=T)
        pooled = per_frame.mean(1)                    # == AdaptiveAvgPool3d over time in original
        return pooled, per_frame, self.fc_cls(pooled)

    def forward(self, x):
        return self.features(x)[2]


def _find_state_dict(obj, depth=0):
    """Return the dict holding the most tensors (handles Lightning / Fabric / raw checkpoints)."""
    if not isinstance(obj, dict) or depth > 3:
        return None
    n_tensors = sum(torch.is_tensor(v) for v in obj.values())
    best, best_n = (obj, n_tensors) if n_tensors else (None, 0)
    for v in obj.values():
        sd = _find_state_dict(v, depth + 1)
        if sd is not None:
            n = sum(torch.is_tensor(t) for t in sd.values())
            if n > best_n:
                best, best_n = sd, n
    return best


def load_aim(ckpt_path, device="cpu", min_match=0.95):
    """Build ViT_CLIP and load TF-CoVR stage-1 weights. Fails loudly if the weights do not match
    (the original script uses strict=False, which would silently leave random weights)."""
    net = ViT_CLIP()
    target = net.state_dict()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _find_state_dict(ck)
    if sd is None:
        raise RuntimeError(f"no tensors found in {ckpt_path}; top-level keys: {list(ck)[:10]}")
    prefixes = ["", "model.", "model.model.", "module.", "module.model.", "backbone.", "net."]

    def matched(p):
        return {k[len(p):]: v for k, v in sd.items()
                if k.startswith(p) and k[len(p):] in target and v.shape == target[k[len(p):]].shape}

    best = max(prefixes, key=lambda p: len(matched(p)))
    use = matched(best)
    frac = len(use) / len(target)
    print(f"checkpoint: {len(sd)} tensors | prefix '{best}' | matched {len(use)}/{len(target)} "
          f"model tensors ({100 * frac:.1f}%)")
    if frac < min_match:
        missing = [k for k in target if k not in use][:10]
        print("example checkpoint keys:", list(sd)[:10])
        print("example missing model keys:", missing)
        raise RuntimeError("checkpoint does not match the AIM model; do not use these features")
    net.load_state_dict(use, strict=False)
    return net.to(device).eval()
