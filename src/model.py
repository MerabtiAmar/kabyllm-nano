"""Modèle de langue minuscule, pensé pour tenir dans un navigateur.

Décodeur Transformer classique (pré-normalisation, positions apprises, embeddings partagés avec
la tête de sortie). Tout ce qui est gros dans un modèle de langue vient du vocabulaire et de la
largeur : on garde les deux petits, et on compense par un tokenizer adapté au kabyle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 2048
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 4
    contexte: int = 192
    d_mlp: int = 640

    def to_dict(self):
        return asdict(self)


class Bloc(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, c.d_mlp), nn.GELU(),
                                 nn.Linear(c.d_mlp, c.d_model))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.c.n_heads, D // self.c.n_heads
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class NanoLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.contexte, c.d_model)
        self.blocs = nn.ModuleList([Bloc(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok.weight          # embeddings partagés : autant de gagné
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, cibles=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocs:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if cibles is None:
            return logits, None
        perte = F.cross_entropy(logits.reshape(-1, self.c.vocab), cibles.reshape(-1))
        return logits, perte

    @torch.no_grad()
    def generer(self, idx, n=60, temperature=0.9, top_k=40):
        for _ in range(n):
            fenetre = idx[:, -self.c.contexte:]
            logits, _ = self(fenetre)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def n_params(self, sans_embeddings=False):
        n = sum(p.numel() for p in self.parameters())
        if sans_embeddings:
            n -= self.tok.weight.numel() + self.pos.weight.numel()
        return n
