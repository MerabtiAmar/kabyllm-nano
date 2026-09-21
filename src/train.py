"""Entraînement du modèle nano, avec une mesure comparable : les bits par caractère.

La perte d'un modèle de langue dépend du tokenizer, donc deux modèles ne se comparent qu'en
ramenant tout au caractère : bits/caractère = perte × jetons_par_caractère / ln(2). C'est la
mesure utilisée dans mon dépôt KabyLLM ; la comparaison n'a de sens que sur le même jeu de
validation, ce qui n'est pas le cas entre le nano (corpus) et KabyLLM (discussion).

    python src/train.py --pas 20000                 # entraînement complet (GPU)
    python src/train.py --pas 300 --lot 16          # essai rapide sur processeur
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import Config, NanoLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lots(tokens, debut, fin, lot, contexte, device, generator=None):
    haut = fin - contexte - 1
    i = torch.randint(debut, haut, (lot,), generator=generator)
    x = torch.stack([torch.from_numpy(tokens[j:j + contexte].astype(np.int64)) for j in i])
    y = torch.stack([torch.from_numpy(tokens[j + 1:j + 1 + contexte].astype(np.int64)) for j in i])
    return x.to(device), y.to(device)


@torch.no_grad()
def perte_validation(model, tokens, debut, fin, lot, contexte, device, n=20):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(n):
        x, y = lots(tokens, debut, fin, lot, contexte, device, gen)
        _, perte = model(x, y)
        total += perte.item()
    model.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--couches", type=int, default=4)
    ap.add_argument("--tetes", type=int, default=4)
    ap.add_argument("--contexte", type=int, default=192)
    ap.add_argument("--pas", type=int, default=20000)
    ap.add_argument("--lot", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--nom", default="nano")
    ap.add_argument("--artefacts", default=os.path.join(ROOT, "artefacts"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = json.load(open(os.path.join(args.artefacts, f"split_{args.vocab}.json"), encoding="utf-8"))
    tokens = np.memmap(os.path.join(args.artefacts, f"tokens_{args.vocab}.bin"), dtype=np.uint16,
                       mode="r")
    n_val = info["n_validation"]
    fin_train = len(tokens) - n_val
    jpc = info["jetons_par_caractere"]

    cfg = Config(vocab=args.vocab, d_model=args.d_model, n_layers=args.couches,
                 n_heads=args.tetes, contexte=args.contexte, d_mlp=4 * args.d_model)
    model = NanoLM(cfg).to(device)
    print(f"{args.nom} : {model.n_params()/1e6:.2f} M paramètres "
          f"({model.n_params(True)/1e6:.2f} M hors embeddings) · {device} · "
          f"{len(tokens)/1e6:.1f} M jetons")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.pas,
                                                pct_start=0.05)
    histoire, t0 = [], time.time()
    for pas in range(1, args.pas + 1):
        x, y = lots(tokens, 0, fin_train, args.lot, cfg.contexte, device)
        _, perte = model(x, y)
        opt.zero_grad(set_to_none=True)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if pas % max(1, args.pas // 40) == 0 or pas == 1:
            v = perte_validation(model, tokens, fin_train, len(tokens), args.lot, cfg.contexte,
                                 device, n=10)
            bpc = v * jpc / math.log(2)
            histoire.append({"pas": pas, "train": perte.item(), "val": v, "bits_par_caractere": bpc})
            print(f"  pas {pas:6d} · train {perte.item():.3f} · val {v:.3f} · {bpc:.3f} bits/car")

    v = perte_validation(model, tokens, fin_train, len(tokens), args.lot, cfg.contexte, device, n=40)
    bpc = v * jpc / math.log(2)
    minutes = (time.time() - t0) / 60
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(),
                "val": v, "bits_par_caractere": bpc, "jetons_par_caractere": jpc,
                "args": vars(args), "minutes": minutes},
               os.path.join(ROOT, "runs", f"{args.nom}.pt"))
    with open(os.path.join(ROOT, "runs", f"{args.nom}_histoire.json"), "w", encoding="utf-8") as f:
        json.dump({"histoire": histoire, "val": v, "bits_par_caractere": bpc,
                   "params": model.n_params(), "minutes": minutes, "config": cfg.to_dict()},
                  f, ensure_ascii=False, indent=2)
    print(f"final : val {v:.3f} · {bpc:.3f} bits par caractère · {minutes:.1f} min")


if __name__ == "__main__":
    main()
