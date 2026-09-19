"""Génère des continuations à partir d'amorces, pour juger le modèle à l'œil.

    python src/echantillons.py --run nano --n 6
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from model import Config, NanoLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMORCES = ["Azul", "Tamurt n", "Aqcic-nni", "Deg uzekka", "Tameṭṭut-nni tenna", "Asmi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="nano")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--jetons", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--artefacts", default=os.path.join(ROOT, "artefacts"))
    args = ap.parse_args()

    ck = torch.load(os.path.join(ROOT, "runs", f"{args.run}.pt"), map_location="cpu",
                    weights_only=False)
    cfg = Config(**ck["config"])
    model = NanoLM(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(args.artefacts, f"tokenizer_{cfg.vocab}.json"))

    sorties = []
    torch.manual_seed(0)
    for amorce in AMORCES[:args.n]:
        ids = torch.tensor([tok.encode(amorce).ids])
        out = model.generer(ids, n=args.jetons, temperature=args.temperature, top_k=40)
        texte = tok.decode(out[0].tolist())
        sorties.append({"amorce": amorce, "texte": texte})
    chemin = os.path.join(ROOT, "runs", f"{args.run}_echantillons.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"bits_par_caractere": ck.get("bits_par_caractere"), "sorties": sorties},
                  f, ensure_ascii=False, indent=2)
    print("écrit :", chemin)
    for s in sorties:
        print("  ·", s["texte"][:110].replace("\n", " "))


if __name__ == "__main__":
    main()
