"""Quantifie le modèle en 8 bits, mesure ce que ça coûte, et l'exporte pour le navigateur.

Chaque tenseur est ramené à des entiers signés de 8 bits avec une échelle qui lui est propre
(quantification par tenseur, symétrique). On mesure la perte de validation avant et après :
c'est le prix à payer pour diviser le poids du fichier par deux face au float16.

    python src/quantize_export.py --run nano --out ../merabtiamar.github.io/assets/data/kabyllm-nano.js
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os

import numpy as np
import torch

from model import Config, NanoLM
from train import perte_validation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LF = chr(10)


def noms_tenseurs(n_layers: int):
    noms = ["tok.weight", "pos.weight"]
    for l in range(n_layers):
        p = f"blocs.{l}."
        noms += [p + "ln1.weight", p + "ln1.bias", p + "qkv.weight", p + "proj.weight",
                 p + "ln2.weight", p + "ln2.bias",
                 p + "mlp.0.weight", p + "mlp.0.bias", p + "mlp.2.weight", p + "mlp.2.bias"]
    return noms + ["ln_f.weight", "ln_f.bias"]


def quantifier(t: np.ndarray):
    """Entiers 8 bits symétriques, une échelle par tenseur."""
    echelle = float(np.max(np.abs(t))) / 127.0 or 1e-8
    q = np.clip(np.round(t / echelle), -127, 127).astype(np.int8)
    return q, echelle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="nano")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--artefacts", default=os.path.join(ROOT, "artefacts"))
    args = ap.parse_args()

    ck = torch.load(os.path.join(ROOT, "runs", f"{args.run}.pt"), map_location="cpu",
                    weights_only=False)
    cfg = Config(**ck["config"])
    model = NanoLM(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    info = json.load(open(os.path.join(args.artefacts, f"split_{cfg.vocab}.json"), encoding="utf-8"))
    tokens = np.memmap(os.path.join(args.artefacts, f"tokens_{cfg.vocab}.bin"), dtype=np.uint16,
                       mode="r")
    fin_train = len(tokens) - info["n_validation"]
    jpc = info["jetons_par_caractere"]
    avant = perte_validation(model, tokens, fin_train, len(tokens), 24, cfg.contexte, "cpu", n=40)

    sd = model.state_dict()
    noms = noms_tenseurs(cfg.n_layers)
    manquants = [n for n in noms if n not in sd]
    assert not manquants, manquants
    paquets, echelles, total = [], [], 0
    sd_q = {}
    for n in noms:
        t = sd[n].numpy().astype(np.float32)
        q, e = quantifier(t)
        paquets.append(q.ravel())
        echelles.append(e)
        total += q.size
        sd_q[n] = torch.tensor(q.astype(np.float32) * e).view_as(sd[n])
    sd_q["head.weight"] = sd_q["tok.weight"]         # embeddings partagés
    model.load_state_dict(sd_q)
    apres = perte_validation(model, tokens, fin_train, len(tokens), 24, cfg.contexte, "cpu", n=40)

    plat = np.concatenate(paquets).astype(np.int8)
    tok_json = json.load(open(os.path.join(args.artefacts, f"tokenizer_{cfg.vocab}.json"),
                              encoding="utf-8"))
    fert = json.load(open(os.path.join(args.artefacts, f"fertilite_{cfg.vocab}.json"),
                          encoding="utf-8"))
    meta = {
        "config": cfg.to_dict(), "n": int(plat.size), "echelles": echelles,
        "params": model.n_params(), "bits_par_caractere": ck.get("bits_par_caractere"),
        "bits_par_caractere_int8": apres * jpc / math.log(2),
        "val_float": avant, "val_int8": apres, "jetons_par_caractere": jpc,
        "fertilite": fert,
        "vocabulaire": tok_json["model"]["vocab"], "fusions": tok_json["model"]["merges"],
        "minutes": ck.get("minutes"),
    }
    body = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    b64 = base64.b64encode(plat.tobytes()).decode("ascii")
    txt = ("/* KabyLLM-nano : poids quantifiés en 8 bits (base64), vocabulaire et fusions BPE."
           + LF + "   Généré par quantize_export.py, dépôt kabyllm-nano. */"
           + LF + "window.KABY = " + body[:-1] + ',"w":"' + b64 + '"};' + LF)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline=LF) as f:
        f.write(txt)
    with open(os.path.join(ROOT, "runs", f"{args.run}_quant.json"), "w", encoding="utf-8") as f:
        json.dump({"val_float": avant, "val_int8": apres, "params": model.n_params(),
                   "poids_fichier_mo": len(txt) / 1024 / 1024, "n_poids": int(plat.size),
                   "bits_par_caractere_int8": meta["bits_par_caractere_int8"],
                   "bits_par_caractere": ck.get("bits_par_caractere"),
                   "jetons_par_caractere": jpc, "config": cfg.to_dict()}, f,
                  ensure_ascii=False, indent=2)
    print(f"{args.out} : {plat.size/1e6:.2f} M poids en 8 bits, fichier {len(txt)/1024/1024:.2f} Mo")
    print(f"perte de validation : {avant:.4f} en flottant, {apres:.4f} en 8 bits "
          f"(+{100*(apres-avant)/avant:.2f} %) · {meta['bits_par_caractere_int8']:.3f} bits/car")


if __name__ == "__main__":
    main()
