"""Tokenise le corpus kabyle une fois pour toutes, en un tableau d'entiers 16 bits.

    python src/prepare.py --vocab 2048

Écrit artefacts/tokens_<vocab>.bin (uint16) et artefacts/split_<vocab>.json.
Les 2 % de la fin servent de validation : ils ne sont jamais vus à l'entraînement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = r"C:\Users\dell\OneDrive\Bureau\acad\kabyllm\kaggle_v2\dataset\kabyle_corpus.txt.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--lignes", type=int, default=0, help="0 = tout le corpus")
    ap.add_argument("--out", default=os.path.join(ROOT, "artefacts"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.out, f"tokenizer_{args.vocab}.json"))
    ouvrir = gzip.open if args.corpus.endswith(".gz") else open
    morceaux, n_chars, n_lignes = [], 0, 0
    lot = []
    with ouvrir(args.corpus, "rt", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            n_lignes += 1
            n_chars += len(ligne) + 1
            lot.append(ligne)
            if len(lot) >= 20000:
                for enc in tok.encode_batch(lot):
                    morceaux.append(np.asarray(enc.ids, dtype=np.uint16))
                lot = []
            if args.lignes and n_lignes >= args.lignes:
                break
    for enc in tok.encode_batch(lot):
        morceaux.append(np.asarray(enc.ids, dtype=np.uint16))

    tokens = np.concatenate(morceaux)
    chemin = os.path.join(args.out, f"tokens_{args.vocab}.bin")
    tokens.tofile(chemin)
    n_val = int(len(tokens) * 0.02)
    info = {"vocab": args.vocab, "n_tokens": int(len(tokens)), "n_chars": n_chars,
            "n_lignes": n_lignes, "n_validation": n_val,
            "jetons_par_caractere": len(tokens) / max(n_chars, 1)}
    with open(os.path.join(args.out, f"split_{args.vocab}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"{chemin} : {len(tokens)/1e6:.1f} M jetons pour {n_chars/1e6:.1f} M caractères "
          f"({info['jetons_par_caractere']:.3f} jeton/caractère) · validation {n_val/1e6:.2f} M")


if __name__ == "__main__":
    main()
