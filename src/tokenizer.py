"""Tokenizer BPE entraîné sur le kabyle, petit exprès.

Le modèle doit tenir dans un navigateur : chaque entrée du vocabulaire coûte d_model paramètres
dans la table d'embedding. On cherche donc le plus petit vocabulaire qui garde une bonne
compression — mesurée en jetons par mot, comme dans le rapport de fertilité de KabyLLM.

    python src/tokenizer.py --vocab 2048
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = r"C:\Users\dell\OneDrive\Bureau\acad\kabyllm\kaggle_v2\dataset\kabyle_corpus.txt.gz"


def lignes(path: str, limite: int | None = None):
    ouvrir = gzip.open if path.endswith(".gz") else open
    with ouvrir(path, "rt", encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            if limite and i >= limite:
                break
            ligne = ligne.strip()
            if ligne:
                yield ligne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--lignes", type=int, default=400000)
    ap.add_argument("--out", default=os.path.join(ROOT, "artefacts"))
    args = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, min_frequency=2,
                                  special_tokens=["<unk>", "<bos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(lignes(args.corpus, args.lignes), trainer=trainer)

    os.makedirs(args.out, exist_ok=True)
    chemin = os.path.join(args.out, f"tokenizer_{args.vocab}.json")
    tok.save(chemin)

    # fertilité : jetons par mot et par caractère, sur des lignes non vues à l'entraînement
    test = list(lignes(args.corpus, args.lignes + 20000))[args.lignes:]
    mots = sum(len(l.split()) for l in test)
    chars = sum(len(l) for l in test)
    jetons = sum(len(tok.encode(l).ids) for l in test)
    mesure = {"vocab": args.vocab, "lignes_test": len(test),
              "jetons_par_mot": jetons / max(mots, 1),
              "jetons_par_caractere": jetons / max(chars, 1)}
    with open(os.path.join(args.out, f"fertilite_{args.vocab}.json"), "w", encoding="utf-8") as f:
        json.dump(mesure, f, ensure_ascii=False, indent=2)
    print(f"{chemin} · {tok.get_vocab_size()} entrées")
    print(f"fertilité : {mesure['jetons_par_mot']:.3f} jetons par mot · "
          f"{mesure['jetons_par_caractere']:.3f} par caractère ({len(test)} lignes de test)")
    exemple = "Azul fell-awen, amek tettiliḍ ass-a ?"
    print("exemple :", exemple)
    print("  ->", tok.encode(exemple).tokens)


if __name__ == "__main__":
    main()
