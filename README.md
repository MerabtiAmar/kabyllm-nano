# KabyLLM-nano — un modèle de langue kabyle qui tient dans un navigateur

Projet personnel (2026), suite de [KabyLLM](https://github.com/MerabtiAmar/kabyllm). La question n'est plus « peut-on entraîner un modèle de langue pour le kabyle ? » mais **jusqu'où peut-on le réduire sans qu'il cesse d'être utile ?** Cible : un fichier de quelques mégaoctets, exécuté dans la page d'un visiteur, sans serveur.

À essayer : [le modèle écrit dans le navigateur](https://merabtiamar.github.io/experiences/kabyllm-nano.html), jeton par jeton, à 8 ms le jeton.

## Le vocabulaire, d'abord

Dans un modèle de cette taille, la table d'embedding pèse autant que les couches. J'ai entraîné quatre tokenizers BPE sur le corpus et mesuré leur **fertilité** (jetons par mot) sur des lignes jamais vues :

| vocabulaire | jetons par mot |
|---|---:|
| 1 024 | 2,642 |
| **2 048** | **2,297** |
| 4 096 | 2,049 |
| 8 192 | 1,863 |
| NLLB-200 (256 204 entrées) | 2,269 |
| Qwen3 (151 669 entrées) | 2,746 |

**Avec 2 048 entrées, le kabyle est découpé aussi finement que par NLLB-200 et son vocabulaire 125 fois plus grand** — et nettement mieux que par Qwen3. Les deux références viennent du rapport de fertilité de KabyLLM.

## Le modèle

Décodeur Transformer : 4 couches, largeur 160, 4 têtes, contexte 192 jetons, embeddings partagés avec la tête de sortie — **1 593 280 paramètres**. Corpus : 123,1 M de caractères de kabyle, soit 47,9 M de jetons.

| | bits par caractère |
|---|---:|
| KabyLLM complet (36,3 M de paramètres) | 1,98 |
| **KabyLLM-nano (1,6 M), run local 4 200 pas** | **2,09** |

Les bits par caractère sont la seule mesure comparable entre modèles qui n'ont pas le même tokenizer : `perte × jetons_par_caractère / ln(2)`.

## La quantification

Chaque tenseur est ramené à des entiers de 8 bits avec sa propre échelle. Le fichier livré pèse **2,07 Mo** (moitié moins qu'en float16), et le coût est mesuré : la perte de validation passe de 3,725 à 3,727, soit **+0,07 %**.

## Ce que ce modèle n'est pas

À cette taille, le modèle apprend l'orthographe, la morphologie et des tournures fréquentes — pas des connaissances. Il ne répond pas aux questions et se contredit volontiers. Le corpus, construit en décodant des PDF, laisse aussi des scories (`(cid:13)` apparaît parfois dans les générations) : le modèle apprend ce qu'on lui donne.

## Lancer

```bash
pip install -r requirements.txt
python src/tokenizer.py --vocab 2048                 # tokenizer + fertilité
python src/prepare.py --vocab 2048                   # corpus -> jetons (uint16)
python src/train.py --pas 20000 --lot 48 --nom nano  # GPU conseillé
python src/echantillons.py --run nano
python src/quantize_export.py --run nano --out ../merabtiamar.github.io/assets/data/kabyllm-nano.js
```

Sur GPU T4, `notebooks/kaggle_nano.ipynb` fait tout en une vingtaine de minutes.

## Contenu

```
src/tokenizer.py        BPE au niveau octet + mesure de fertilité
src/prepare.py          tokenisation du corpus en mémoire disque (uint16)
src/model.py            décodeur Transformer minuscule, embeddings partagés
src/train.py            entraînement, validation, bits par caractère
src/echantillons.py     générations à partir d'amorces
src/quantize_export.py  quantification 8 bits, coût mesuré, export pour le navigateur
notebooks/kaggle_nano.ipynb   le même pipeline, prêt pour un T4
artefacts/              tokenizers et fertilités (les jetons du corpus ne sont pas versionnés)
```

## Données

Corpus kabyle rassemblé pour [KabyLLM](https://github.com/MerabtiAmar/kabyllm) (textes publics, PDF décodés). Il n'est pas versionné ici.

## Portage navigateur

Découpage BPE identique à Python (8 textes sur 8), cache clés-valeurs donnant exactement les mêmes logits que le recalcul complet, écart avec PyTorch de 1,4·10⁻⁵, 8 ms par jeton.

## Licence

Code distribué sous [licence MIT](LICENSE).

## Auteur

**Amar Merabti** — 2026.
