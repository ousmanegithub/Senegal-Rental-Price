# Données — provenance et description

## Provenance

Jeu de données constitué par **scraping** d'annonces de location immobilière au
Sénégal sur [NeoBien](https://neobien.com/) (usage strictement académique,
dans le respect des conditions d'utilisation du site). Le fichier brut est
`raw/locations.csv` (séparateur `;`, encodage UTF-8 avec BOM).

> Les dossiers `raw/` et `processed/` ne sont **pas versionnés** (cf. `.gitignore`).

## Schéma brut (`raw/locations.csv`)

| Colonne | Type | Description |
|---|---|---|
| `id` | str | Identifiant de l'annonce (supprimé au nettoyage) |
| `ville` | str | Ville / région administrative |
| `quartier` | str | Quartier (utile surtout pour Dakar) |
| `type_bien` | str | `appartement`, `maison` |
| `surface_m2` | float | Surface en m² |
| `surface_estimee` | bool | `True` si la surface a été **estimée** (regex sur le titre) au scraping |
| `nb_pieces` | float | Nombre de pièces (valeurs manquantes possibles) |
| `nb_chambres` | float | Nombre de chambres (valeurs manquantes possibles) |
| `meuble` | str/bool | Meublé — **peu fiable** (voir plus bas) |
| `equipements` | str | Équipements séparés par `\|` (`piscine\|parking\|...`) |
| `prix_loyer_mensuel` | int | **Cible** — loyer mensuel en FCFA |
| `titre` | str | Titre de l'annonce (utilisé pour détecter les biens commerciaux) |
| `adresse` | str | Quasi toujours vide (supprimée) |
| `date_publication` | str | Date ISO (supprimée) |

## Caractéristiques (166 annonces brutes)

- **Villes** : Thiès (94), Dakar (71), Diakhirate (1). Les quartiers rattachés à
  « Thiès » correspondent en réalité à la Petite-Côte (Saly, Ngaparou, Somone…).
- **Types** : maison (98), appartement (68).
- **Prix brut** : 3 500 à 15 000 000 FCFA (fortes valeurs aberrantes).
- **Équipements** (fréquence) : piscine 95, gardiennage 89, terrasse 73,
  meuble 60, parking 54, jardin 54, climatisation 27, salle_de_sport 16, wifi 8.

## Problèmes de qualité identifiés et traitement

1. **`meuble` incohérent** — la colonne n'indique que 3 biens meublés, alors que
   le token `meuble` apparaît 60 fois dans `equipements`. → **Réconciliation** :
   meublé si la colonne **ou** les équipements l'indiquent (→ 53 meublés).
2. **Biens non résidentiels** — plusieurs « maisons » sont des locaux
   commerciaux, bureaux ou entrepôts (loyers 3 500–12 000 FCFA), plus un
   « immeuble R+1 » à 15 000 000. → **Filtrés** par mots-clés du titre et bornes
   de prix/surface plausibles (`conf/data/default.yaml`).
3. **`Diakhirate` (1 annonce)** — hors périmètre, insuffisant pour généraliser.
   → **Écarté** (restriction aux villes connues : Dakar, Thiès).
4. **Valeurs manquantes** — `nb_pieces` (6), `nb_chambres` (7) → imputées par la
   médiane par type de bien ; `equipements` (11) → liste vide.
5. **`adresse`** — 163/166 vides → **colonne supprimée**.

Après nettoyage : **155 annonces** propres, prix 210 000–3 250 000 FCFA
(médiane 1 000 000).

Toute la logique est tracée dans `notebooks/01_exploration.ipynb` et implémentée
dans `src/senegal_rental_price/data/preprocessing.py`.
