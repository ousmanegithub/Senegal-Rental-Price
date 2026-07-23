"""Nettoyage et prétraitement des annonces immobilières brutes.

Ce module ne contient **que** de la logique pure et testable : les chemins et
seuils proviennent d'un :class:`CleaningConfig` (construit depuis Hydra en amont),
jamais codés en dur. Il gère les problèmes de qualité identifiés lors de
l'exploration (cf. ``notebooks/01_exploration.ipynb``) :

* colonne ``meuble`` incohérente avec le token ``meuble`` de ``equipements`` ;
* annonces non résidentielles (locaux commerciaux, bureaux, entrepôts) ;
* prix aberrants (loyers implausibles) ;
* valeurs manquantes sur ``nb_pieces`` / ``nb_chambres`` / ``meuble``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

# Colonnes purement informatives, inutiles pour la modélisation.
_DROP_COLUMNS: tuple[str, ...] = ("id", "adresse", "date_publication")


@dataclass(frozen=True)
class CleaningConfig:
    """Paramètres de nettoyage (miroir typé de ``conf/data/default.yaml``)."""

    target: str = "prix_loyer_mensuel"
    min_price: float = 150_000
    max_price: float = 10_000_000
    min_surface: float = 9
    max_surface: float = 2000
    known_villes: tuple[str, ...] = ("Dakar", "Thiès")
    known_types: tuple[str, ...] = ("appartement", "maison")
    commercial_keywords: tuple[str, ...] = (
        "commercial",
        "bureau",
        "entrep",
        "local",
        "immeuble",
    )
    known_equipements: tuple[str, ...] = (
        "piscine",
        "climatisation",
        "gardiennage",
        "parking",
        "jardin",
        "terrasse",
        "salle_de_sport",
        "wifi",
        "meuble",
    )
    premium_quartiers: tuple[str, ...] = field(default_factory=tuple)


def load_raw(path: str | Path, sep: str = ";", encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Charge le CSV brut d'annonces.

    Args:
        path: Chemin du fichier CSV.
        sep: Séparateur de colonnes.
        encoding: Encodage (le scraping produit de l'UTF-8 avec BOM).

    Returns:
        Le DataFrame brut, non modifié.
    """
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    logger.info("Données brutes chargées : %d lignes, %d colonnes", len(df), df.shape[1])
    return df


def _parse_bool(value: object) -> bool:
    """Convertit une valeur hétérogène (``"True"``, ``True``, ``NaN``...) en booléen."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "oui", "yes"}


def split_equipements(value: object) -> list[str]:
    """Découpe la chaîne ``equipements`` (``"piscine|parking"``) en liste normalisée.

    Args:
        value: Valeur brute de la cellule (peut être ``NaN``).

    Returns:
        Liste de tokens en minuscules, sans doublon, vide si absent.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    tokens = [tok.strip().lower() for tok in str(value).split("|") if tok.strip()]
    return list(dict.fromkeys(tokens))


def reconcile_meuble(df: pd.DataFrame) -> pd.Series[bool]:
    """Réconcilie la colonne ``meuble`` avec le token ``meuble`` des équipements.

    La colonne brute est peu fiable (rarement renseignée) alors que le token
    ``meuble`` apparaît fréquemment dans ``equipements``. On considère le bien
    meublé si **l'une des deux** sources l'indique.

    Args:
        df: DataFrame contenant ``meuble`` et ``equipements``.

    Returns:
        Une série booléenne alignée sur l'index de ``df``.
    """
    col_flag = df["meuble"].map(_parse_bool)
    eq_flag = df["equipements"].map(lambda v: "meuble" in split_equipements(v))
    reconciled = col_flag | eq_flag
    return reconciled.astype(bool)


def filter_listings(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Écarte les annonces non résidentielles et les valeurs hors bornes.

    Args:
        df: DataFrame brut (doit contenir ``titre`` et la cible).
        cfg: Paramètres de nettoyage.

    Returns:
        Un DataFrame filtré (copie), sans les lignes aberrantes.
    """
    n_before = len(df)
    title = df["titre"].fillna("").str.lower()
    is_commercial = title.apply(lambda t: any(kw in t for kw in cfg.commercial_keywords))

    price = df[cfg.target]
    surface = df["surface_m2"]
    keep = (
        ~is_commercial
        & price.between(cfg.min_price, cfg.max_price)
        & surface.between(cfg.min_surface, cfg.max_surface)
        & df["ville"].isin(cfg.known_villes)
        & df["type_bien"].isin(cfg.known_types)
    )
    filtered = df.loc[keep].copy()
    logger.info(
        "Filtrage : %d -> %d lignes (%d écartées : commercial/aberrant)",
        n_before,
        len(filtered),
        n_before - len(filtered),
    )
    return filtered


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes numériques par la médiane par type de bien.

    Args:
        df: DataFrame filtré.

    Returns:
        Un DataFrame (copie) sans ``NaN`` sur ``nb_pieces`` / ``nb_chambres``.
    """
    out = df.copy()
    for col in ("nb_pieces", "nb_chambres"):
        median_by_type = out.groupby("type_bien")[col].transform("median")
        out[col] = out[col].fillna(median_by_type)
        # Repli global si un type entier était manquant.
        out[col] = out[col].fillna(out[col].median()).round().astype(int)
    return out


def clean(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Pipeline de nettoyage complet : filtrage -> imputation -> normalisation.

    Args:
        df: DataFrame brut issu de :func:`load_raw`.
        cfg: Paramètres de nettoyage.

    Returns:
        Un DataFrame propre, prêt pour la construction de features.
    """
    out = filter_listings(df, cfg)
    out = impute_missing(out)
    out["meuble"] = reconcile_meuble(out)
    out["equipements"] = out["equipements"].map(split_equipements)
    out["surface_estimee"] = out["surface_estimee"].map(_parse_bool)
    out = out.drop(columns=[c for c in _DROP_COLUMNS if c in out.columns])
    logger.info("Nettoyage terminé : %d lignes propres", len(out))
    return out.reset_index(drop=True)
