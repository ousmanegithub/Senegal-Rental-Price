"""Construction des features à partir des données nettoyées.

La contrainte forte de ce module : produire **exactement les mêmes colonnes**
à l'entraînement (DataFrame de plusieurs lignes) et à l'inférence (un seul bien
reçu par l'API). On y parvient en fixant le vocabulaire catégoriel dans une
:class:`FeatureConfig` plutôt qu'en laissant ``pd.get_dummies`` déduire les
catégories des données présentes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from senegal_rental_price.data.preprocessing import split_equipements
from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

_NUMERIC_BASE: tuple[str, ...] = ("surface_m2", "nb_pieces", "nb_chambres")


@dataclass(frozen=True)
class FeatureConfig:
    """Vocabulaire figé garantissant un schéma de features stable."""

    known_villes: tuple[str, ...] = ("Dakar", "Thiès")
    known_types: tuple[str, ...] = ("appartement", "maison")
    known_equipements: tuple[str, ...] = (
        "piscine",
        "climatisation",
        "gardiennage",
        "parking",
        "jardin",
        "terrasse",
        "salle_de_sport",
        "wifi",
    )
    premium_quartiers: tuple[str, ...] = (
        "almadies",
        "ngor",
        "point e",
        "mermoz",
        "fann",
        "plateau",
        "saly",
        "ngaparou",
        "somone",
    )


def feature_names(cfg: FeatureConfig) -> list[str]:
    """Retourne la liste ordonnée et exhaustive des colonnes de features.

    Args:
        cfg: Vocabulaire catégoriel figé.

    Returns:
        La liste des noms de colonnes, dans un ordre déterministe.
    """
    names = list(_NUMERIC_BASE)
    names += ["meuble", "surface_estimee", "nb_equipements", "quartier_premium"]
    names += [f"equip_{e}" for e in cfg.known_equipements]
    names += [f"ville_{v}" for v in cfg.known_villes]
    names += [f"type_{t}" for t in cfg.known_types]
    return names


def _as_equipement_list(value: object) -> list[str]:
    """Normalise une cellule ``equipements`` qu'elle soit déjà une liste ou une chaîne."""
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value]
    return split_equipements(value)


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Transforme un DataFrame nettoyé en matrice de features numériques.

    Args:
        df: Données nettoyées (issues de :func:`preprocessing.clean`) ou un seul
            bien à prédire. Doit contenir au minimum ``ville``, ``type_bien``,
            ``surface_m2``, ``nb_pieces``, ``nb_chambres``, ``meuble``,
            ``equipements`` et ``quartier``.
        cfg: Vocabulaire catégoriel figé.

    Returns:
        Un DataFrame de features aligné sur :func:`feature_names`, sans ``NaN``.
    """
    out = pd.DataFrame(index=df.index)

    for col in _NUMERIC_BASE:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    out["meuble"] = df["meuble"].astype(bool).astype(int)
    out["surface_estimee"] = (
        df.get("surface_estimee", pd.Series(False, index=df.index)).astype(bool).astype(int)
    )

    equip_lists = df["equipements"].map(_as_equipement_list)
    out["nb_equipements"] = equip_lists.map(len)
    for equip in cfg.known_equipements:
        out[f"equip_{equip}"] = equip_lists.map(lambda lst, e=equip: int(e in lst))

    quartier = df.get("quartier", pd.Series("", index=df.index)).fillna("").str.lower()
    out["quartier_premium"] = quartier.map(
        lambda q: int(any(pq in q for pq in cfg.premium_quartiers))
    )

    for ville in cfg.known_villes:
        out[f"ville_{ville}"] = (df["ville"] == ville).astype(int)
    for type_bien in cfg.known_types:
        out[f"type_{type_bien}"] = (df["type_bien"] == type_bien).astype(int)

    out = out.reindex(columns=feature_names(cfg), fill_value=0)
    out = out.fillna(0)
    logger.debug("Features construites : %d lignes x %d colonnes", *out.shape)
    return out
