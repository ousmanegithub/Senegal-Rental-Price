"""Chargement d'un modèle entraîné et prédiction sur un bien unique.

Ce module définit le « bundle » sérialisé (estimateur + vocabulaire de features
+ métadonnées) produit par :mod:`senegal_rental_price.models.train` et consommé
par l'API. Isoler ces objets ici permet à l'API et aux tests de charger un
modèle sans dépendre de la logique d'entraînement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from senegal_rental_price.features.build_features import FeatureConfig, build_features
from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

_BUNDLE_FILENAME = "model.joblib"
_META_FILENAME = "model_meta.json"


@dataclass(frozen=True)
class ModelMetadata:
    """Métadonnées exposées par l'endpoint ``/model/info``."""

    model_name: str
    model_version: str
    trained_at: str
    metrics: dict[str, float]
    n_train: int
    feature_names: list[str]


@dataclass(frozen=True)
class ModelBundle:
    """Tout ce dont l'API a besoin pour prédire : estimateur + features + méta."""

    estimator: Any  # estimateur compatible scikit-learn (fit/predict)
    feature_config: FeatureConfig
    metadata: ModelMetadata


def save_bundle(bundle: ModelBundle, output_dir: str | Path) -> Path:
    """Sérialise le bundle sur disque (``model.joblib`` + ``model_meta.json``).

    Args:
        bundle: Le bundle à sauvegarder.
        output_dir: Répertoire de destination (créé si absent).

    Returns:
        Le chemin du fichier joblib écrit.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / _BUNDLE_FILENAME
    joblib.dump(bundle, bundle_path)
    # Copie lisible des métadonnées (pratique pour l'inspection / debug).
    (out / _META_FILENAME).write_text(
        json.dumps(asdict(bundle.metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Modèle sérialisé dans %s", bundle_path)
    return bundle_path


def load_bundle(output_dir: str | Path) -> ModelBundle:
    """Charge un bundle sérialisé depuis un répertoire.

    Args:
        output_dir: Répertoire contenant ``model.joblib``.

    Returns:
        Le :class:`ModelBundle` désérialisé.

    Raises:
        FileNotFoundError: Si aucun modèle n'est trouvé.
    """
    bundle_path = Path(output_dir) / _BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Aucun modèle trouvé à {bundle_path}. Lancez d'abord l'entraînement "
            "(senegal-rental-train)."
        )
    bundle: ModelBundle = joblib.load(bundle_path)
    logger.info(
        "Modèle chargé : %s v%s (%d features)",
        bundle.metadata.model_name,
        bundle.metadata.model_version,
        len(bundle.metadata.feature_names),
    )
    return bundle


def predict_price(bundle: ModelBundle, listing: Mapping[str, Any]) -> float:
    """Prédit le loyer mensuel d'un bien unique.

    Args:
        bundle: Le modèle chargé.
        listing: Caractéristiques du bien. Clés attendues : ``ville``,
            ``type_bien``, ``surface_m2``, ``nb_pieces``, ``nb_chambres``,
            ``meuble``, ``equipements`` (liste), et optionnellement ``quartier``
            et ``surface_estimee``.

    Returns:
        Le loyer mensuel estimé en FCFA (arrondi, positif).
    """
    frame = pd.DataFrame([dict(listing)])
    features = build_features(frame, bundle.feature_config)
    prediction = bundle.estimator.predict(features)
    value = float(prediction[0])
    return round(max(value, 0.0), 2)
