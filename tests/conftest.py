"""Fixtures partagées par la suite de tests.

On n'utilise **jamais** le vrai fichier de données ni un vrai modèle entraîné :
les tests doivent être reproductibles, rapides et indépendants de l'exécution
préalable d'un entraînement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from senegal_rental_price.data.preprocessing import CleaningConfig
from senegal_rental_price.features.build_features import FeatureConfig, feature_names
from senegal_rental_price.models.predict import ModelBundle, ModelMetadata


@pytest.fixture
def cleaning_config() -> CleaningConfig:
    """Configuration de nettoyage par défaut."""
    return CleaningConfig(premium_quartiers=("almadies", "saly"))


@pytest.fixture
def feature_config() -> FeatureConfig:
    """Configuration de features par défaut."""
    return FeatureConfig()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Jeu brut miniature reproduisant les défauts observés dans les vraies données.

    Contient volontairement : un bien commercial, un loyer aberrant, une ville
    hors périmètre, des valeurs manquantes et une incohérence ``meuble``.
    """
    return pd.DataFrame(
        [
            {  # 0 - valide, meublé via la colonne equipements uniquement
                "id": "a1",
                "ville": "Dakar",
                "quartier": "Almadies",
                "type_bien": "appartement",
                "surface_m2": 70.0,
                "surface_estimee": True,
                "nb_pieces": 3.0,
                "nb_chambres": 2.0,
                "meuble": False,
                "equipements": "piscine|climatisation|meuble",
                "prix_loyer_mensuel": 900_000,
                "titre": "Bel appartement aux Almadies",
                "adresse": None,
                "date_publication": "2026-06-23T09:48:33Z",
            },
            {  # 1 - valide, valeurs manquantes à imputer
                "id": "a2",
                "ville": "Thiès",
                "quartier": "Saly",
                "type_bien": "maison",
                "surface_m2": 150.0,
                "surface_estimee": False,
                "nb_pieces": np.nan,
                "nb_chambres": np.nan,
                "meuble": np.nan,
                "equipements": None,
                "prix_loyer_mensuel": 600_000,
                "titre": "Maison à louer",
                "adresse": None,
                "date_publication": "2026-06-20T09:48:33Z",
            },
            {  # 2 - valide
                "id": "a3",
                "ville": "Dakar",
                "quartier": "Ouakam",
                "type_bien": "maison",
                "surface_m2": 200.0,
                "surface_estimee": False,
                "nb_pieces": 5.0,
                "nb_chambres": 4.0,
                "meuble": True,
                "equipements": "jardin|parking",
                "prix_loyer_mensuel": 1_500_000,
                "titre": "Villa spacieuse",
                "adresse": None,
                "date_publication": "2026-06-19T09:48:33Z",
            },
            {  # 3 - À ÉCARTER : bien commercial
                "id": "b1",
                "ville": "Dakar",
                "quartier": "Plateau",
                "type_bien": "maison",
                "surface_m2": 88.0,
                "surface_estimee": False,
                "nb_pieces": 3.0,
                "nb_chambres": 1.0,
                "meuble": False,
                "equipements": "parking",
                "prix_loyer_mensuel": 800_000,
                "titre": "Local commercial à louer - La Rotonde",
                "adresse": None,
                "date_publication": "2026-06-18T09:48:33Z",
            },
            {  # 4 - À ÉCARTER : loyer aberrant (trop bas)
                "id": "b2",
                "ville": "Dakar",
                "quartier": "Mermoz",
                "type_bien": "appartement",
                "surface_m2": 60.0,
                "surface_estimee": False,
                "nb_pieces": 2.0,
                "nb_chambres": 1.0,
                "meuble": False,
                "equipements": "",
                "prix_loyer_mensuel": 3_500,
                "titre": "Appartement à louer",
                "adresse": None,
                "date_publication": "2026-06-17T09:48:33Z",
            },
            {  # 5 - À ÉCARTER : ville hors périmètre
                "id": "b3",
                "ville": "Diakhirate",
                "quartier": "Nouvel Horizon",
                "type_bien": "appartement",
                "surface_m2": 75.0,
                "surface_estimee": True,
                "nb_pieces": 4.0,
                "nb_chambres": 3.0,
                "meuble": np.nan,
                "equipements": None,
                "prix_loyer_mensuel": 170_000,
                "titre": "Appartement à louer",
                "adresse": None,
                "date_publication": "2026-06-12T21:07:45Z",
            },
        ]
    )


class FakeEstimator:
    """Estimateur factice déterministe (évite d'entraîner un vrai modèle en test)."""

    def __init__(self, value: float = 750_000.0) -> None:
        self.value = value
        self.last_input: pd.DataFrame | None = None

    def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Retourne une constante, en mémorisant l'entrée reçue."""
        self.last_input = features
        return np.full(len(features), self.value, dtype=np.float64)

    def fit(self, features: pd.DataFrame, target: Any) -> FakeEstimator:
        """Ne fait rien : présent pour respecter l'interface scikit-learn."""
        return self


@pytest.fixture
def fake_estimator() -> FakeEstimator:
    """Estimateur factice renvoyant toujours 750 000 FCFA."""
    return FakeEstimator()


@pytest.fixture
def model_bundle(fake_estimator: FakeEstimator, feature_config: FeatureConfig) -> ModelBundle:
    """Bundle complet basé sur l'estimateur factice."""
    metadata = ModelMetadata(
        model_name="fake_model",
        model_version="0.0.1-test",
        trained_at="2026-01-01T00:00:00+00:00",
        metrics={"rmse": 1000.0, "mae": 800.0, "r2": 0.5},
        n_train=100,
        feature_names=feature_names(feature_config),
    )
    return ModelBundle(estimator=fake_estimator, feature_config=feature_config, metadata=metadata)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Payload valide pour ``POST /predict``."""
    return {
        "ville": "Dakar",
        "quartier": "Almadies",
        "type_bien": "appartement",
        "surface_m2": 70.0,
        "nb_pieces": 3,
        "nb_chambres": 2,
        "meuble": True,
        "equipements": ["climatisation", "parking"],
    }
