"""Test d'intégration de l'orchestration d'entraînement.

On exécute :func:`run_training` de bout en bout sur un CSV temporaire, avec un
modèle volontairement minuscule : l'objectif est de vérifier la **chaîne**
(chargement → nettoyage → features → fit → évaluation → sérialisation), pas la
qualité de la prédiction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from senegal_rental_price.models.predict import load_bundle
from senegal_rental_price.models.train import run_training


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    """Écrit un CSV d'entraînement minimal mais suffisant pour un split train/test."""
    rows = []
    for index in range(40):
        is_dakar = index % 2 == 0
        rows.append(
            {
                "id": f"id-{index}",
                "ville": "Dakar" if is_dakar else "Thiès",
                "quartier": "Almadies" if is_dakar else "Saly",
                "type_bien": "appartement" if index % 3 else "maison",
                "surface_m2": 50.0 + index * 5,
                "surface_estimee": index % 2 == 0,
                "nb_pieces": 2 + index % 4,
                "nb_chambres": 1 + index % 3,
                "meuble": index % 4 == 0,
                "equipements": "piscine|parking" if is_dakar else "jardin",
                "prix_loyer_mensuel": 400_000 + index * 25_000,
                "titre": "Bien à louer",
                "adresse": "",
                "date_publication": "2026-06-01T00:00:00Z",
            }
        )
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def training_config(training_csv: Path, tmp_path: Path) -> DictConfig:
    """Configuration Hydra équivalente, construite à la main pour le test."""
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "models"),
            "seed": 42,
            "test_size": 0.25,
            "log_level": "WARNING",
            "mlflow": {
                "tracking_uri": str(tmp_path / "mlruns"),
                "experiment_name": "test-experiment",
                "registered_model_name": None,
            },
            "model": {
                "name": "ridge",
                "estimator": "ridge",
                "params": {"alpha": 1.0},
            },
            "data": {
                "raw_path": str(training_csv),
                "sep": ";",
                "encoding": "utf-8-sig",
                "target": "prix_loyer_mensuel",
                "known_villes": ["Dakar", "Thiès"],
                "known_types": ["appartement", "maison"],
                "min_price": 150_000,
                "max_price": 10_000_000,
                "min_surface": 9,
                "max_surface": 2000,
                "commercial_keywords": ["commercial", "bureau"],
                "known_equipements": [
                    "piscine",
                    "climatisation",
                    "gardiennage",
                    "parking",
                    "jardin",
                    "terrasse",
                    "salle_de_sport",
                    "wifi",
                    "meuble",
                ],
                "premium_quartiers": ["almadies", "saly"],
            },
        }
    )
    assert isinstance(config, DictConfig)
    return config


class TestRunTraining:
    """Chaîne d'entraînement complète."""

    def test_returns_a_bundle_with_metadata(self, training_config: DictConfig) -> None:
        bundle = run_training(training_config)
        assert bundle.metadata.model_name == "ridge"
        assert bundle.metadata.n_train > 0

    def test_computes_all_metrics(self, training_config: DictConfig) -> None:
        metrics = run_training(training_config).metadata.metrics
        assert set(metrics) == {"rmse", "mae", "r2"}

    def test_persists_the_model_on_disk(self, training_config: DictConfig) -> None:
        run_training(training_config)
        reloaded = load_bundle(training_config.output_dir)
        assert reloaded.metadata.model_name == "ridge"

    def test_trained_model_can_predict(self, training_config: DictConfig) -> None:
        from senegal_rental_price.models.predict import predict_price

        bundle = run_training(training_config)
        price = predict_price(
            bundle,
            {
                "ville": "Dakar",
                "quartier": "Almadies",
                "type_bien": "appartement",
                "surface_m2": 80.0,
                "nb_pieces": 3,
                "nb_chambres": 2,
                "meuble": True,
                "equipements": ["piscine"],
            },
        )
        assert price > 0

    def test_respects_the_configured_test_size(self, training_config: DictConfig) -> None:
        """40 lignes, test_size=0.25 -> 30 exemples d'entraînement."""
        assert run_training(training_config).metadata.n_train == 30

    def test_supports_switching_estimator_via_config(self, training_config: DictConfig) -> None:
        """Changer de modèle ne demande aucune modification du code."""
        training_config.model.name = "random_forest"
        training_config.model.params = OmegaConf.create({"n_estimators": 5, "random_state": 42})
        assert run_training(training_config).metadata.model_name == "random_forest"

    def test_training_succeeds_even_if_mlflow_fails(
        self, training_config: DictConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Une panne de tracking ne doit jamais bloquer l'entraînement.

        On simule l'indisponibilité de MLflow plutôt que de viser un serveur
        injoignable : le test reste instantané (pas de timeout réseau).
        """
        import mlflow

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Serveur MLflow indisponible")

        monkeypatch.setattr(mlflow, "set_experiment", _boom)
        bundle = run_training(training_config)
        assert bundle.metadata.n_train > 0
