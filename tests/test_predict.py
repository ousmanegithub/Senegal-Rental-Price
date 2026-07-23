"""Tests de la couche modèle : prédiction, sérialisation, helpers d'entraînement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from senegal_rental_price.features.build_features import feature_names
from senegal_rental_price.models.predict import (
    ModelBundle,
    load_bundle,
    predict_price,
    save_bundle,
)
from senegal_rental_price.models.train import build_estimator, evaluate

from .conftest import FakeEstimator

LISTING: dict[str, Any] = {
    "ville": "Dakar",
    "quartier": "Almadies",
    "type_bien": "appartement",
    "surface_m2": 70.0,
    "nb_pieces": 3,
    "nb_chambres": 2,
    "meuble": True,
    "equipements": ["climatisation", "parking"],
}


class TestPredictPrice:
    """Prédiction sur un bien unique, avec un estimateur mocké."""

    def test_returns_a_float(self, model_bundle: ModelBundle) -> None:
        assert isinstance(predict_price(model_bundle, LISTING), float)

    def test_returns_the_estimator_output(self, model_bundle: ModelBundle) -> None:
        assert predict_price(model_bundle, LISTING) == 750_000.0

    def test_feeds_the_expected_feature_schema(
        self, model_bundle: ModelBundle, fake_estimator: FakeEstimator
    ) -> None:
        predict_price(model_bundle, LISTING)
        assert fake_estimator.last_input is not None
        expected = feature_names(model_bundle.feature_config)
        assert list(fake_estimator.last_input.columns) == expected

    def test_feeds_exactly_one_row(
        self, model_bundle: ModelBundle, fake_estimator: FakeEstimator
    ) -> None:
        predict_price(model_bundle, LISTING)
        assert fake_estimator.last_input is not None
        assert len(fake_estimator.last_input) == 1

    def test_negative_prediction_is_clamped_to_zero(
        self, model_bundle: ModelBundle, fake_estimator: FakeEstimator
    ) -> None:
        fake_estimator.value = -50_000.0
        assert predict_price(model_bundle, LISTING) == 0.0

    def test_works_without_optional_fields(self, model_bundle: ModelBundle) -> None:
        minimal = {k: v for k, v in LISTING.items() if k != "quartier"}
        assert predict_price(model_bundle, minimal) == 750_000.0


class TestBundleSerialisation:
    """Aller-retour disque du bundle."""

    def test_save_then_load_preserves_metadata(
        self, model_bundle: ModelBundle, tmp_path: Path
    ) -> None:
        save_bundle(model_bundle, tmp_path)
        loaded = load_bundle(tmp_path)
        assert loaded.metadata == model_bundle.metadata

    def test_save_writes_readable_metadata_file(
        self, model_bundle: ModelBundle, tmp_path: Path
    ) -> None:
        save_bundle(model_bundle, tmp_path)
        assert (tmp_path / "model_meta.json").exists()

    def test_loaded_bundle_still_predicts(self, model_bundle: ModelBundle, tmp_path: Path) -> None:
        save_bundle(model_bundle, tmp_path)
        assert predict_price(load_bundle(tmp_path), LISTING) == 750_000.0

    def test_load_missing_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Aucun modèle"):
            load_bundle(tmp_path / "vide")


class TestBuildEstimator:
    """Fabrique d'estimateurs pilotée par la configuration."""

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("ridge", {"alpha": 1.0}),
            ("random_forest", {"n_estimators": 5, "random_state": 42}),
            ("xgboost", {"n_estimators": 5, "random_state": 42}),
        ],
    )
    def test_builds_known_estimators(self, name: str, params: dict[str, Any]) -> None:
        estimator = build_estimator(name, params)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")

    def test_applies_hyperparameters(self) -> None:
        estimator = build_estimator("random_forest", {"n_estimators": 7})
        assert estimator.n_estimators == 7

    def test_unknown_estimator_raises(self) -> None:
        with pytest.raises(ValueError, match="Estimateur inconnu"):
            build_estimator("reseau_de_neurones", {})


class TestEvaluate:
    """Calcul des métriques de régression."""

    def test_returns_expected_metric_keys(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        assert set(evaluate(y_true, y_pred)) == {"rmse", "mae", "r2"}

    def test_perfect_prediction_scores_perfectly(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        metrics = evaluate(y_true, y_true)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_mae_is_the_mean_absolute_error(self) -> None:
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        assert evaluate(y_true, y_pred)["mae"] == pytest.approx(15.0)
