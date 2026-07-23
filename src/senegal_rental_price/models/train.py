"""Entraînement d'un modèle de prédiction de loyer, piloté par Hydra + MLflow.

Exemples d'utilisation (une fois le package installé) :

    # Modèle par défaut (random_forest)
    python -m senegal_rental_price.models.train

    # Autre modèle + surcharge d'hyperparamètre, sans toucher au code
    python -m senegal_rental_price.models.train model=xgboost model.params.max_depth=8

    # Comparaison des trois modèles en une commande (Hydra multirun)
    python -m senegal_rental_price.models.train -m model=ridge,random_forest,xgboost

Chaque exécution est tracée comme un run MLflow (paramètres + métriques), ce qui
permet de comparer les modèles dans l'UI MLflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import hydra
import mlflow
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from senegal_rental_price import __version__
from senegal_rental_price.data.preprocessing import CleaningConfig, clean, load_raw
from senegal_rental_price.features.build_features import (
    FeatureConfig,
    build_features,
    feature_names,
)
from senegal_rental_price.models.predict import ModelBundle, ModelMetadata, save_bundle
from senegal_rental_price.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def build_estimator(name: str, params: dict[str, Any]) -> Any:
    """Instancie l'estimateur demandé à partir de son nom et de ses paramètres.

    Args:
        name: ``"ridge"``, ``"random_forest"`` ou ``"xgboost"``.
        params: Hyperparamètres passés au constructeur.

    Returns:
        Un estimateur compatible scikit-learn (interface ``fit``/``predict``).

    Raises:
        ValueError: Si le nom d'estimateur est inconnu.
    """
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "xgboost":
        return XGBRegressor(**params)
    raise ValueError(f"Estimateur inconnu : {name!r}")


def evaluate(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> dict[str, float]:
    """Calcule les métriques de régression (RMSE, MAE, R²).

    Args:
        y_true: Valeurs réelles du loyer.
        y_pred: Valeurs prédites.

    Returns:
        Un dictionnaire ``{"rmse": ..., "mae": ..., "r2": ...}``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _cleaning_config(cfg: DictConfig) -> CleaningConfig:
    """Construit un :class:`CleaningConfig` typé depuis la config Hydra."""
    d = cfg.data
    return CleaningConfig(
        target=d.target,
        min_price=d.min_price,
        max_price=d.max_price,
        min_surface=d.min_surface,
        max_surface=d.max_surface,
        known_villes=tuple(d.known_villes),
        known_types=tuple(d.known_types),
        commercial_keywords=tuple(d.commercial_keywords),
        known_equipements=tuple(d.known_equipements),
        premium_quartiers=tuple(d.premium_quartiers),
    )


def _feature_config(cfg: DictConfig) -> FeatureConfig:
    """Construit un :class:`FeatureConfig` typé depuis la config Hydra."""
    d = cfg.data
    # Les flags d'équipements excluent "meuble" (déjà une colonne dédiée).
    equipements = tuple(e for e in d.known_equipements if e != "meuble")
    return FeatureConfig(
        known_villes=tuple(d.known_villes),
        known_types=tuple(d.known_types),
        known_equipements=equipements,
        premium_quartiers=tuple(d.premium_quartiers),
    )


def run_training(cfg: DictConfig) -> ModelBundle:
    """Exécute l'entraînement complet et retourne le bundle sérialisable.

    Cette fonction concentre toute la logique métier (hors décorateur Hydra),
    ce qui la rend directement testable.

    Args:
        cfg: Configuration Hydra résolue.

    Returns:
        Le :class:`ModelBundle` entraîné (également sauvegardé sur disque).
    """
    clean_cfg = _cleaning_config(cfg)
    feat_cfg = _feature_config(cfg)

    raw = load_raw(cfg.data.raw_path, sep=cfg.data.sep, encoding=cfg.data.encoding)
    data = clean(raw, clean_cfg)

    features = build_features(data, feat_cfg)
    target = data[cfg.data.target].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=cfg.test_size, random_state=cfg.seed
    )

    raw_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    params = cast("dict[str, Any]", raw_params)
    estimator = build_estimator(cfg.model.name, params)
    logger.info("Entraînement du modèle '%s' sur %d exemples", cfg.model.name, len(x_train))
    estimator.fit(x_train, y_train)

    metrics = evaluate(y_test.to_numpy(), estimator.predict(x_test))
    logger.info(
        "Performances (test) — RMSE=%.0f | MAE=%.0f | R2=%.3f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = ModelMetadata(
        model_name=cfg.model.name,
        model_version=f"{__version__}+{datetime.now(UTC):%Y%m%d%H%M%S}",
        trained_at=trained_at,
        metrics=metrics,
        n_train=len(x_train),
        feature_names=feature_names(feat_cfg),
    )
    bundle = ModelBundle(estimator=estimator, feature_config=feat_cfg, metadata=metadata)

    _track_mlflow(cfg, params, metrics, estimator, bundle)
    save_bundle(bundle, cfg.output_dir)
    return bundle


def _track_mlflow(
    cfg: DictConfig,
    params: dict[str, Any],
    metrics: dict[str, float],
    estimator: Any,
    bundle: ModelBundle,
) -> None:
    """Enregistre un run MLflow (params, métriques, modèle) de façon robuste.

    Toute erreur MLflow (serveur indisponible...) est journalisée sans faire
    échouer l'entraînement, afin que la sérialisation locale reste garantie.
    """
    try:
        if cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run(run_name=cfg.model.name):
            mlflow.log_params(params)
            mlflow.log_param("model_name", cfg.model.name)
            mlflow.log_param("n_train", bundle.metadata.n_train)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                estimator,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                registered_model_name=cfg.mlflow.registered_model_name,
            )
        logger.info("Run MLflow enregistré (expérience '%s')", cfg.mlflow.experiment_name)
    except Exception as exc:  # noqa: BLE001 - MLflow ne doit jamais bloquer l'entraînement
        logger.warning("Tracking MLflow ignoré (%s)", exc)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Point d'entrée CLI (``senegal-rental-train``) décoré par Hydra."""
    configure_logging(cfg.log_level)
    logger.info("Configuration résolue :\n%s", OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
