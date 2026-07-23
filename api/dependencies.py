"""Dépendances de l'API : configuration et accès au modèle chargé.

Le modèle est chargé **une seule fois au démarrage** (cf. ``lifespan`` dans
:mod:`api.main`) puis exposé aux endpoints via la dépendance
:func:`get_model_bundle`. Aucune désérialisation n'a donc lieu par requête.

La configuration provient de variables d'environnement (cf. §7 du sujet :
paramètres injectés par l'environnement plutôt qu'en dur).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from pydantic_settings import BaseSettings, SettingsConfigDict

from senegal_rental_price.models.predict import ModelBundle, load_bundle
from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    """Configuration de l'API, injectée par variables d'environnement."""

    model_config = SettingsConfigDict(env_prefix="API_", protected_namespaces=())

    model_dir: str = "models/"
    """Répertoire contenant le modèle sérialisé (``API_MODEL_DIR``)."""

    mlflow_tracking_uri: str | None = None
    """URI du serveur MLflow (``API_MLFLOW_TRACKING_URI``). Si absent : chargement local."""

    mlflow_model_uri: str | None = None
    """URI du modèle au Registry, ex. ``models:/senegal-rental-price-model/Production``."""

    log_level: str = "INFO"
    """Niveau de log (``API_LOG_LEVEL``)."""


@lru_cache
def get_settings() -> Settings:
    """Retourne la configuration de l'API (mise en cache)."""
    return Settings()


def load_model_bundle(settings: Settings) -> ModelBundle:
    """Charge le modèle au démarrage, depuis MLflow si configuré, sinon localement.

    La stratégie privilégie le **Model Registry MLflow** (comme vu en cours) et
    retombe sur l'artefact local en cas d'indisponibilité, afin que le service
    reste démarrable hors ligne (démo, CI, conteneur isolé).

    Args:
        settings: Configuration de l'API.

    Returns:
        Le :class:`ModelBundle` prêt à servir les prédictions.
    """
    if settings.mlflow_model_uri:
        try:
            import mlflow

            if settings.mlflow_tracking_uri:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            estimator = mlflow.sklearn.load_model(settings.mlflow_model_uri)
            local = load_bundle(settings.model_dir)
            logger.info("Estimateur chargé depuis le Model Registry MLflow")
            return ModelBundle(
                estimator=estimator,
                feature_config=local.feature_config,
                metadata=local.metadata,
            )
        except Exception as exc:  # noqa: BLE001 - repli volontaire sur l'artefact local
            logger.warning("Chargement MLflow impossible (%s), repli sur le modèle local", exc)

    return load_bundle(settings.model_dir)


def get_model_bundle(request: Request) -> ModelBundle:
    """Dépendance FastAPI exposant le modèle chargé au démarrage.

    Args:
        request: Requête courante (donne accès à ``app.state``).

    Returns:
        Le :class:`ModelBundle` en mémoire.

    Raises:
        HTTPException: ``503`` si aucun modèle n'a pu être chargé au démarrage.
    """
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun modèle chargé : le service n'est pas prêt.",
        )
    return bundle


ModelDependency = Depends(get_model_bundle)
