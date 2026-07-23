"""Application FastAPI exposant le service de prédiction de loyer.

Endpoints :

* ``GET  /health``     — état du service ;
* ``GET  /model/info`` — métadonnées du modèle chargé ;
* ``POST /predict``    — estimation du loyer à partir des caractéristiques du bien.

Le modèle est chargé **au démarrage** via le gestionnaire ``lifespan`` et stocké
dans ``app.state``, jamais rechargé par requête.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import get_model_bundle, get_settings, load_model_bundle
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RentalFeatures,
)
from senegal_rental_price.models.predict import ModelBundle, predict_price
from senegal_rental_price.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Service d'estimation du **loyer mensuel** d'un bien immobilier au Sénégal
(Dakar et région de Thiès / Petite-Côte) à partir de ses caractéristiques.

Le modèle est entraîné sur des annonces de location réelles collectées par
scraping, puis nettoyées (cf. `data/README.md`).
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle au démarrage et libère les ressources à l'arrêt."""
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        app.state.model_bundle = load_model_bundle(settings)
        logger.info("API prête : modèle '%s' chargé", app.state.model_bundle.metadata.model_name)
    except FileNotFoundError as exc:
        # On ne fait pas échouer le démarrage : /health doit rester interrogeable
        # et signaler l'absence de modèle (utile en orchestration / healthcheck).
        app.state.model_bundle = None
        logger.error("Démarrage sans modèle : %s", exc)
    yield
    app.state.model_bundle = None
    logger.info("Arrêt de l'API")


app = FastAPI(
    title="Sénégal — API de prédiction du prix de location",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    contact={"name": "Projet M2 DSIA"},
    license_info={"name": "MIT"},
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Renvoie une erreur ``422`` lisible plutôt qu'une trace brute.

    Les erreurs Pydantic peuvent contenir des objets non sérialisables (``ctx``
    embarque l'exception d'origine) : on ne conserve donc que les champs utiles
    et sérialisables.
    """
    errors = [
        {
            "champ": ".".join(str(part) for part in err.get("loc", ())),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    logger.warning("Payload invalide sur %s : %s", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"detail": "Payload invalide.", "errors": errors},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état du service",
    tags=["Monitoring"],
)
def health(request: Request) -> HealthResponse:
    """Indique si le service répond et si un modèle est chargé."""
    loaded = getattr(request.app.state, "model_bundle", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Métadonnées du modèle chargé",
    tags=["Modèle"],
    responses={503: {"model": ErrorResponse, "description": "Aucun modèle chargé"}},
)
def model_info(
    bundle: Annotated[ModelBundle, Depends(get_model_bundle)],
) -> ModelInfoResponse:
    """Retourne la version, la date d'entraînement et les métriques du modèle."""
    meta = bundle.metadata
    return ModelInfoResponse(
        model_name=meta.model_name,
        model_version=meta.model_version,
        trained_at=meta.trained_at,
        metrics=meta.metrics,
        n_train=meta.n_train,
        feature_names=meta.feature_names,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimation du loyer mensuel",
    tags=["Prédiction"],
    responses={
        422: {"model": ErrorResponse, "description": "Caractéristiques invalides"},
        503: {"model": ErrorResponse, "description": "Aucun modèle chargé"},
    },
)
def predict(
    features: RentalFeatures,
    bundle: Annotated[ModelBundle, Depends(get_model_bundle)],
) -> PredictionResponse:
    """Estime le loyer mensuel (FCFA) d'un bien à partir de ses caractéristiques."""
    estimated = predict_price(bundle, features.to_listing())
    logger.info(
        "Prédiction : %s %s %.0f m² -> %.0f FCFA",
        features.ville.value,
        features.type_bien.value,
        features.surface_m2,
        estimated,
    )
    return PredictionResponse(
        prix_loyer_mensuel_estime=estimated,
        model_version=bundle.metadata.model_version,
    )
