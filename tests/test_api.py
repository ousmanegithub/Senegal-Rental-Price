"""Tests des endpoints de l'API (TestClient FastAPI).

Le modèle réel n'est jamais chargé : la dépendance :func:`get_model_bundle` est
surchargée par un bundle factice, ce qui rend les tests rapides et indépendants
d'un entraînement préalable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.dependencies import Settings, get_model_bundle
from api.main import app
from senegal_rental_price.models.predict import ModelBundle


@pytest.fixture
def client(model_bundle: ModelBundle) -> Iterator[TestClient]:
    """Client de test avec un modèle factice injecté."""
    app.dependency_overrides[get_model_bundle] = lambda: model_bundle
    with TestClient(app) as test_client:
        test_client.app.state.model_bundle = model_bundle  # type: ignore[attr-defined]
        yield test_client
    app.dependency_overrides.clear()


class TestHealth:
    """``GET /health``."""

    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_reports_ok_status(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_reports_model_loaded(self, client: TestClient) -> None:
        assert client.get("/health").json()["model_loaded"] is True


class TestModelInfo:
    """``GET /model/info``."""

    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/model/info").status_code == 200

    def test_exposes_model_identity(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert body["model_name"] == "fake_model"
        assert body["model_version"] == "0.0.1-test"

    def test_exposes_training_metrics(self, client: TestClient) -> None:
        metrics = client.get("/model/info").json()["metrics"]
        assert set(metrics) == {"rmse", "mae", "r2"}

    def test_exposes_feature_names(self, client: TestClient) -> None:
        assert len(client.get("/model/info").json()["feature_names"]) > 0


class TestPredictSuccess:
    """``POST /predict`` — cas nominal."""

    def test_returns_200(self, client: TestClient, valid_payload: dict[str, Any]) -> None:
        assert client.post("/predict", json=valid_payload).status_code == 200

    def test_returns_the_estimated_price(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        body = client.post("/predict", json=valid_payload).json()
        assert body["prix_loyer_mensuel_estime"] == 750_000.0

    def test_returns_currency_and_version(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        body = client.post("/predict", json=valid_payload).json()
        assert body["devise"] == "FCFA"
        assert body["model_version"] == "0.0.1-test"

    def test_accepts_payload_without_optional_quartier(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        payload = {k: v for k, v in valid_payload.items() if k != "quartier"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_accepts_empty_equipements(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        assert client.post("/predict", json={**valid_payload, "equipements": []}).status_code == 200


class TestPredictValidation:
    """``POST /predict`` — la validation Pydantic doit renvoyer 422."""

    @pytest.mark.parametrize(
        ("case", "override"),
        [
            ("surface négative", {"surface_m2": -10}),
            ("surface nulle", {"surface_m2": 0}),
            ("surface hors plage", {"surface_m2": 5000}),
            ("ville inconnue", {"ville": "Kaolack"}),
            ("type de bien inconnu", {"type_bien": "chateau"}),
            ("nb_pieces nul", {"nb_pieces": 0}),
            ("nb_pieces hors plage", {"nb_pieces": 99}),
            ("nb_chambres négatif", {"nb_chambres": -1}),
            ("équipement inconnu", {"equipements": ["heliport"]}),
            ("surface non numérique", {"surface_m2": "grande"}),
            ("champ inconnu", {"champ_pirate": "x"}),
        ],
    )
    def test_invalid_input_returns_422(
        self,
        client: TestClient,
        valid_payload: dict[str, Any],
        case: str,
        override: dict[str, Any],
    ) -> None:
        response = client.post("/predict", json={**valid_payload, **override})
        assert response.status_code == 422, case

    def test_missing_required_field_returns_422(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        payload = {k: v for k, v in valid_payload.items() if k != "surface_m2"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_more_bedrooms_than_rooms_returns_422(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        payload = {**valid_payload, "nb_pieces": 2, "nb_chambres": 5}
        assert client.post("/predict", json=payload).status_code == 422

    def test_error_body_is_explicit_and_serialisable(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        response = client.post("/predict", json={**valid_payload, "surface_m2": -1})
        body = response.json()
        assert body["detail"] == "Payload invalide."
        assert body["errors"][0]["champ"].endswith("surface_m2")
        assert body["errors"][0]["message"]

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        assert client.post("/predict", json={}).status_code == 422


class TestServiceWithoutModel:
    """Comportement dégradé lorsqu'aucun modèle n'est chargé."""

    def test_predict_returns_503(self, valid_payload: dict[str, Any]) -> None:
        app.dependency_overrides.clear()
        with TestClient(app) as client:
            client.app.state.model_bundle = None  # type: ignore[attr-defined]
            response = client.post("/predict", json=valid_payload)
        assert response.status_code == 503

    def test_health_still_reports_unloaded_model(self) -> None:
        app.dependency_overrides.clear()
        with TestClient(app) as client:
            client.app.state.model_bundle = None  # type: ignore[attr-defined]
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False


class TestDocumentation:
    """Documentation interactive générée par FastAPI."""

    def test_docs_are_available(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_openapi_exposes_all_routes(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/model/info", "/predict"} <= set(paths)

    def test_fields_are_documented(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()["components"]["schemas"]["RentalFeatures"]
        assert "description" in schema["properties"]["surface_m2"]


class TestSettings:
    """Configuration injectée par variables d'environnement."""

    def test_defaults_to_local_model_directory(self) -> None:
        assert Settings().model_dir == "models/"

    def test_reads_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_MODEL_DIR", "/tmp/models")
        assert Settings().model_dir == "/tmp/models"
