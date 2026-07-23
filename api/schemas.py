"""Schémas Pydantic de validation des entrées/sorties de l'API.

La validation est **stricte** : toute contrainte métier est déclarée ici plutôt
que vérifiée à la main dans les endpoints. Une entrée invalide est donc rejetée
par FastAPI avec une erreur ``422`` explicite, sans jamais atteindre le modèle.

Les descriptions et exemples renseignés ci-dessous alimentent directement la
documentation interactive ``/docs``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Ville(str, Enum):
    """Villes couvertes par le modèle (restreintes aux données d'entraînement)."""

    dakar = "Dakar"
    thies = "Thiès"


class TypeBien(str, Enum):
    """Types de biens couverts par le modèle."""

    appartement = "appartement"
    maison = "maison"


class Equipement(str, Enum):
    """Équipements reconnus. Tout autre équipement est rejeté (422)."""

    piscine = "piscine"
    climatisation = "climatisation"
    gardiennage = "gardiennage"
    parking = "parking"
    jardin = "jardin"
    terrasse = "terrasse"
    salle_de_sport = "salle_de_sport"
    wifi = "wifi"


class RentalFeatures(BaseModel):
    """Caractéristiques d'un bien soumises à ``POST /predict``."""

    model_config = ConfigDict(
        extra="forbid",  # un champ inconnu est une erreur, pas un silence
        json_schema_extra={
            "example": {
                "ville": "Dakar",
                "quartier": "Almadies",
                "type_bien": "appartement",
                "surface_m2": 70.0,
                "nb_pieces": 3,
                "nb_chambres": 2,
                "meuble": True,
                "equipements": ["climatisation", "parking", "gardiennage"],
            }
        },
    )

    ville: Ville = Field(description="Ville du bien (liste fermée).")
    quartier: str | None = Field(
        default=None,
        max_length=100,
        description="Quartier du bien. Certains quartiers prisés (Almadies, Point E, "
        "Saly...) augmentent l'estimation.",
    )
    type_bien: TypeBien = Field(description="Type de bien.")
    surface_m2: float = Field(
        gt=0,
        le=2000,
        description="Surface habitable en m² (strictement positive, max 2000).",
    )
    nb_pieces: int = Field(ge=1, le=20, description="Nombre total de pièces (1 à 20).")
    nb_chambres: int = Field(ge=0, le=15, description="Nombre de chambres (0 à 15).")
    meuble: bool = Field(description="Le bien est-il meublé ?")
    equipements: list[Equipement] = Field(
        default_factory=list,
        description="Équipements disponibles (valeurs de la liste fermée, sans doublon).",
    )

    @field_validator("equipements")
    @classmethod
    def _dedupe_equipements(cls, value: list[Equipement]) -> list[Equipement]:
        """Supprime les doublons éventuels tout en conservant l'ordre."""
        return list(dict.fromkeys(value))

    @field_validator("quartier")
    @classmethod
    def _strip_quartier(cls, value: str | None) -> str | None:
        """Normalise le quartier (espaces superflus, chaîne vide -> ``None``)."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def check_coherence(self) -> RentalFeatures:
        """Vérifie la cohérence croisée des champs.

        Returns:
            L'instance validée.

        Raises:
            ValueError: Si le nombre de chambres dépasse le nombre de pièces.
                Pydantic convertit cette erreur en réponse ``422``.
        """
        if self.nb_chambres > self.nb_pieces:
            raise ValueError(
                "nb_chambres ne peut pas dépasser nb_pieces "
                f"({self.nb_chambres} > {self.nb_pieces})."
            )
        return self

    def to_listing(self) -> dict[str, object]:
        """Convertit le payload en dictionnaire attendu par la couche modèle."""
        return {
            "ville": self.ville.value,
            "quartier": self.quartier or "",
            "type_bien": self.type_bien.value,
            "surface_m2": self.surface_m2,
            "nb_pieces": self.nb_pieces,
            "nb_chambres": self.nb_chambres,
            "meuble": self.meuble,
            "equipements": [e.value for e in self.equipements],
            "surface_estimee": False,
        }


class PredictionResponse(BaseModel):
    """Réponse de ``POST /predict``."""

    model_config = ConfigDict(
        protected_namespaces=(),  # autorise le champ `model_version`
        json_schema_extra={
            "example": {
                "prix_loyer_mensuel_estime": 1099750.0,
                "devise": "FCFA",
                "model_version": "0.1.0+20260722230541",
            }
        },
    )

    prix_loyer_mensuel_estime: float = Field(
        description="Loyer mensuel estimé, en FCFA.",
    )
    devise: str = Field(default="FCFA", description="Devise de l'estimation.")
    model_version: str = Field(description="Version du modèle ayant produit l'estimation.")


class HealthResponse(BaseModel):
    """Réponse de ``GET /health``."""

    status: str = Field(description="``ok`` si le service répond.")
    model_loaded: bool = Field(description="Un modèle est-il chargé en mémoire ?")

    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    """Réponse de ``GET /model/info`` — métadonnées du modèle chargé."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(description="Nom de l'algorithme (ex. ``random_forest``).")
    model_version: str = Field(description="Version du modèle déployé.")
    trained_at: str = Field(description="Date d'entraînement (ISO 8601, UTC).")
    metrics: dict[str, float] = Field(description="Métriques sur le jeu de test (RMSE, MAE, R²).")
    n_train: int = Field(description="Nombre d'exemples d'entraînement.")
    feature_names: list[str] = Field(description="Features attendues par le modèle.")


class ErrorResponse(BaseModel):
    """Corps d'erreur renvoyé par l'API (documenté dans ``/docs``)."""

    detail: str = Field(description="Description lisible de l'erreur.")
