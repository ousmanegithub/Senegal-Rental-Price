"""Tests de la génération de features."""

from __future__ import annotations

import pandas as pd

from senegal_rental_price.data.preprocessing import CleaningConfig, clean
from senegal_rental_price.features.build_features import (
    FeatureConfig,
    build_features,
    feature_names,
)


class TestFeatureNames:
    """Contrat de nommage des colonnes."""

    def test_is_deterministic(self, feature_config: FeatureConfig) -> None:
        assert feature_names(feature_config) == feature_names(feature_config)

    def test_has_no_duplicates(self, feature_config: FeatureConfig) -> None:
        names = feature_names(feature_config)
        assert len(names) == len(set(names))

    def test_contains_one_flag_per_equipement(self, feature_config: FeatureConfig) -> None:
        names = feature_names(feature_config)
        for equipement in feature_config.known_equipements:
            assert f"equip_{equipement}" in names

    def test_contains_one_flag_per_city_and_type(self, feature_config: FeatureConfig) -> None:
        names = feature_names(feature_config)
        assert "ville_Dakar" in names
        assert "type_appartement" in names


class TestBuildFeatures:
    """Construction de la matrice de features."""

    def test_columns_match_the_declared_schema(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig, feature_config: FeatureConfig
    ) -> None:
        features = build_features(clean(raw_df, cleaning_config), feature_config)
        assert list(features.columns) == feature_names(feature_config)

    def test_contains_no_missing_values(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig, feature_config: FeatureConfig
    ) -> None:
        features = build_features(clean(raw_df, cleaning_config), feature_config)
        assert not bool(features.isna().to_numpy().any())

    def test_row_count_is_preserved(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig, feature_config: FeatureConfig
    ) -> None:
        data = clean(raw_df, cleaning_config)
        assert len(build_features(data, feature_config)) == len(data)

    def test_equipement_flags_are_binary(self, feature_config: FeatureConfig) -> None:
        df = pd.DataFrame(
            [
                {
                    "ville": "Dakar",
                    "quartier": "Almadies",
                    "type_bien": "appartement",
                    "surface_m2": 70.0,
                    "nb_pieces": 3,
                    "nb_chambres": 2,
                    "meuble": True,
                    "equipements": ["piscine"],
                }
            ]
        )
        features = build_features(df, feature_config)
        assert features.loc[0, "equip_piscine"] == 1
        assert features.loc[0, "equip_wifi"] == 0

    def test_counts_equipements(self, feature_config: FeatureConfig) -> None:
        df = pd.DataFrame(
            [
                {
                    "ville": "Dakar",
                    "type_bien": "maison",
                    "surface_m2": 100.0,
                    "nb_pieces": 4,
                    "nb_chambres": 3,
                    "meuble": False,
                    "equipements": ["piscine", "jardin", "parking"],
                }
            ]
        )
        assert build_features(df, feature_config).loc[0, "nb_equipements"] == 3

    def test_flags_premium_quartier(self, feature_config: FeatureConfig) -> None:
        base = {
            "ville": "Dakar",
            "type_bien": "appartement",
            "surface_m2": 70.0,
            "nb_pieces": 3,
            "nb_chambres": 2,
            "meuble": False,
            "equipements": [],
        }
        df = pd.DataFrame([{**base, "quartier": "Almadies"}, {**base, "quartier": "Pikine"}])
        features = build_features(df, feature_config)
        assert features.loc[0, "quartier_premium"] == 1
        assert features.loc[1, "quartier_premium"] == 0

    def test_one_hot_encodes_city(self, feature_config: FeatureConfig) -> None:
        df = pd.DataFrame(
            [
                {
                    "ville": "Thiès",
                    "quartier": "Saly",
                    "type_bien": "maison",
                    "surface_m2": 120.0,
                    "nb_pieces": 4,
                    "nb_chambres": 3,
                    "meuble": False,
                    "equipements": [],
                }
            ]
        )
        features = build_features(df, feature_config)
        assert features.loc[0, "ville_Thiès"] == 1
        assert features.loc[0, "ville_Dakar"] == 0

    def test_accepts_pipe_separated_string(self, feature_config: FeatureConfig) -> None:
        """Le module accepte aussi bien une liste qu'une chaîne brute."""
        df = pd.DataFrame(
            [
                {
                    "ville": "Dakar",
                    "quartier": "Almadies",
                    "type_bien": "appartement",
                    "surface_m2": 70.0,
                    "nb_pieces": 3,
                    "nb_chambres": 2,
                    "meuble": True,
                    "equipements": "piscine|jardin",
                }
            ]
        )
        assert build_features(df, feature_config).loc[0, "nb_equipements"] == 2

    def test_handles_missing_optional_columns(self, feature_config: FeatureConfig) -> None:
        """``quartier`` et ``surface_estimee`` sont facultatifs (cas de l'API)."""
        df = pd.DataFrame(
            [
                {
                    "ville": "Dakar",
                    "type_bien": "appartement",
                    "surface_m2": 70.0,
                    "nb_pieces": 3,
                    "nb_chambres": 2,
                    "meuble": True,
                    "equipements": [],
                }
            ]
        )
        features = build_features(df, feature_config)
        assert list(features.columns) == feature_names(feature_config)

    def test_single_row_matches_training_schema(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig, feature_config: FeatureConfig
    ) -> None:
        """Garantie clé : même schéma à l'entraînement et à l'inférence unitaire."""
        training = build_features(clean(raw_df, cleaning_config), feature_config)
        single = build_features(
            pd.DataFrame(
                [
                    {
                        "ville": "Dakar",
                        "quartier": "Almadies",
                        "type_bien": "appartement",
                        "surface_m2": 70.0,
                        "nb_pieces": 3,
                        "nb_chambres": 2,
                        "meuble": True,
                        "equipements": ["piscine"],
                    }
                ]
            ),
            feature_config,
        )
        assert list(single.columns) == list(training.columns)
