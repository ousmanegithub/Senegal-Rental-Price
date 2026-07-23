"""Tests du nettoyage et du prétraitement des données."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from senegal_rental_price.data.preprocessing import (
    CleaningConfig,
    clean,
    filter_listings,
    impute_missing,
    load_raw,
    reconcile_meuble,
    split_equipements,
)


class TestSplitEquipements:
    """Découpage de la colonne ``equipements``."""

    def test_splits_on_pipe(self) -> None:
        assert split_equipements("piscine|parking") == ["piscine", "parking"]

    def test_normalises_case_and_spaces(self) -> None:
        assert split_equipements(" Piscine | PARKING ") == ["piscine", "parking"]

    def test_removes_duplicates(self) -> None:
        assert split_equipements("piscine|piscine|jardin") == ["piscine", "jardin"]

    @pytest.mark.parametrize("missing", [None, np.nan, ""])
    def test_missing_returns_empty_list(self, missing: object) -> None:
        assert split_equipements(missing) == []


class TestReconcileMeuble:
    """Réconciliation du booléen ``meuble`` avec les équipements."""

    def test_true_when_only_equipements_mentions_it(self) -> None:
        df = pd.DataFrame([{"meuble": False, "equipements": "piscine|meuble"}])
        assert bool(reconcile_meuble(df).iloc[0]) is True

    def test_true_when_only_column_is_true(self) -> None:
        df = pd.DataFrame([{"meuble": True, "equipements": "piscine"}])
        assert bool(reconcile_meuble(df).iloc[0]) is True

    def test_false_when_neither_source_mentions_it(self) -> None:
        df = pd.DataFrame([{"meuble": False, "equipements": "piscine"}])
        assert bool(reconcile_meuble(df).iloc[0]) is False

    def test_handles_missing_values(self) -> None:
        df = pd.DataFrame([{"meuble": np.nan, "equipements": None}])
        assert bool(reconcile_meuble(df).iloc[0]) is False

    def test_parses_string_booleans(self) -> None:
        df = pd.DataFrame([{"meuble": "True", "equipements": None}])
        assert bool(reconcile_meuble(df).iloc[0]) is True


class TestFilterListings:
    """Filtrage des annonces non résidentielles et aberrantes."""

    def test_removes_commercial_listings(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = filter_listings(raw_df, cleaning_config)
        assert "b1" not in set(result["id"])

    def test_removes_price_outliers(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = filter_listings(raw_df, cleaning_config)
        assert "b2" not in set(result["id"])

    def test_removes_unknown_cities(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = filter_listings(raw_df, cleaning_config)
        assert "b3" not in set(result["id"])

    def test_keeps_valid_listings(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = filter_listings(raw_df, cleaning_config)
        assert set(result["id"]) == {"a1", "a2", "a3"}

    def test_removes_surface_outliers(self, cleaning_config: CleaningConfig) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "x",
                    "ville": "Dakar",
                    "type_bien": "maison",
                    "surface_m2": 99_999.0,
                    "prix_loyer_mensuel": 500_000,
                    "titre": "Grande maison",
                }
            ]
        )
        assert len(filter_listings(df, cleaning_config)) == 0


class TestImputeMissing:
    """Imputation des valeurs manquantes."""

    def test_no_missing_values_remain(self, raw_df: pd.DataFrame) -> None:
        result = impute_missing(raw_df)
        assert result["nb_pieces"].isna().sum() == 0
        assert result["nb_chambres"].isna().sum() == 0

    def test_imputed_columns_are_integers(self, raw_df: pd.DataFrame) -> None:
        result = impute_missing(raw_df)
        assert result["nb_pieces"].dtype.kind == "i"

    def test_uses_median_of_same_property_type(self) -> None:
        df = pd.DataFrame(
            [
                {"type_bien": "maison", "nb_pieces": 4.0, "nb_chambres": 3.0},
                {"type_bien": "maison", "nb_pieces": 6.0, "nb_chambres": 5.0},
                {"type_bien": "maison", "nb_pieces": np.nan, "nb_chambres": np.nan},
            ]
        )
        result = impute_missing(df)
        assert result.loc[2, "nb_pieces"] == 5


class TestClean:
    """Pipeline de nettoyage complet."""

    def test_returns_only_valid_rows(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        assert len(clean(raw_df, cleaning_config)) == 3

    def test_drops_informational_columns(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = clean(raw_df, cleaning_config)
        for column in ("id", "adresse", "date_publication"):
            assert column not in result.columns

    def test_equipements_become_lists(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = clean(raw_df, cleaning_config)
        assert all(isinstance(value, list) for value in result["equipements"])

    def test_meuble_is_boolean(self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig) -> None:
        result = clean(raw_df, cleaning_config)
        assert result["meuble"].dtype == bool

    def test_index_is_reset(self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig) -> None:
        result = clean(raw_df, cleaning_config)
        assert list(result.index) == list(range(len(result)))

    def test_target_column_is_preserved(
        self, raw_df: pd.DataFrame, cleaning_config: CleaningConfig
    ) -> None:
        result = clean(raw_df, cleaning_config)
        assert cleaning_config.target in result.columns


class TestLoadRaw:
    """Chargement du CSV brut."""

    def test_reads_semicolon_separated_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.csv"
        path.write_text("ville;prix_loyer_mensuel\nDakar;900000\n", encoding="utf-8-sig")
        result = load_raw(path)
        assert list(result.columns) == ["ville", "prix_loyer_mensuel"]
        assert len(result) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_raw(tmp_path / "absent.csv")
