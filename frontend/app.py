"""Interface Streamlit — formulaire de prédiction du loyer mensuel.

Application volontairement simple : un formulaire, un appel HTTP vers l'API
FastAPI (`POST /predict`), et un affichage clair du résultat ou de l'erreur.
Aucune logique métier ici (pas de nettoyage, pas de features) : tout passe par
l'API, seule source de vérité.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_S = 10

VILLES = ["Dakar", "Thiès"]
TYPES_BIEN = ["appartement", "maison"]
EQUIPEMENTS = [
    "piscine",
    "climatisation",
    "gardiennage",
    "parking",
    "jardin",
    "terrasse",
    "salle_de_sport",
    "wifi",
]

st.set_page_config(page_title="Prix des locations — Sénégal", page_icon="🏠")
st.title("🏠 Estimation du loyer mensuel — Sénégal")
st.caption(
    "Renseignez les caractéristiques du bien pour obtenir une estimation du "
    "loyer mensuel (FCFA), calculée par l'API de prédiction."
)

with st.sidebar:
    st.subheader("Connexion à l'API")
    st.code(API_URL, language=None)
    try:
        health = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT_S)
        if health.ok and health.json().get("model_loaded"):
            st.success("API en ligne — modèle chargé")
        elif health.ok:
            st.warning("API en ligne — aucun modèle chargé")
        else:
            st.error(f"API injoignable (HTTP {health.status_code})")
    except requests.RequestException:
        st.error("Impossible de contacter l'API. Vérifiez qu'elle est démarrée.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        ville = st.selectbox("Ville", VILLES)
        type_bien = st.selectbox("Type de bien", TYPES_BIEN)
        surface_m2 = st.number_input(
            "Surface (m²)", min_value=1.0, max_value=2000.0, value=70.0, step=1.0
        )
        quartier = st.text_input("Quartier (optionnel)", value="")
    with col2:
        nb_pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20, value=3)
        nb_chambres = st.number_input("Nombre de chambres", min_value=0, max_value=15, value=2)
        meuble = st.checkbox("Meublé", value=False)
        equipements = st.multiselect("Équipements", EQUIPEMENTS)

    submitted = st.form_submit_button("Estimer le loyer", use_container_width=True)

if submitted:
    if nb_chambres > nb_pieces:
        st.error("Le nombre de chambres ne peut pas dépasser le nombre de pièces.")
    else:
        payload: dict[str, Any] = {
            "ville": ville,
            "quartier": quartier or None,
            "type_bien": type_bien,
            "surface_m2": surface_m2,
            "nb_pieces": nb_pieces,
            "nb_chambres": nb_chambres,
            "meuble": meuble,
            "equipements": equipements,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            st.error(f"Impossible de contacter l'API : {exc}")
        else:
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"**Loyer mensuel estimé : "
                    f"{result['prix_loyer_mensuel_estime']:,.0f} {result['devise']}**".replace(
                        ",", " "
                    )
                )
                st.caption(f"Modèle : {result['model_version']}")
            elif response.status_code == 422:
                st.error("Caractéristiques invalides :")
                for err in response.json().get("errors", []):
                    st.write(f"- **{err.get('champ')}** : {err.get('message')}")
            elif response.status_code == 503:
                st.error("Aucun modèle n'est chargé côté API pour le moment.")
            else:
                st.error(f"Erreur inattendue de l'API (HTTP {response.status_code}).")
