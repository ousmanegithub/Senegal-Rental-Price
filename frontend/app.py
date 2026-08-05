from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_S = 10

VILLES = ["Dakar", "Thiès"]
TYPES_BIEN = [("appartement", "Appartement"), ("maison", "Maison")]
EQUIPEMENTS = [
    ("piscine", "Piscine"),
    ("climatisation", "Climatisation"),
    ("gardiennage", "Gardiennage"),
    ("parking", "Parking"),
    ("jardin", "Jardin"),
    ("terrasse", "Terrasse"),
    ("salle_de_sport", "Salle de sport"),
    ("wifi", "Wifi"),
]
EQUIPEMENT_LABELS = dict(EQUIPEMENTS)
TYPE_BIEN_LABELS = dict(TYPES_BIEN)

# Quartiers observés dans data/raw/locations.csv pour chaque ville (cf.
# notebooks/01_exploration.ipynb). "Autre" laisse la main si le quartier n'est
# pas répertorié : l'API ne restreint pas ce champ à une liste fermée.
AUTRE_QUARTIER = "Autre (préciser)"
QUARTIERS_PAR_VILLE: dict[str, list[str]] = {
    "Dakar": [
        "Almadies",
        "Cité Keur Gorgui",
        "Fann",
        "Grand Dakar",
        "Lac Rose",
        "Mamelles",
        "Mermoz",
        "Ngor",
        "Ouakam",
        "Plateau",
        "Point E",
        "Sacré-Cœur",
        "Toubab Dialaw",
        "Yoff",
    ],
    "Thiès": [
        "Guéréo",
        "Ngaparou",
        "Nguerine",
        "Nianing",
        "Saly",
        "Somone",
    ],
}

st.set_page_config(
    page_title="Estimation locative - Sénégal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_STYLE = """
<style>
:root {
    --bg: #F5F6F8;
    --surface: #FFFFFF;
    --border: #E3E6EB;
    --text: #101828;
    --text-muted: #667085;
    --primary: #16305C;
    --primary-hover: #0E2144;
    --accent-soft: #EEF2F7;
    --danger: #B42318;
    --danger-soft: #FDEDEC;
    --success-dot: #1E7F4F;
    --neutral-dot: #98A2B3;
    --danger-dot: #D92D20;
}

html, body, [class^="css"], [class*=" css"] {
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
    visibility: hidden;
    height: 0;
}

/* Bandeau natif Streamlit (barre vide au-dessus du contenu) : on le retire
   entierement, notre propre bandeau de marque (.app-header) le remplace. */
header[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 2rem;
    max-width: 1080px;
}

/* ---- Header ---- */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
}
.app-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.01em;
}
.app-header p {
    font-size: 0.92rem;
    color: var(--text-muted);
    margin: 0.25rem 0 0 0;
}
.status-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    white-space: nowrap;
    padding-top: 0.3rem;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    flex-shrink: 0;
}

/* ---- Section labels ---- */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.4rem 0 0.5rem 0;
}
.section-label:first-child {
    margin-top: 0;
}

/* ---- Cards ---- */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
}

/* ---- Result panel ---- */
.result-placeholder {
    color: var(--text-muted);
    font-size: 0.9rem;
    text-align: center;
    padding: 2.5rem 1rem;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}
.result-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.result-unit {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-left: 0.35rem;
}
.result-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--border);
}
.error-card {
    background: var(--danger-soft);
    border: 1px solid #F3B9B4;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    color: var(--danger);
    font-size: 0.88rem;
}
.error-card ul {
    margin: 0.4rem 0 0 1.1rem;
    padding: 0;
}

/* ---- Form widgets ---- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    border-radius: 8px !important;
    border-color: var(--border) !important;
}
.stTextInput input, .stNumberInput input {
    border-radius: 8px;
}
div[data-testid="stFormSubmitButton"] button {
    background: var(--primary);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 0;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--primary-hover);
    color: #FFFFFF;
}

.app-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--text-muted);
}
</style>
"""

st.markdown(_STYLE, unsafe_allow_html=True)


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_health(api_url: str) -> dict[str, Any] | None:
    """Interroge `/health`, mis en cache 5 s pour éviter un appel à chaque frappe."""
    try:
        response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return dict(response.json())
    except requests.RequestException:
        return None


def _render_header() -> None:
    health = _fetch_health(API_URL)
    if health is None:
        dot_color, label = "var(--danger-dot)", "Service indisponible"
    elif not health.get("model_loaded"):
        dot_color, label = "var(--neutral-dot)", "Modèle non chargé"
    else:
        dot_color, label = "var(--success-dot)", "Service disponible"

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>Estimation locative</h1>
                <p>Prédiction du loyer mensuel - Dakar &amp; Thiès</p>
            </div>
            <div class="status-badge">
                <span class="status-dot" style="background:{dot_color};"></span>
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_panel() -> None:
    """Affiche le panneau d'estimation dans une seule carte.

    La carte (ouverture, contenu, fermeture) est construite dans un seul
    appel à ``st.markdown`` : chaque appel produit un bloc HTML indépendant.
    """
    result = st.session_state.get("result")
    errors = st.session_state.get("errors")

    if errors:
        items = "".join(f"<li>{e}</li>" for e in errors)
        inner = (
            '<div class="error-card"><strong>Impossible de calculer une '
            f"estimation.</strong><ul>{items}</ul></div>"
        )
    elif result:
        price = f"{result['prix_loyer_mensuel_estime']:,.0f}".replace(",", " ")
        inner = f"""
            <div class="result-label">Estimation</div>
            <div class="result-value">{price}<span class="result-unit">
                {result["devise"]} / mois</span></div>
            <div class="result-meta">Modèle&nbsp;: {result["model_version"]}</div>
        """
    else:
        inner = (
            '<div class="result-placeholder">Complétez le formulaire pour '
            "obtenir une estimation.</div>"
        )

    st.markdown(f'<div class="card">{inner}</div>', unsafe_allow_html=True)


def _submit(payload: dict[str, Any]) -> None:
    st.session_state["result"] = None
    st.session_state["errors"] = None
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        st.session_state["errors"] = [f"Connexion à l'API impossible ({exc})."]
        return

    if response.status_code == 200:
        st.session_state["result"] = response.json()
    elif response.status_code == 422:
        details = response.json().get("errors", [])
        st.session_state["errors"] = [
            f"{e.get('champ')} : {e.get('message')}" for e in details
        ] or ["Caractéristiques invalides."]
    elif response.status_code == 503:
        st.session_state["errors"] = ["Aucun modèle n'est chargé côté service pour le moment."]
    else:
        st.session_state["errors"] = [
            f"Erreur inattendue du service (HTTP {response.status_code})."
        ]


_render_header()

form_col, result_col = st.columns([3, 2], gap="large")

with form_col:
    # Hors du formulaire : un changement de ville doit immédiatement rafraîchir
    # la liste des quartiers proposés (les widgets d'un st.form ne déclenchent
    # un rerun qu'à la soumission, ce qui empêcherait cet enchaînement).
    st.markdown('<div class="section-label">Localisation</div>', unsafe_allow_html=True)
    loc_col1, loc_col2 = st.columns(2)
    ville = loc_col1.selectbox("Ville", VILLES, key="ville")

    quartier_options = [*QUARTIERS_PAR_VILLE.get(ville, []), AUTRE_QUARTIER]
    quartier_choice = loc_col2.selectbox("Quartier", quartier_options, key=f"quartier_{ville}")
    if quartier_choice == AUTRE_QUARTIER:
        quartier = loc_col2.text_input(
            "Préciser le quartier",
            key=f"quartier_autre_{ville}",
            placeholder="Nom du quartier",
        )
    else:
        quartier = quartier_choice

    with st.form("prediction_form"):
        st.markdown('<div class="section-label">Caractéristiques</div>', unsafe_allow_html=True)
        char_col1, char_col2 = st.columns(2)
        with char_col1:
            type_bien = st.selectbox(
                "Type de bien",
                [v for v, _ in TYPES_BIEN],
                format_func=lambda v: TYPE_BIEN_LABELS[v],
            )
            surface_m2 = st.number_input(
                "Surface (m²)", min_value=1.0, max_value=2000.0, value=70.0, step=1.0
            )
        with char_col2:
            nb_pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20, value=3)
            nb_chambres = st.number_input("Nombre de chambres", min_value=0, max_value=15, value=2)
        meuble = st.toggle("Meublé", value=False)

        st.markdown('<div class="section-label">Équipements</div>', unsafe_allow_html=True)
        equipements = st.multiselect(
            "Équipements disponibles",
            options=[v for v, _ in EQUIPEMENTS],
            format_func=lambda v: EQUIPEMENT_LABELS[v],
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Estimer le loyer", use_container_width=True)

    if submitted:
        if nb_chambres > nb_pieces:
            st.session_state["result"] = None
            st.session_state["errors"] = [
                "Le nombre de chambres ne peut pas dépasser le nombre de pièces."
            ]
        else:
            _submit(
                {
                    "ville": ville,
                    "quartier": quartier or None,
                    "type_bien": type_bien,
                    "surface_m2": surface_m2,
                    "nb_pieces": nb_pieces,
                    "nb_chambres": nb_chambres,
                    "meuble": meuble,
                    "equipements": equipements,
                }
            )
        st.rerun()

with result_col:
    _render_result_panel()

st.markdown(
    '<div class="app-footer">Estimation calculée par un modèle statistique '
    "entraîné sur des annonces réelles. Fournie à titre indicatif, sans valeur "
    "contractuelle.</div>",
    unsafe_allow_html=True,
)
