# Prédiction du prix des locations au Sénégal 🇸🇳

<!-- Remplacez UTILISATEUR/DEPOT par le chemin réel de votre dépôt GitHub -->
[![CI](https://github.com/UTILISATEUR/DEPOT/actions/workflows/ci.yml/badge.svg)](https://github.com/UTILISATEUR/DEPOT/actions/workflows/ci.yml)

Service de prédiction du **loyer mensuel** d'un bien immobilier au Sénégal
(Dakar, région de Thiès / Petite-Côte) à partir de ses caractéristiques :
localisation, surface, nombre de pièces, type de bien, équipements.

Projet **M2 DSIA** — l'accent est mis sur la qualité d'ingénierie logicielle et
la chaîne MLOps complète : typage strict, tests, configuration Hydra, tracking
MLflow, API typée, conteneurisation et intégration continue.

---

## 1. Prérequis

| Outil | Version | Vérification |
|---|---|---|
| Python | ≥ 3.11 | `python --version` |
| pip | récent | `pip --version` |
| Git | — | `git --version` |
| Docker (facultatif à ce stade) | ≥ 24 | `docker --version` |

## 2. Installation

```bash
# 1. Récupérer le projet
git clone <URL_DU_DEPOT>
cd senegal-rental-price

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell

# 3. Installer le package et ses dépendances de développement
pip install --upgrade pip
pip install -e ".[dev]"
```

> `-e` (mode éditable) est recommandé en développement : vos modifications du
> code sont prises en compte sans réinstaller. Pour une installation classique,
> `pip install .` suffit.

Vérification :

```bash
python -c "import senegal_rental_price as p; print(p.__version__)"   # -> 0.1.0
```

### Données

Placez le fichier d'annonces collectées par scraping dans :

```
data/raw/locations.csv
```

Format attendu : CSV, séparateur `;`, encodage UTF-8. Voir `data/README.md` pour
le schéma détaillé, la provenance et les traitements de nettoyage appliqués.

## 3. Entraîner un modèle

Toute la configuration passe par **Hydra** (`conf/`) : aucun hyperparamètre ni
chemin n'est codé en dur.

```bash
# Modèle par défaut (random_forest)
python -m senegal_rental_price.models.train

# ... ou via le point d'entrée CLI installé avec le package
senegal-rental-train

# Changer de modèle sans toucher au code
python -m senegal_rental_price.models.train model=ridge
python -m senegal_rental_price.models.train model=xgboost

# Surcharger un hyperparamètre à la volée
python -m senegal_rental_price.models.train model=xgboost model.params.max_depth=8

# Comparer les trois modèles en une commande (multirun Hydra)
python -m senegal_rental_price.models.train -m model=ridge,random_forest,xgboost
```

Chaque exécution :

1. charge et nettoie les données (`data/raw/locations.csv`) ;
2. construit les features ;
3. entraîne le modèle et calcule RMSE / MAE / R² sur le jeu de test ;
4. enregistre un **run MLflow** (paramètres, métriques, modèle) ;
5. sérialise le modèle dans `models/model.joblib` (+ `models/model_meta.json`).

> Le fichier `models/model.joblib` correspond au **dernier** entraînement lancé.
> Workflow conseillé : comparer les modèles via `-m`, puis relancer seul le
> modèle retenu pour figer l'artefact servi par l'API.

### Visualiser la comparaison des modèles (MLflow)

Par défaut (`conf/config.yaml`), l'entraînement pointe vers un **serveur MLflow**
tournant dans Docker (`http://localhost:5000`) — à démarrer une fois :

```bash
docker compose -f docker/docker-compose.yml up -d mlflow
# puis ouvrir http://localhost:5000
```

> Sans Docker, l'entraînement fonctionne quand même (le tracking échoue
> silencieusement avec un avertissement, le modèle est sérialisé normalement).
> Pour un tracking 100 % local sans Docker :
> `python -m senegal_rental_price.models.train mlflow.tracking_uri=sqlite:///mlflow.db`

## 4. Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

| Route | Méthode | Description |
|---|---|---|
| `/health` | GET | État du service et présence d'un modèle chargé |
| `/model/info` | GET | Métadonnées du modèle (version, date, métriques) |
| `/predict` | POST | Estimation du loyer mensuel |
| `/docs` | GET | Documentation interactive Swagger |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ville": "Dakar",
    "quartier": "Almadies",
    "type_bien": "appartement",
    "surface_m2": 70,
    "nb_pieces": 3,
    "nb_chambres": 2,
    "meuble": true,
    "equipements": ["climatisation", "parking", "gardiennage"]
  }'
```

Réponse :

```json
{
  "prix_loyer_mensuel_estime": 1099750.0,
  "devise": "FCFA",
  "model_version": "0.1.0+20260722230541"
}
```

Une entrée invalide (surface négative, ville hors liste, plus de chambres que de
pièces...) renvoie une erreur **422** explicite, jamais une exception non gérée.

### Configuration de l'API (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `API_MODEL_DIR` | `models/` | Répertoire du modèle sérialisé |
| `API_MLFLOW_TRACKING_URI` | — | URI du serveur MLflow |
| `API_MLFLOW_MODEL_URI` | — | Modèle du Registry, ex. `models:/senegal-rental-price-model/Production` |
| `API_LOG_LEVEL` | `INFO` | Niveau de log |

Si `API_MLFLOW_MODEL_URI` est défini, l'API charge le modèle depuis le **Model
Registry MLflow** et retombe automatiquement sur l'artefact local en cas
d'indisponibilité (le service reste démarrable hors ligne).

## 5. Lancer le frontend

Interface Streamlit (formulaire → appel API → prix estimé) :

```bash
pip install -e ".[frontend]"      # streamlit + requests, si pas déjà fait
streamlit run frontend/app.py     # puis ouvrir http://localhost:8501
```

L'URL de l'API est configurable via la variable d'environnement `API_URL`
(défaut : `http://localhost:8000`). La sidebar de l'application indique si
l'API est joignable et si un modèle est chargé.

## 6. Conteneurisation Docker

```bash
# Lance mlflow + api + frontend en une commande
docker compose -f docker/docker-compose.yml up --build
```

| Service | Port hôte | Notes |
|---|---|---|
| `mlflow` | 5000 | Serveur de tracking (backend SQLite + artefacts persistés dans `mlflow-data/`, non versionné) |
| `api` | 8000 | Charge `models/model.joblib` monté en lecture seule depuis l'hôte — **entraînez un modèle avant** (§3) |
| `frontend` | 8501 | Appelle `api` via le réseau Docker interne (`API_URL=http://api:8000`) |

Toutes les images sont construites en **multi-stage** (dépendances dans un
`venv` isolé, jetées à l'exécution) et tournent avec un **utilisateur
non-root** (`appuser`). Les chemins et URLs sont injectés par variables
d'environnement, jamais codés en dur.

```bash
docker compose -f docker/docker-compose.yml down   # arrêt propre
```

## 7. Tests et qualité du code

```bash
# Tests + couverture (seuil imposé : 70 %)
pytest --cov=src --cov-fail-under=70

# Typage statique strict
mypy src/

# Lint et format
ruff check .
black --check .

# Installer les hooks pre-commit (exécutés à chaque commit)
pre-commit install
```

## 8. Structure du projet

```
senegal-rental-price/
├── conf/                     # Configuration Hydra
│   ├── config.yaml
│   ├── model/                # ridge · random_forest · xgboost
│   └── data/default.yaml
├── data/
│   ├── raw/                  # Données brutes (non versionnées)
│   ├── processed/            # Données nettoyées
│   └── README.md             # Provenance et description
├── notebooks/                # Exploration (hors code de production)
├── src/senegal_rental_price/
│   ├── data/preprocessing.py
│   ├── features/build_features.py
│   ├── models/{train,predict}.py
│   └── utils/logger.py
├── api/                      # FastAPI + schémas Pydantic
├── frontend/                 # Interface utilisateur
├── tests/                    # pytest (couverture ≥ 70 %)
├── models/                   # Modèles sérialisés (non versionnés)
├── docker/                   # Dockerfiles + docker-compose
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## 9. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `FileNotFoundError: Aucun modèle trouvé` | Aucun entraînement lancé | `python -m senegal_rental_price.models.train` |
| `/health` renvoie `model_loaded: false` | Idem | Idem, puis relancer l'API |
| `FileNotFoundError` sur `locations.csv` | Données absentes | Placer le CSV dans `data/raw/` |
| `senegal-rental-train: command not found` | Package non installé | `pip install -e ".[dev]"` |
| Erreur d'encodage à la lecture du CSV | Séparateur ou encodage | Vérifier `sep` et `encoding` dans `conf/data/default.yaml` |
| `Tracking MLflow ignoré` au log de l'entraînement | Le serveur `mlflow` (Docker) n'est pas démarré | `docker compose -f docker/docker-compose.yml up -d mlflow`, ou repli local (§3) |
| `error during connect ... dockerDesktopLinuxEngine` | Docker Desktop n'est pas lancé (Windows) | Démarrer Docker Desktop, réessayer une fois l'icône stable |
