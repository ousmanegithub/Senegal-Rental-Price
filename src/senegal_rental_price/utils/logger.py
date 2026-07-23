"""Configuration centralisée du logging pour tout le projet.

Aucun ``print()`` ne doit être utilisé dans le code de production (cf. §3.3 du
sujet). On expose :

* :func:`configure_logging` — à appeler **une seule fois** au démarrage d'un
  point d'entrée (entraînement, API...) pour fixer le niveau et le format.
* :func:`get_logger` — à utiliser dans chaque module pour récupérer un logger
  nommé, sans reconfigurer le handler racine.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: bool = False


def _ensure_utf8_streams() -> None:
    """Force stdout/stderr en UTF-8 si la console ne l'est pas déjà.

    Sur Windows, la console utilise par défaut un encodage historique (ex.
    cp1252) qui ne sait pas encoder les emojis que MLflow >=3 imprime en fin
    de run (ex. "🏃 View run..."). Sans ce correctif, l'exception lève hors
    de ``with mlflow.start_run()`` et le run reste bloqué au statut RUNNING
    côté serveur, alors que le tracking a pourtant réussi. No-op sur
    Linux/macOS où l'encodage est déjà UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and getattr(stream, "encoding", "").lower() != "utf-8":
            reconfigure(encoding="utf-8", errors="replace")


def configure_logging(level: str | int | None = None) -> None:
    """Configure le logger racine une seule fois.

    Args:
        level: Niveau de log (``"INFO"``, ``"DEBUG"``, ``logging.WARNING``...).
            Si ``None``, la variable d'environnement ``LOG_LEVEL`` est utilisée,
            avec repli sur ``"INFO"``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    _ensure_utf8_streams()
    resolved = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved)
    # Évite les handlers dupliqués si la fonction est appelée plusieurs fois
    # indirectement (ex. rechargement uvicorn).
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé, en garantissant une configuration de base.

    Args:
        name: Nom du logger, typiquement ``__name__`` du module appelant.

    Returns:
        Une instance :class:`logging.Logger` prête à l'emploi.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
