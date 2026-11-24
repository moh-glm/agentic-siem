"""
Politique de réponse de l'Agent 3.

Cette politique est une convention documentée (voir docs/agent3_policy.md).
Elle n'utilise que les probabilités d'attaque des agents 1 et 2, jamais les
labels de vérité terrain ni la catégorie d'attaque.
"""

import math

import pandas as pd

ACTIONS = ("MONITOR", "ALERT", "INVESTIGATE", "RECOMMEND_BLOCK")

# Valeurs par défaut. Les seuils finaux doivent être choisis sur le jeu de validation.
DEFAULT_T1 = 0.50
DEFAULT_T2 = 0.50
DEFAULT_HIGH = 0.90


def check_probability(name: str, value) -> float:
    """Retourne la valeur sous forme de float dans [0, 1], sinon lève ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} doit être un nombre, reçu {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} doit être dans [0, 1], reçu {number}")
    return number


def check_threshold(name: str, value) -> float:
    """Retourne un seuil dans (0, 1], sinon lève ValueError."""
    number = check_probability(name, value)
    if number == 0.0:
        raise ValueError(f"{name} doit être strictement supérieur à 0")
    return number


def choose_action(p_agent1, p_agent2, t1=DEFAULT_T1, t2=DEFAULT_T2, high=DEFAULT_HIGH) -> str:
    """
    Applique la politique de réponse.

    - MONITOR : aucun agent ne dépasse son seuil.
    - INVESTIGATE : un seul agent dépasse son seuil (désaccord).
    - ALERT : les deux agents dépassent leur seuil, sans être très sûrs.
    - RECOMMEND_BLOCK : les deux agents dépassent `high`.
    """
    p_agent1 = check_probability("p_agent1", p_agent1)
    p_agent2 = check_probability("p_agent2", p_agent2)
    t1 = check_threshold("t1", t1)
    t2 = check_threshold("t2", t2)
    high = check_threshold("high", high)

    flag_lr = p_agent1 >= t1
    flag_lstm = p_agent2 >= t2

    if not flag_lr and not flag_lstm:
        return "MONITOR"
    if flag_lr and flag_lstm:
        if min(p_agent1, p_agent2) >= high:
            return "RECOMMEND_BLOCK"
        return "ALERT"
    return "INVESTIGATE"


def apply_policy(features: pd.DataFrame, t1=DEFAULT_T1, t2=DEFAULT_T2, high=DEFAULT_HIGH) -> pd.Series:
    """Applique choose_action à chaque ligne d'un DataFrame contenant p_agent1 et p_agent2."""
    if features.empty:
        return pd.Series([], dtype=object)
    return features.apply(
        lambda row: choose_action(row["p_agent1"], row["p_agent2"], t1, t2, high), axis=1
    )
