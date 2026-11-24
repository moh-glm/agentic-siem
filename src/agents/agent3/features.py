"""
Construction des features de l'Agent 3 à partir des sorties des agents 1 et 2.

Contrats utilisés :
- Agent 1 (modèle sélectionné : Gaussian Naive Bayes ou Logistic Regression) :
  prediction["label"] ∈ {"normal", "attack"} et confidence = probabilité de la classe prédite.
  La probabilité d'attaque vaut donc confidence si label == "attack", sinon 1 - confidence.
- Agent 2 (LSTM) : prediction["attack_probability"] = probabilité d'attaque de la fenêtre.
"""

from src.agents.agent3.policy import check_probability

FEATURE_NAMES = ["p_agent1", "p_agent2"]
AGENT1_LABELS = ("normal", "attack")


def _prediction_from(result, expected_agent: str, field_name: str) -> dict:
    """Vérifie l'origine d'une sortie d'agent et retourne son champ prediction."""
    if not isinstance(result, dict):
        raise ValueError(f"{field_name} doit être un dictionnaire")
    if result.get("agent") != expected_agent:
        raise ValueError(
            f"{field_name} doit venir de '{expected_agent}', reçu '{result.get('agent')}'"
        )
    prediction = result.get("prediction")
    if not isinstance(prediction, dict):
        raise ValueError(f"{field_name} : 'prediction' est absent ou invalide")
    return prediction


def attack_probability_from_agent1(result: dict) -> float:
    """Probabilité d'attaque de l'agent 1, dérivée de son label et de sa confiance."""
    prediction = _prediction_from(result, "agent_1", "agent1_result")
    label = prediction.get("label")
    if label not in AGENT1_LABELS:
        raise ValueError(f"agent1_result : label inconnu {label!r}")
    confidence = check_probability("agent1_result.confidence", result.get("confidence"))
    return confidence if label == "attack" else 1.0 - confidence


def attack_probability_from_agent2(result: dict) -> float:
    """Probabilité d'attaque de l'agent 2, lue directement dans sa sortie."""
    prediction = _prediction_from(result, "agent_2", "agent2_result")
    if "attack_probability" not in prediction:
        raise ValueError("agent2_result : 'attack_probability' est absent de 'prediction'")
    return check_probability(
        "agent2_result.attack_probability", prediction["attack_probability"]
    )


def build_feature_row(agent1_result: dict, agent2_result: dict) -> dict:
    """Retourne les deux features de l'arbre pour un même identifiant d'entrée."""
    return {
        "p_agent1": attack_probability_from_agent1(agent1_result),
        "p_agent2": attack_probability_from_agent2(agent2_result),
    }


def decision_confidence(p_agent1: float, p_agent2: float) -> float:
    """
    Confiance de la décision : la plus faible des certitudes des deux agents.
    Chaque certitude vaut max(p, 1 - p), donc la valeur est dans [0.5, 1].
    """
    return min(max(p_agent1, 1.0 - p_agent1), max(p_agent2, 1.0 - p_agent2))
