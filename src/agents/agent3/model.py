"""
Entraînement, sauvegarde et explication de l'arbre de décision de l'Agent 3.
"""

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.agents.agent3.features import FEATURE_NAMES
from src.agents.agent3.policy import ACTIONS

RANDOM_STATE = 42
DEFAULT_MAX_DEPTH = 4
TREE_LEAF = -1


def build_decision_tree(max_depth: int = DEFAULT_MAX_DEPTH) -> DecisionTreeClassifier:
    if max_depth < 1:
        raise ValueError("max_depth doit être au moins 1")
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train_decision_tree(features: pd.DataFrame, targets, max_depth: int = DEFAULT_MAX_DEPTH):
    """Entraîne l'arbre sur les features p_agent1 et p_agent2, avec les actions cibles."""
    missing = [name for name in FEATURE_NAMES if name not in features.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    unknown = set(targets) - set(ACTIONS)
    if unknown:
        raise ValueError(f"Actions inconnues dans les cibles : {sorted(unknown)}")

    model = build_decision_tree(max_depth)
    model.fit(features[FEATURE_NAMES], targets)
    return model


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path) -> DecisionTreeClassifier:
    """Charge un arbre sauvegardé et vérifie qu'il correspond au contrat de l'Agent 3."""
    model = joblib.load(path)
    if not isinstance(model, DecisionTreeClassifier):
        raise ValueError("Le fichier ne contient pas un DecisionTreeClassifier")
    if model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            f"Nombre de features attendu : {len(FEATURE_NAMES)}, trouvé : {model.n_features_in_}"
        )
    unknown = set(model.classes_) - set(ACTIONS)
    if unknown:
        raise ValueError(f"Classes inconnues dans le modèle : {sorted(unknown)}")
    return model


def decision_rules(model, features: pd.DataFrame) -> list:
    """
    Retourne les conditions rencontrées sur le chemin de décision d'une ligne.
    Exemple : ["p_agent2 > 0.900", "p_agent1 <= 0.500"].
    """
    tree = model.tree_
    rules = []
    for node in model.decision_path(features).indices:
        if tree.children_left[node] == TREE_LEAF:
            continue
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        name = FEATURE_NAMES[feature]
        value = features.iloc[0, feature]
        operator = "<=" if value <= threshold else ">"
        rules.append(f"{name} {operator} {threshold:.3f}")
    return rules


def simplified_rules(model, features: pd.DataFrame) -> list:
    """
    Règles du chemin de décision, avec les conditions répétées fusionnées.
    Exemple : ["p_agent1 in (0.500, 0.900]"] au lieu de deux conditions séparées.
    """
    tree = model.tree_
    x = features.iloc[0]
    bounds = {name: [float("-inf"), float("inf")] for name in FEATURE_NAMES}

    for node in model.decision_path(features).indices:
        if tree.children_left[node] == TREE_LEAF:
            continue
        feature = tree.feature[node]
        threshold = float(tree.threshold[node])
        name = FEATURE_NAMES[feature]
        if x.iloc[feature] <= threshold:
            bounds[name][1] = min(bounds[name][1], threshold)
        else:
            bounds[name][0] = max(bounds[name][0], threshold)

    rules = []
    for name in FEATURE_NAMES:
        low, high = bounds[name]
        if low == float("-inf") and high == float("inf"):
            continue
        if low == float("-inf"):
            rules.append(f"{name} <= {high:.3f}")
        elif high == float("inf"):
            rules.append(f"{name} > {low:.3f}")
        else:
            rules.append(f"{low:.3f} < {name} <= {high:.3f}")
    return rules


def export_rules_text(model) -> str:
    """Affiche l'arbre complet sous forme de règles textuelles (pour le rapport)."""
    return export_text(model, feature_names=FEATURE_NAMES)
