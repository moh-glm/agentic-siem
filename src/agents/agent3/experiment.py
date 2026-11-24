"""
Évaluation de l'Agent 3 sur des sorties exportées des agents 1 et 2.

Entrée : un tableau par flux, avec les colonnes
    input_id    identifiant du flux (identique pour les deux agents)
    p_agent1    probabilité d'attaque de l'agent 1
    p_agent2    probabilité d'attaque de l'agent 2 pour la fenêtre se terminant à ce flux
    y_true      vérité terrain (0 normal, 1 attaque), utilisée seulement pour l'évaluation
    attack_cat  catégorie d'attaque (optionnelle), utilisée seulement pour l'analyse de sévérité

Protocole :
1. Choisir t1 et t2 sur la validation (seuil maximisant le F1 de chaque agent).
2. Entraîner l'arbre sur la validation, avec les actions de la politique comme cibles.
3. Évaluer sur le test : fidélité à la politique, performance réelle contre la vérité
   terrain, et comparaison avec des baselines.

Usage :
    python -m src.agents.agent3.experiment --validation val.csv --test test.csv --out reports/results/agent3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.agents.agent3.features import FEATURE_NAMES
from src.agents.agent3.model import export_rules_text, train_decision_tree
from src.agents.agent3.policy import (
    ACTIONS,
    DEFAULT_HIGH,
    apply_policy,
    check_probability,
)

REQUIRED_COLUMNS = ["input_id", "p_agent1", "p_agent2", "y_true"]
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.05, 0.96, 0.05), 2))
DEPTHS_TO_COMPARE = (2, 3, 4, 5)
HIGH_GRID = (0.80, 0.85, 0.90, 0.95, 0.99)
MAX_FALSE_BLOCK_RATE = 0.01


def ratio(numerator, denominator):
    """Division sûre : retourne None quand le dénominateur est nul."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def validate_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie le format du tableau d'évaluation et renvoie une copie propre."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df.copy()

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Des valeurs manquantes sont présentes dans les colonnes requises.")

    if df["input_id"].duplicated().any():
        raise ValueError("Les identifiants input_id doivent être uniques.")

    for column in ("p_agent1", "p_agent2"):
        df[column] = [check_probability(f"{column}[{i}]", value) for i, value in enumerate(df[column])]

    if not df["y_true"].isin([0, 1]).all():
        raise ValueError("y_true doit contenir uniquement 0 (normal) ou 1 (attaque).")
    df["y_true"] = df["y_true"].astype(int)

    return df


def load_outputs(path) -> pd.DataFrame:
    return validate_outputs(pd.read_csv(path))


def merge_outputs(agent1_table: pd.DataFrame, agent2_table: pd.DataFrame):
    """
    Joint les sorties des agents 1 et 2 sur input_id.
    Retourne le tableau joint et le nombre de flux non appariés.
    """
    merged = agent1_table.merge(agent2_table, on="input_id", how="inner", suffixes=("", "_2"))
    report = {
        "matched": int(len(merged)),
        "agent1_only": int(len(agent1_table) - len(merged)),
        "agent2_only": int(len(agent2_table) - len(merged)),
    }
    return merged, report


def select_threshold(scores, y_true, grid=THRESHOLD_GRID) -> float:
    """Seuil maximisant le F1 sur les données fournies (à appeler sur la validation seulement)."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    best_threshold, best_f1 = grid[0], -1.0
    for threshold in grid:
        f1 = f1_score(y_true, scores >= threshold, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return float(best_threshold)


def build_policy_targets(df: pd.DataFrame, t1: float, t2: float, high: float = DEFAULT_HIGH) -> pd.Series:
    return apply_policy(df[FEATURE_NAMES], t1=t1, t2=t2, high=high).reset_index(drop=True)


def fit_tree_on_validation(validation: pd.DataFrame, t1, t2, high, max_depth):
    targets = build_policy_targets(validation, t1, t2, high)
    return train_decision_tree(validation[FEATURE_NAMES].reset_index(drop=True), targets, max_depth)


def security_metrics(actions, y_true) -> dict:
    """
    Mesure la réponse contre la vérité terrain.
    - detection_rate : part des attaques pour lesquelles une action autre que MONITOR est prise.
    - missed_attack_rate : part des attaques restées en MONITOR.
    - false_alert_rate : part des flux normaux pour lesquels une action autre que MONITOR est prise.
    - false_block_rate : part des flux normaux bloqués (RECOMMEND_BLOCK).
    - block_precision : part des blocages qui visent une vraie attaque.
    """
    actions = pd.Series(list(actions)).reset_index(drop=True)
    y = pd.Series(list(y_true)).reset_index(drop=True).astype(int)

    attack = y == 1
    normal = ~attack
    flagged = actions != "MONITOR"
    blocked = actions == "RECOMMEND_BLOCK"

    return {
        "flows": int(len(y)),
        "attacks": int(attack.sum()),
        "normals": int(normal.sum()),
        "detection_rate": ratio((flagged & attack).sum(), attack.sum()),
        "missed_attack_rate": ratio((~flagged & attack).sum(), attack.sum()),
        "false_alert_rate": ratio((flagged & normal).sum(), normal.sum()),
        "false_block_rate": ratio((blocked & normal).sum(), normal.sum()),
        "block_precision": ratio((blocked & attack).sum(), blocked.sum()),
        "blocks": int(blocked.sum()),
        "action_counts": {str(k): int(v) for k, v in actions.value_counts().items()},
    }


def baseline_actions(df: pd.DataFrame, t1: float, t2: float, high: float = DEFAULT_HIGH) -> dict:
    """Réponses de référence, à comparer à l'arbre."""
    n = len(df)
    return {
        "toujours_MONITOR": pd.Series(["MONITOR"] * n),
        "agent_1_seul": pd.Series(np.where(df["p_agent1"] >= t1, "ALERT", "MONITOR")),
        "agent_2_seul": pd.Series(np.where(df["p_agent2"] >= t2, "ALERT", "MONITOR")),
        "politique_sans_arbre": build_policy_targets(df, t1, t2, high),
    }


def high_sweep(validation: pd.DataFrame, t1: float, t2: float, grid=HIGH_GRID) -> list:
    """Effet de `high` sur la validation : blocages, précision et faux blocages."""
    rows = []
    for high in sorted(grid):
        metrics = security_metrics(build_policy_targets(validation, t1, t2, high), validation["y_true"])
        rows.append({
            "high": float(high),
            "blocks": metrics["blocks"],
            "block_precision": metrics["block_precision"],
            "false_block_rate": metrics["false_block_rate"],
            "detection_rate": metrics["detection_rate"],
        })
    return rows


def select_high(validation: pd.DataFrame, t1: float, t2: float, grid=HIGH_GRID,
                max_false_block: float = MAX_FALSE_BLOCK_RATE) -> float:
    """
    Critère écrit à l'avance : le plus petit `high` dont le taux de faux blocages,
    mesuré sur la validation, reste sous le plafond. Un `high` plus bas bloque davantage ;
    on prend donc le plus agressif qui respecte la limite. Si aucun ne la respecte,
    on prend le plus élevé.
    """
    for row in high_sweep(validation, t1, t2, grid):
        rate = row["false_block_rate"]
        if rate is None or rate <= max_false_block:
            return row["high"]
    return float(max(grid))


def fidelity(tree, df: pd.DataFrame, t1, t2, high) -> float:
    """Part des flux où l'arbre reproduit la politique (vérification de cohérence)."""
    predicted = pd.Series(tree.predict(df[FEATURE_NAMES])).reset_index(drop=True)
    targets = build_policy_targets(df, t1, t2, high)
    return float((predicted == targets).mean())


def depth_comparison(validation, test, t1, t2, high, depths=DEPTHS_TO_COMPARE) -> list:
    """Compare la fidélité et la taille de l'arbre selon sa profondeur."""
    rows = []
    for depth in depths:
        tree = fit_tree_on_validation(validation, t1, t2, high, depth)
        rows.append({
            "max_depth": int(depth),
            "leaves": int(tree.get_n_leaves()),
            "fidelity_validation": fidelity(tree, validation, t1, t2, high),
            "fidelity_test": fidelity(tree, test, t1, t2, high),
        })
    return rows


def run_evaluation(validation: pd.DataFrame, test: pd.DataFrame, max_depth: int = 4, high: float | None = None) -> dict:
    validation = validate_outputs(validation).reset_index(drop=True)
    test = validate_outputs(test).reset_index(drop=True)

    t1 = select_threshold(validation["p_agent1"], validation["y_true"])
    t2 = select_threshold(validation["p_agent2"], validation["y_true"])
    high_selected = high is None
    if high is None:
        high = select_high(validation, t1, t2)

    tree = fit_tree_on_validation(validation, t1, t2, high, max_depth)
    tree_actions = pd.Series(tree.predict(test[FEATURE_NAMES])).reset_index(drop=True)

    test_metrics = {
        "agent_3_arbre": security_metrics(tree_actions, test["y_true"]),
        "politique_sans_arbre": security_metrics(build_policy_targets(test, t1, t2, high), test["y_true"]),
    }
    for name, actions in baseline_actions(test, t1, t2, high).items():
        if name != "politique_sans_arbre":
            test_metrics[name] = security_metrics(actions, test["y_true"])

    severity = None
    if "attack_cat" in test.columns:
        attacks = test["y_true"] == 1
        frame = pd.DataFrame({
            "attack_cat": test.loc[attacks, "attack_cat"].astype(str).to_numpy(),
            "action": tree_actions[attacks.to_numpy()].to_numpy(),
        })
        severity = {
            str(category): {str(action): int(count) for action, count in row.items()}
            for category, row in pd.crosstab(frame["attack_cat"], frame["action"]).iterrows()
        }

    return {
        "protocol": "seuils choisis sur validation ; arbre entraîné sur validation ; évaluation sur test",
        "sizes": {
            "validation": int(len(validation)),
            "test": int(len(test)),
            "validation_attack_rate": float(validation["y_true"].mean()),
            "test_attack_rate": float(test["y_true"].mean()),
        },
        "thresholds": {
            "t1": t1,
            "t2": t2,
            "high": float(high),
            "t1_t2_selected_on": "validation_f1",
            "high_selected_on": "validation_false_block_cap" if high_selected else "fixed_by_caller",
            "max_false_block_rate": MAX_FALSE_BLOCK_RATE,
        },
        "high_sweep_validation": high_sweep(validation, t1, t2),
        "tree": {
            "max_depth": int(max_depth),
            "leaves": int(tree.get_n_leaves()),
            "fidelity_validation": fidelity(tree, validation, t1, t2, high),
            "fidelity_test": fidelity(tree, test, t1, t2, high),
        },
        "test_metrics": test_metrics,
        "depth_comparison": depth_comparison(validation, test, t1, t2, high),
        "severity_test_tree_actions": severity,
        "tree_rules": export_rules_text(tree),
        "actions_order": list(ACTIONS),
    }


def save_report(report: dict, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "agent3_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "agent3_tree_rules.txt").write_text(report["tree_rules"], encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer l'Agent 3 sur des sorties exportées.")
    parser.add_argument("--validation", required=True, help="CSV de validation")
    parser.add_argument("--test", required=True, help="CSV de test")
    parser.add_argument("--out", default="reports/results/agent3", help="Dossier de sortie")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    report = run_evaluation(load_outputs(args.validation), load_outputs(args.test), max_depth=args.max_depth)
    path = save_report(report, args.out)
    print(f"Rapport enregistré : {path}")
    print(f"Fidélité test : {report['tree']['fidelity_test']:.4f}")


if __name__ == "__main__":
    main()
