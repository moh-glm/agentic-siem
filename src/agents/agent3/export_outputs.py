"""
Export des sorties des agents 1 et 2 sur le découpage temporel commun.

Le découpage est celui de src/agents/agent1/temporal_experiment.py (create_temporal_splits),
appliqué aux données traitées complètes. Pour chaque split utilisé par l'évaluation
(validation et test), un CSV par flux est produit, au format attendu par
src/agents/agent3/experiment.py.

Règles de l'agent 2 :
- une fenêtre = WINDOW flux consécutifs, dans le même fichier source et dans le même split ;
- la fenêtre se termine au flux évalué (causale) ;
- un flux sans assez de prédécesseurs dans son fichier et son split n'a pas de score de l'agent 2
  et est exclu de l'export. Le nombre de flux exclus est rapporté.

Les modèles doivent avoir été entraînés sur le split `training` temporal, sinon l'évaluation est biaisée.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.agent1.temporal_experiment import DATA_DIR, FILES, create_temporal_splits
from src.agents.agent3.experiment import save_report, validate_outputs
from src.data.preprocess import clean_categorical_columns, clean_port_columns, get_feature_columns

WINDOW = 10
CHUNK = 4096
EXPORTED_SPLITS = {"validation": "agent3_validation.csv", "testing": "agent3_test.csv"}


def load_labelled_flows(data_dir=DATA_DIR, files=FILES) -> pd.DataFrame:
    """Charge les flux traités, avec leur fichier source et leur position dans ce fichier."""
    feature_columns = get_feature_columns()
    frames = []

    for filename in files:
        path = Path(data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Fichier traité introuvable : {path}")

        header = pd.read_csv(path, nrows=0).columns
        wanted = ["Stime", "Label", "attack_cat"] + feature_columns
        frame = pd.read_csv(path, usecols=[c for c in wanted if c in header], low_memory=False)

        missing = [c for c in feature_columns + ["Stime", "Label"] if c not in frame.columns]
        if missing:
            raise ValueError(f"{filename} : colonnes manquantes {missing}")

        frame["source_file"] = filename
        frame["row_in_file"] = np.arange(len(frame))
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["Stime"] = pd.to_numeric(df["Stime"], errors="coerce")
    if df["Stime"].isna().any():
        raise ValueError("Stime contient des valeurs manquantes ou invalides.")

    # Même tri que temporal_experiment.load_temporal_dataset : requis par create_temporal_splits.
    return df.sort_values("Stime", kind="mergesort").reset_index(drop=True)


def order_for_windows(split_df: pd.DataFrame) -> pd.DataFrame:
    """Ordre utilisé pour les fenêtres : fichier, puis temps, puis position dans le fichier."""
    return split_df.sort_values(
        ["source_file", "Stime", "row_in_file"], kind="mergesort"
    ).reset_index(drop=True)


def window_end_positions(source_files, window: int = WINDOW) -> np.ndarray:
    """Positions des flux qui peuvent terminer une fenêtre complète dans le même fichier."""
    files = np.asarray(source_files)
    ends = np.arange(window - 1, len(files))
    starts = ends - (window - 1)
    same_file = files[starts] == files[ends]
    return ends[same_file]


def score_agent2_windows(features_2d, ends, window: int, scorer, chunk: int = CHUNK) -> np.ndarray:
    """Calcule le score de l'agent 2 pour chaque fenêtre, par blocs pour limiter la mémoire."""
    features_2d = np.asarray(features_2d, dtype=np.float32)
    offsets = np.arange(window)
    scores = np.empty(len(ends), dtype=float)

    for start in range(0, len(ends), chunk):
        block_ends = ends[start:start + chunk]
        index = block_ends[:, None] - (window - 1) + offsets[None, :]
        scores[start:start + len(block_ends)] = np.asarray(scorer(features_2d[index])).reshape(-1)

    return scores


def export_split(split_df, agent1_scorer, agent2_preprocess, agent2_scorer,
                 feature_columns=None, window: int = WINDOW):
    """Produit le tableau d'évaluation d'un split. Retourne (tableau, rapport)."""
    feature_columns = feature_columns or get_feature_columns()
    ordered = order_for_windows(split_df)

    p_agent1 = np.asarray(agent1_scorer(ordered[feature_columns]), dtype=float)
    features_2d = agent2_preprocess(ordered[feature_columns])

    ends = window_end_positions(ordered["source_file"].to_numpy(), window)
    p_agent2 = score_agent2_windows(features_2d, ends, window, agent2_scorer)

    rows = ordered.iloc[ends]
    table = pd.DataFrame({
        "input_id": rows["source_file"].astype(str).to_numpy()
        + ":" + rows["row_in_file"].astype(str).to_numpy(),
        "p_agent1": p_agent1[ends],
        "p_agent2": p_agent2,
        "y_true": rows["Label"].astype(int).to_numpy(),
    })
    if "attack_cat" in ordered.columns:
        table["attack_cat"] = rows["attack_cat"].astype(str).to_numpy()

    report = {
        "flows_in_split": int(len(ordered)),
        "exported": int(len(table)),
        "excluded_without_window": int(len(ordered) - len(table)),
        "attack_rate_exported": float(table["y_true"].mean()) if len(table) else None,
    }
    return validate_outputs(table), report


def make_agent1_scorer(model):
    """
    Probabilité d'attaque du modèle sélectionné de l'agent 1, en batch.
    Équivaut à la valeur dérivée par Agent1.predict : P(attaque) = confidence si attaque, sinon 1 - confidence.
    """
    def score(features: pd.DataFrame) -> np.ndarray:
        flow = clean_categorical_columns(clean_port_columns(features.copy()))
        probabilities = model.predict_proba(flow)
        attack_index = list(model.classes_).index(1)
        return probabilities[:, attack_index]
    return score


def make_agent2_preprocess(preprocessor):
    def preprocess(features: pd.DataFrame) -> np.ndarray:
        transformed = preprocessor.transform(features)
        return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    return preprocess


def make_agent2_scorer(model, device: str = "cpu"):
    """Probabilité d'attaque du LSTM pour un lot de fenêtres (n, WINDOW, features)."""
    import torch  # import local : l'évaluation n'exige pas torch

    model.eval()

    def score(windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(device)
            return torch.sigmoid(model(tensor)).reshape(-1).cpu().numpy()
    return score


def export_all(agent1_model, agent2_model, agent2_preprocessor, out_dir,
               data_dir=DATA_DIR, files=FILES, window: int = WINDOW) -> dict:
    """Exporte les splits de validation et de test. Retourne le rapport d'export."""
    df = load_labelled_flows(data_dir, files)
    splits = create_temporal_splits(df)

    agent1_scorer = make_agent1_scorer(agent1_model)
    agent2_preprocess = make_agent2_preprocess(agent2_preprocessor)
    agent2_scorer = make_agent2_scorer(agent2_model)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"window": window, "splits": {}}

    for split_name, file_name in EXPORTED_SPLITS.items():
        table, split_report = export_split(
            splits[split_name], agent1_scorer, agent2_preprocess, agent2_scorer, window=window
        )
        table.to_csv(out_dir / file_name, index=False)
        report["splits"][split_name] = split_report

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import joblib
    from src.agents.agent2.models import load_trained_sequence_model  # torch requis ici seulement

    parser = argparse.ArgumentParser(description="Exporter les sorties des agents 1 et 2 pour l'agent 3.")
    parser.add_argument("--agent1-models", required=True, help="Dossier contenant best_model.pkl")
    parser.add_argument("--agent2-checkpoint", required=True, help="Checkpoint .pt de l'agent 2")
    parser.add_argument("--agent2-preprocessor", default=None, help="Preprocessor .joblib de l'agent 2")
    parser.add_argument("--out", default="data/export", help="Dossier de sortie des CSV")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    agent1_model = joblib.load(Path(args.agent1_models) / "best_model.pkl")
    agent2_model, _, agent2_preprocessor = load_trained_sequence_model(
        checkpoint_path=args.agent2_checkpoint,
        preprocessor_path=args.agent2_preprocessor,
    )

    report = export_all(agent1_model, agent2_model, agent2_preprocessor, args.out, window=args.window)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
