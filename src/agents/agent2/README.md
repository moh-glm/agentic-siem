# Agent 2 — Behaviour & Sequence Analysis Agent

Agent 2 analyzes ordered sequences of network flows using PyTorch Recurrent Neural Networks (**Vanilla RNN** and **LSTM**) to detect temporal attack patterns.

## Sequence-Labeling Rule
A sliding sequence window of length $L = 10$ (stride $S = 5$) is classified as an **attack (`1`)** if **at least one flow** within the window has `Label == 1`; otherwise it is classified as **normal (`0`)**. Windows are built strictly per CSV file/session and never cross file boundaries.

## Running Agent 2 Independently

### 1. Run Unit Tests
```bash
pytest tests/test_agent2.py -v
python -m src.agents.agent2.experiment
python -m src.agents.agent2.agent --checkpoint models/agent2_best_model.pt --demo
Dans la capture d'écran, PowerShell est bloqué en mode saisie multiligne (`>>`) parce que la chaîne `@'` du fichier `README.md` n'a pas encore été fermée par `'@`.

Pour sortir proprement de ce mode, créer `README.md`, lancer les tests et pousser sur GitHub :

1. **Appuie sur `Ctrl + C**` dans PowerShell pour quitter le mode `>>` et revenir à l'invite normale `PS C:\Users\ISSER TECH\Desktop\agentic-siem>`.
2. **Colle et exécute ce bloc unique** qui crée `README.md`, exécute `pytest` et pousse tes modifications sur GitHub :

```powershell
@'
# Agent 2 — Behaviour & Sequence Analysis Agent

Agent 2 analyzes ordered sequences of network flows using PyTorch Recurrent Neural Networks (**Vanilla RNN** and **LSTM**) to detect temporal attack patterns.

## Sequence-Labeling Rule
A sliding sequence window of length $L = 10$ (stride $S = 5$) is classified as an **attack (`1`)** if **at least one flow** within the window has `Label == 1`; otherwise it is classified as **normal (`0`)**. Windows are built strictly per CSV file/session and never cross file boundaries.

## Running Agent 2 Independently

### 1. Run Unit Tests
```bash
pytest tests/test_agent2.py -v

```

### 2. Run Training & GPU Evaluation Experiment

```bash
python -m src.agents.agent2.experiment

```

### 3. Run Standalone CLI Sanity Check

```bash
python -m src.agents.agent2.agent --checkpoint models/agent2_best_model.pt --demo

```

