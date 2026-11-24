# Politique de réponse de l'Agent 3

## 1. Rôle

L'Agent 3 transforme les sorties des agents 1 et 2 en une action de réponse recommandée :
`MONITOR`, `ALERT`, `INVESTIGATE` ou `RECOMMEND_BLOCK`. Il ne bloque jamais de trafic réel.

## 2. Entrées (contrat)

Les trois éléments suivants sont obligatoires pour un même flux :

| Source | Champ utilisé | Signification |
|---|---|---|
| Agent 1 (`agent_1`) | `prediction.label` et `confidence` | Modèle sélectionné (Naive Bayes ou Logistic Regression). Probabilité d'attaque : `confidence` si `label = attack`, sinon `1 - confidence` |
| Agent 2 (`agent_2`) | `prediction.attack_probability` | Probabilité d'attaque du LSTM sur la fenêtre se terminant à ce flux |
| Orchestrateur | `input_id` | Identifiant du flux, identique dans les deux sorties |

Si l'une des sorties a `status = "error"`, si les identifiants diffèrent, si un label est inconnu, ou si une probabilité est absente ou hors de [0, 1], l'Agent 3 renvoie `status = "error"`.

## 3. Features de l'arbre

- `p_agent1` : probabilité d'attaque de l'agent 1.
- `p_agent2` : probabilité d'attaque de l'agent 2.

L'agent 1 n'exécute que son modèle sélectionné et ne renvoie que la probabilité de la classe prédite. L'agent 3 la convertit donc en probabilité d'attaque. Le nom `p_agent1` remplace `p_lr`, car le modèle sélectionné peut être Naive Bayes.

## 4. Politique

Soit `t1` le seuil de l'agent 1, `t2` le seuil de l'agent 2 et `high` le seuil de confiance élevée.

| Condition | Action |
|---|---|
| `p_agent1 < t1` et `p_agent2 < t2` | `MONITOR` |
| Un seul des deux agents atteint son seuil | `INVESTIGATE` |
| Les deux atteignent leur seuil, et `min(p_agent1, p_agent2) < high` | `ALERT` |
| Les deux dépassent `high` | `RECOMMEND_BLOCK` |

Un seuil est atteint lorsque la probabilité est supérieure ou égale à ce seuil.

## 5. Choix des seuils

- `t1` et `t2` sont choisis sur le **jeu de validation** (valeur maximisant le F1 de chaque agent). Ils ne sont jamais choisis sur le test.
- `high` est choisi sur la validation avec un critère écrit à l'avance : le plus petit `high` dans {0,80 ; 0,85 ; 0,90 ; 0,95 ; 0,99} dont le taux de faux blocages reste inférieur à 1 %. Un `high` plus bas bloque davantage, donc on retient le plus agressif qui respecte la limite. Si aucun ne la respecte, on retient 0,99. La valeur choisie et le tableau de sensibilité (`high_sweep_validation`) sont dans le rapport d'évaluation.

## 6. Confiance de la décision

La confiance de la décision est la plus faible des certitudes des deux agents :

```
confidence = min(max(p_agent1, 1 - p_agent1), max(p_agent2, 1 - p_agent2))   ∈ [0.5, 1]
```

Une décision n'est pas plus sûre que l'agent le moins certain sur lequel elle repose.

## 7. Entraînement de l'arbre

- L'arbre est entraîné sur les sorties des agents 1 et 2 du **jeu de validation**, avec les actions produites par la politique ci-dessus.
- Profondeur maximale : 4 par défaut, à comparer entre 2 et 5 dans l'évaluation.
- `class_weight = "balanced"` pour compenser la rareté de `RECOMMEND_BLOCK`.
- `random_state = 42`.

Comme la politique est une fonction déterministe de `p_agent1` et `p_agent2`, l'arbre reproduit presque parfaitement la politique. Cette **fidélité** vérifie la cohérence de l'implémentation, mais ne mesure pas la qualité de la détection.

## 8. Explicabilité

Chaque sortie contient `evidence.rules`, la liste des conditions du chemin de décision, par exemple `["p_agent2 > 0.487", "p_agent1 > 0.887", "p_agent2 > 0.887"]`. Ces seuils sont **appris** par l'arbre à partir des données : ils ne sont pas forcément égaux à `t1`, `t2` ou `high`. L'exemple ci-dessus provient d'une grille synthétique, pas des données réelles. Le rapport doit aussi présenter l'arbre complet avec `export_rules_text`.

## 9. Sortie

```python
{
    "agent": "agent_3",
    "status": "success",
    "input_id": "flow_001",
    "prediction": {"action": "RECOMMEND_BLOCK"},
    "confidence": 0.95,
    "evidence": {
        "features": {"p_agent1": 0.95, "p_agent2": 0.97},
        "rules": ["p_agent2 > 0.487", "p_agent1 > 0.887", "p_agent2 > 0.887"],
        "policy_action": "RECOMMEND_BLOCK",
        "thresholds": {"t1": 0.5, "t2": 0.5, "high": 0.9},
    },
}
```

## 10. Limites

- La politique est une convention documentée, pas une vérité terrain.
- La sévérité de l'attaque (`attack_cat`) n'est pas utilisée à l'inférence, car elle n'est pas connue en production. Elle peut servir à une analyse : répartition des actions par catégorie d'attaque.
- La qualité de l'Agent 3 dépend directement de celle des agents 1 et 2, et de leur découpage commun des données.
