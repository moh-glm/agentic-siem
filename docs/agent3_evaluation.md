# Évaluation de l'Agent 3

Ce document décrit le format de données attendu par l'évaluation de l'Agent 3 et le protocole suivi. Il ne demande aucune modification du code des agents 1 et 2 : il suffit de fournir un tableau par flux.

## 1. Format d'entrée

Un fichier CSV par découpage (`validation` et `test`), avec une ligne par flux :

| Colonne | Type | Obligatoire | Description |
|---|---|---|---|
| `input_id` | texte | oui | Identifiant du flux. Doit être unique et identique pour les deux agents. |
| `p_agent1` | flottant dans [0, 1] | oui | Probabilité d'attaque de l'agent 1 (`confidence` si `label = attack`, sinon `1 - confidence`). |
| `p_agent2` | flottant dans [0, 1] | oui | `attack_probability` de l'agent 2 pour la fenêtre se terminant à ce flux. |
| `y_true` | entier 0 ou 1 | oui | Vérité terrain : 0 normal, 1 attaque. Utilisée uniquement pour l'évaluation. |
| `attack_cat` | texte | non | Catégorie d'attaque. Si présente, une analyse de sévérité est produite. |

Important :
- `y_true` et `attack_cat` ne doivent jamais être utilisés comme entrées de l'agent 1 ou de l'agent 2.
- Les deux probabilités doivent concerner **les mêmes flux**, issus du **même découpage**.

## 2. Lancement

```bash
python -m src.agents.agent3.experiment \
    --validation data/export/agent3_validation.csv \
    --test data/export/agent3_test.csv \
    --out reports/results/agent3 \
    --max-depth 4
```

Sorties : `agent3_evaluation.json` (toutes les mesures) et `agent3_tree_rules.txt` (l'arbre en règles lisibles, pour le rapport).

## 3. Protocole

1. **Seuils** : `t1` et `t2` sont choisis sur la validation, en maximisant le F1 de chaque agent. `high` est choisi sur la validation selon le critère du plafond de faux blocages (voir `agent3_policy.md`). Rien n'est choisi sur le test.
2. **Arbre** : entraîné sur la validation, avec les actions de la politique comme cibles (voir `agent3_policy.md`).
3. **Évaluation sur le test** : performance réelle contre la vérité terrain, comparée à des baselines.

## 4. Mesures

- `detection_rate` : part des attaques traitées (action autre que MONITOR).
- `missed_attack_rate` : part des attaques restées en MONITOR.
- `false_alert_rate` : part des flux normaux pour lesquels une action autre que MONITOR est prise.
- `false_block_rate` : part des flux normaux bloqués.
- `block_precision` : part des blocages qui visent une vraie attaque.

Toute valeur `null` signifie qu'elle n'est pas définie sur le découpage (par exemple, aucun blocage effectué).

## 5. Baselines

| Baseline | Réponse |
|---|---|
| `toujours_MONITOR` | Ne traite aucune alerte. Sert de référence minimale. |
| `agent_1_seul` | ALERT si `p_agent1 >= t1`, sinon MONITOR. |
| `agent_2_seul` | ALERT si `p_agent2 >= t2`, sinon MONITOR. |
| `politique_sans_arbre` | La politique appliquée directement, sans apprentissage. |

L'apport de la combinaison se juge en comparant `agent_3_arbre` aux agents seuls : une détection plus élevée ne suffit pas si elle s'accompagne d'un grand nombre de fausses alertes.

## 6. Interprétation

- **Fidélité** (`fidelity_test`) : part des flux où l'arbre reproduit la politique. Elle doit être proche de 100 % et sert à vérifier l'implémentation. Ce n'est pas une mesure de détection, et elle ne doit pas être présentée comme un résultat principal.
- **Profondeur** (`depth_comparison`) : compare la fidélité et le nombre de feuilles de 2 à 5 niveaux. Une profondeur plus faible donne un arbre plus lisible, au prix d'une fidélité plus faible.
- **Sévérité** (`severity_test_tree_actions`) : répartition des actions par catégorie d'attaque. Elle sert à vérifier que les attaques graves reçoivent des actions plus fortes. Elle n'est pas une feature de l'arbre.
- **Limites** : les résultats dépendent de la qualité des agents 1 et 2 et du découpage utilisé. Ils doivent être présentés comme tels.

## 7. Export depuis les agents (découpage temporel)

Le découpage retenu par l'équipe est le découpage temporel de `src/agents/agent1/temporal_experiment.py`. Le script `export_outputs.py` l'applique, puis produit les CSV de validation et de test.

Prérequis :
- le modèle de l'agent 1 (`best_model.pkl`) doit être entraîné sur le split `training` temporal ;
- le checkpoint et le preprocessor de l'agent 2 doivent aussi être entraînés sur le split `training` temporal.

Commande (sur Kaggle, avec les fichiers traités) :

```bash
python -m src.agents.agent3.export_outputs \
    --agent1-models reports/results/agent1/models \
    --agent2-checkpoint models/agent2_best_model.pt \
    --agent2-preprocessor models/agent2_best_model_preprocessor.joblib \
    --out data/export
```

Règles appliquées :
- une fenêtre de l'agent 2 contient `WINDOW` flux consécutifs, dans le même fichier source et dans le même split ;
- la fenêtre se termine au flux évalué, sans regarder le futur ;
- un flux sans assez de prédécesseurs dans son fichier et son split est exclu, et le nombre d'exclusions est rapporté dans `export_report.json` ;
- `input_id` vaut `fichier:position`, unique et traçable.

## 8. Orchestrateur minimal

`src/orchestration/coordinator.py` enchaîne les trois agents pour chaque flux, avec le même `input_id`. Il mesure le temps de traitement par flux et produit un résumé (actions, erreurs, temps moyen). Les agents sont injectés : l'orchestrateur ne dépend que de leur méthode `predict`.
