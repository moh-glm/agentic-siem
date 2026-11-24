"""
Orchestrateur minimal : enchaîne l'agent 1 (par flux), l'agent 2 (par fenêtre) et l'agent 3.

Les agents sont injectés : l'orchestrateur ne dépend pas de leur implémentation,
seulement de leur méthode predict(dict) -> dict au format commun.
Il mesure le temps de traitement de chaque flux, comme le demande l'architecture.
"""

import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class Coordinator:
    agent1: object
    agent2: object
    agent3: object

    def process_flow(self, flow_id: str, flow_features: dict, sequence) -> dict:
        """
        Traite un flux :
        - flow_features : champs du flux pour l'agent 1 ;
        - sequence : fenêtre de flux (tableau 2D) pour l'agent 2, se terminant à ce flux.
        Le même input_id est transmis aux trois agents.
        """
        start = time.perf_counter()

        agent1_result = self.agent1.predict({**flow_features, "input_id": flow_id})
        agent2_result = self.agent2.predict({"input_id": flow_id, "sequence": sequence})
        decision = self.agent3.predict({
            "input_id": flow_id,
            "agent1_result": agent1_result,
            "agent2_result": agent2_result,
        })

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "input_id": flow_id,
            "decision": decision,
            "agent1": agent1_result,
            "agent2": agent2_result,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def process_batch(self, items) -> dict:
        """
        items : itérable de tuples (flow_id, flow_features, sequence).
        Retourne les résultats et un résumé : actions, erreurs, temps moyen.
        """
        results = [self.process_flow(*item) for item in items]

        actions = Counter()
        errors = 0
        for result in results:
            decision = result["decision"]
            if decision.get("status") == "success":
                actions[decision["prediction"]["action"]] += 1
            else:
                errors += 1

        times = [r["elapsed_ms"] for r in results]
        summary = {
            "flows": len(results),
            "actions": dict(actions),
            "errors": errors,
            "mean_ms_per_flow": round(sum(times) / len(times), 3) if times else None,
        }
        return {"results": results, "summary": summary}
