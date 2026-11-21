
import pandas as pd
import joblib

# Load trained Decision Tree model
decision_tree = joblib.load("agent3_decision_tree.pkl")


def run_agent3(agent1_result, agent2_result, input_id):

    # Check agent status
    if (agent1_result.get("status") != "success"
            or agent2_result.get("status") != "success"):
        return {
            "agent": "agent_3",
            "status": "error",
            "input_id": input_id,
            "prediction": {},
            "confidence": 0.0,
            "evidence": {"reason": "An input agent failed"}
        }

    # Check input IDs
    if (agent1_result.get("input_id") != input_id
            or agent2_result.get("input_id") != input_id):
        return {
            "agent": "agent_3",
            "status": "error",
            "input_id": input_id,
            "prediction": {},
            "confidence": 0.0,
            "evidence": {"reason": "Input ID mismatch"}
        }

    # Extract Agent 1 information
    agent1_is_attack = (
        agent1_result["prediction"]["label"] == "attack"
    )
    agent1_confidence = agent1_result["confidence"]

    # Extract Agent 2 information
    agent2_is_suspicious = (
        agent2_result["prediction"]["label"] == "suspicious_sequence"
    )
    agent2_confidence = agent2_result["confidence"]

    # Prepare model input
    features = pd.DataFrame([{
        "agent1_is_attack": int(agent1_is_attack),
        "agent1_confidence": agent1_confidence,
        "agent2_is_suspicious": int(agent2_is_suspicious),
        "agent2_confidence": agent2_confidence
    }])

    # Predict mitigation action
    action = decision_tree.predict(features)[0]
    confidence = decision_tree.predict_proba(features)[0].max()

    return {
        "agent": "agent_3",
        "status": "success",
        "input_id": input_id,
        "prediction": {"action": action},
        "confidence": float(confidence),
        "evidence": {
            "reason": "Decision based on Agent 1 and Agent 2 results"
        }
    }
