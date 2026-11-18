
import unittest

from src.agents.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """Concrete agent used only for testing."""

    def __init__(self):
        super().__init__("test_agent")

    def predict(self, data):
        return self.build_response(
            input_id="test_001",
            prediction={"label": "normal"},
            confidence=0.95,
            evidence={"source": "unit_test"},
        )


class BaseAgentTests(unittest.TestCase):

    def test_abstract_agent_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseAgent("incomplete_agent")

    def test_concrete_agent_can_be_instantiated(self):
        agent = TestAgent()
        self.assertEqual(agent.agent_name, "test_agent")

    def test_response_has_standard_structure(self):
        agent = TestAgent()
        result = agent.predict({"example": 1})

        self.assertEqual(result["agent"], "test_agent")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_id"], "test_001")
        self.assertEqual(result["prediction"]["label"], "normal")
        self.assertEqual(result["confidence"], 0.95)

    def test_invalid_confidence_is_rejected(self):
        agent = TestAgent()

        with self.assertRaises(ValueError):
            agent.build_response(
                input_id="test_002",
                prediction={"label": "attack"},
                confidence=1.5,
                evidence={},
            )


if __name__ == "__main__":
    unittest.main()
