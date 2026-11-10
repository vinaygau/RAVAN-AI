import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_game_runs import summarize_run
from src.ravan_game.logger import FIELDS, SimulationLogger
from src.ravan_game.risk import analyze_risk
from src.ravan_game.simulation import DrivingSimulation


class RunLoggingTests(unittest.TestCase):
    def test_logger_writes_structured_telemetry_without_conversation_data(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = SimulationLogger(Path(directory), sample_interval_s=1.0)
            simulation = DrivingSimulation(seed=93)
            assessment = analyze_risk(simulation)
            try:
                self.assertTrue(logger.log(simulation, assessment, force=True))
                self.assertFalse(logger.log(simulation, assessment))
            finally:
                logger.close()

            with logger.path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], simulation.run_id)
            self.assertEqual(rows[0]["risk_level"], assessment.risk_level)
            self.assertIn("seed", FIELDS)
            self.assertNotIn("chat", " ".join(FIELDS).lower())
            self.assertNotIn("audio", " ".join(FIELDS).lower())

    def test_evaluator_summarizes_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = SimulationLogger(Path(directory))
            simulation = DrivingSimulation(seed=93)
            assessment = analyze_risk(simulation)
            try:
                logger.log(simulation, assessment, force=True)
            finally:
                logger.close()

            report = summarize_run(logger.path)
            self.assertEqual(report["run_ids"], [simulation.run_id])
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["maximum_risk_score"], assessment.risk_score)


if __name__ == "__main__":
    unittest.main()
