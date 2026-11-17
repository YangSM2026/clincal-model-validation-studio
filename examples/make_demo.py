"""Generate a fictional binary-outcome dataset with two predictor types."""
import csv
import math
import random
from pathlib import Path


def generate(destination: Path, n: int = 600, seed: int = 173) -> None:
    rng = random.Random(seed)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["age", "baseline_score", "group", "event_30d", "fixed_risk"])
        for _ in range(n):
            age = round(rng.uniform(25, 80), 1)
            score = round(rng.gauss(0, 1.2), 3)
            group = rng.choice(["01", "02", "03"])
            eta = -1.55 + (age - 50) * 0.032 + 1.05 * score + 0.6 * (group == "03")
            event = "yes" if rng.random() < 1 / (1 + math.exp(-eta)) else "no"
            measured_age = age if rng.random() > 0.06 else None
            measured_score = score if rng.random() > 0.05 else None
            measured_group = group if rng.random() > 0.04 else ""
            # A separate illustrative score with prespecified coefficients.
            fixed_eta = -1.45 + ((measured_age or 50) - 50) * 0.028 + 0.85 * (measured_score or 0)
            fixed_risk = 1 / (1 + math.exp(-fixed_eta))
            writer.writerow([measured_age, measured_score, measured_group, event, f"{fixed_risk:.6f}"])


if __name__ == "__main__":
    generate(Path(__file__).with_name("demo.csv"))
