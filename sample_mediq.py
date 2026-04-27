"""
One-shot sampler: pick a deterministic 10-row subset from the MediQ dev
file (using a fixed random seed) and write the slim 4-column view used
by the GraphRAG retrieval test.

Note: the source filename on disk is `medqa_dev_convo.jsonl` (legacy name),
but the actual dataset is MediQ — every other identifier in this codebase
uses `mediq`.

Run once before `mediq_graphrag_test.py`:

    python sample_mediq.py

Output: ./output/mediq_sample.json
"""

import json
import os
import random

import config

# Source file kept under its original on-disk name to match the user's data dir.
INPUT_PATH = "./data/medqa_dev_convo.jsonl"
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "mediq_sample.json")

NUM_SAMPLES = 10
RANDOM_SEED = 42  # fixed → reproducible across runs


def main():
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"MediQ source file not found at {INPUT_PATH}. "
            f"Please place it there before running."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(RANDOM_SEED)
    indices = sorted(rng.sample(range(len(rows)), NUM_SAMPLES))

    samples = []
    for idx in indices:
        r = rows[idx]
        samples.append({
            "id": r.get("id", idx),
            "question": r.get("question", ""),
            "context": r.get("context", []),
            "options": r.get("options", {}),
            "answer": r.get("answer", ""),
            "answer_idx": r.get("answer_idx", ""),
        })

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source_path": INPUT_PATH,
                    "total_rows": len(rows),
                    "num_samples": NUM_SAMPLES,
                    "random_seed": RANDOM_SEED,
                    "sampled_indices": indices,
                },
                "samples": samples,
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"Sampled {len(samples)} rows from {len(rows)}")
    print(f"Indices: {indices}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
