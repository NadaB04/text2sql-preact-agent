"""Draw a fixed 100-question subset of the 176 held-out BIRD questions (stratified by database, seed 2026).

    python scripts/sample_heldout.py data/questions_heldout.json data/questions_heldout100.json

The sample is drawn before any held-out run, from the question ids only (no results are read), so it
cannot be tuned. Each database keeps its share of the 176 questions (largest-remainder rounding).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict


def main(src: str, dst: str, n: int = 100, seed: int = 2026) -> None:
    qs = json.load(open(src, encoding="utf-8"))
    by_db = defaultdict(list)
    for q in qs:
        by_db[q["db_id"]].append(q)
    exact = {db: n * len(v) / len(qs) for db, v in by_db.items()}
    take = {db: int(x) for db, x in exact.items()}
    for db in sorted(exact, key=lambda d: exact[d] - take[d], reverse=True)[: n - sum(take.values())]:
        take[db] += 1
    rng = random.Random(seed)
    picked = []
    for db in sorted(by_db):
        picked += rng.sample(by_db[db], take[db])
    picked.sort(key=lambda q: q["question_id"])
    json.dump(picked, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(picked)} questions to {dst}: {take}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
