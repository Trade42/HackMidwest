"""Generate a controlled pronoun corpus with held-out sentence configurations."""

import csv
from itertools import product
from pathlib import Path

import numpy as np


NAMES = (("Ava", "Mia"), ("Zoe", "Bea"), ("Kira", "Lana"), ("Nora", "Iris"))
OBJECTS = ("book", "coat", "charger", "camera", "scarf", "tablet", "umbrella", "folder")
TIMES = ("after class", "before lunch", "that evening", "the next day")
ACTIONS = (("returned it", "kept it"), ("reclaimed it", "asked for it back"))
OUTPUT = Path(__file__).with_name("pronouns_nuanced.csv")


def main():
    rng = np.random.default_rng(241124)
    worlds = list(product(range(2), range(2), range(2), range(2),
                          range(len(OBJECTS)), range(len(TIMES))))
    validation = set()
    test = set()
    for transfer, role, voice in product(range(2), range(2), range(2)):
        pool = [world for world in worlds if (world[0], world[1], world[3]) == (transfer, role, voice)]
        order = rng.permutation(len(pool))
        test.update(pool[i] for i in order[:16])
        validation.update(pool[i] for i in order[16:24])

    rows = []
    for world in worlds:
        transfer, role, synonym, voice, obj, when = world
        a, b = NAMES[rng.integers(len(NAMES))]
        action = ACTIONS[role][synonym]
        if transfer == 0:
            holder, owner = b, a
            event = (f"{a} lent {b} the {OBJECTS[obj]}" if voice == 0 else
                     f"{b} was lent the {OBJECTS[obj]} by {a}")
        else:
            holder, owner = a, b
            event = (f"{a} borrowed {b}'s {OBJECTS[obj]}" if voice == 0 else
                     f"{b}'s {OBJECTS[obj]} was borrowed by {a}")
        sentence = f"{event}; she {action} {TIMES[when]}."
        first_mentioned, second_mentioned = (a, b) if voice == 0 else (b, a)
        referent = holder if role == 0 else owner
        split = ("fresh_test" if world in test else
                 "validation" if world in validation else "train")
        world_id = "-".join(map(str, world))
        for order in (0, 1):
            candidates = ((first_mentioned, second_mentioned) if order == 0 else
                          (second_mentioned, first_mentioned))
            rows.append({"group_id": f"{world_id}-{order}", "sentence": sentence,
                         "candidate_0": candidates[0], "candidate_1": candidates[1],
                         "label": candidates.index(referent), "split": split})
    rng.shuffle(rows)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sentences to {OUTPUT}")


if __name__ == "__main__":
    main()
