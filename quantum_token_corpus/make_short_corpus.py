"""Create a <=15-word edition while preserving original IDs, labels and splits."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "short"
MAX_WORDS = 15


def words(sentence):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())


def shorten(row, metadata):
    a, b = row["candidate_0"], row["candidate_1"]
    factors = json.loads(metadata["factors"])
    style = metadata["template_id"].split("-", 1)[1]
    if row["tier"] == "1":
        verbs = ["won't fit inside", "cannot fit inside", "won't fit into",
                 "cannot fit into", "won't enter", "cannot enter",
                 "won't go inside", "cannot go inside"]
        sentence = f"{a} {verbs[int(style)]} {b}; it's too {factors['size_cue']}."
    elif row["tier"] == "2":
        prefixes = {"0": "won't", "1": "cannot", "2": "fails to", "2b": "failed to"}
        relation = "fit on" if factors["relation"] == "fit" else "cover"
        sentence = f"{a} {prefixes[style]} {relation} {b}; it's too {factors['size_cue']}."
    else:
        properties = {
            "brightness settings": ("brighter", "dimmer"),
            "temperatures": ("warmer", "cooler"),
            "speed settings": ("faster", "slower"),
            "volume settings": ("louder", "quieter"),
            "pressure settings": ("higher", "lower"),
            "voltage settings": ("higher", "lower"),
        }
        high, low = properties[factors["property"]]
        initial = high if factors["initial_left_high"] else low
        query = high if factors["query_high"] else low
        if factors["property"] in ("pressure settings", "voltage settings"):
            initial = f"at {initial} {factors['property'].split()[0]}"
        start, subject, changed, verb = {
            "0": ("starts", "Settings", "exchanged", "inspect"),
            "1": ("begins", "Settings", "swapped", "select"),
            "2": ("was", "Settings", "swapped", "examine"),
            "3": ("began", "Values", "exchanged", "choose"),
        }[style]
        change = changed if factors["swapped"] else "unchanged"
        sentence = f"{a} {start} {initial} than {b}. {subject} {change}; {verb} it, now {query}."
    sentence = sentence[0].upper() + sentence[1:]
    if len(words(sentence)) > MAX_WORDS:
        raise ValueError(f"Sentence exceeds {MAX_WORDS} words: {sentence}")
    return sentence


def generate():
    with (ROOT / "metadata_all.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {r["id"]: r for r in csv.DictReader(handle)}
    sentences = {}
    for split in ("train", "validation", "test", "test_ood"):
        with (ROOT / f"{split}_mixed.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sentences[row["id"]] = shorten(row, metadata[row["id"]])
    if len(set(sentences.values())) != len(sentences):
        raise ValueError("Shortening produced duplicate sentences")
    OUTPUT.mkdir(exist_ok=True)
    for path in sorted(ROOT.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if "sentence" in row:
                row["sentence"] = sentences[row["id"]]
        with (OUTPUT / path.name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for path in sorted(ROOT.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["sentence"] = sentences[row["id"]]
        (OUTPUT / path.name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    manifest.update(edition="short-v1", max_words=MAX_WORDS,
                    word_count="Alphabetic words; contractions count as one; punctuation excluded.",
                    source="Parent corpus; IDs, labels, candidate order and splits preserved.")
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lengths = [len(words(s)) for s in sentences.values()]
    print(f"Wrote {len(sentences)} unique sentences, {min(lengths)}–{max(lengths)} words, to {OUTPUT}")


if __name__ == "__main__":
    generate()
