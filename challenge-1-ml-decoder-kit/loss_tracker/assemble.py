"""Write RESULTS.md from RESULTS.template.md: every line `{{include FILE}}` is replaced by that file's text.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.assemble

The included files are the tables that the scripts of this folder wrote into loss_tracker/results/, so the tables
of RESULTS.md are those files' lines, unchanged except that their headings are moved down two levels to sit
under the template's own (`shift K` moves them K levels instead). A file that does not exist yet becomes
"NOT MEASURED YET". `{{include FILE from N to M}}` takes lines N..M (1-based) of the file.
"""
import os
import re

HERE = os.path.dirname(__file__)


def main():
    out = []
    for line in open(os.path.join(HERE, "RESULTS.template.md")):
        m = re.fullmatch(r"\{\{include (\S+)(?: from (\d+) to (\d+))?(?: shift (\d+))?\}\}\s*", line)
        if not m:
            out.append(line)
            continue
        path = os.path.join(HERE, m.group(1))
        if not os.path.exists(path):
            out.append(f"**NOT MEASURED YET** (`{m.group(1)}`)\n")
            continue
        lines = open(path).read().rstrip("\n").split("\n")
        if m.group(2):
            lines = lines[int(m.group(2)) - 1:int(m.group(3))]
        shift = int(m.group(4)) if m.group(4) else 2
        lines = [("#" * shift + x) if re.match(r"#+ ", x) else x for x in lines]
        out.append("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write("".join(out))
    print(f"wrote RESULTS.md ({len(out)} blocks)")


if __name__ == "__main__":
    main()
