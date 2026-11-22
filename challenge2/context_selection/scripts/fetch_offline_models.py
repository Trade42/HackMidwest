"""Download the offline stand-in reader (T5-base, ONNX) and its tokenizer.

    python scripts/fetch_offline_models.py              # into models/t5
    python scripts/fetch_offline_models.py --dir D:/models/t5

These files let the pipeline run end to end without Hugging Face access. They are
a stand-in for the planned instruct model, not a replacement for it.

Sources (both on GitHub, pinned to a commit, checked against SHA-256):

* T5-base encoder and decoder from the ONNX model zoo (onnx/models);
* the T5 tokenizer file vendored in ComfyUI (the same 32k SentencePiece vocabulary).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ONNX_COMMIT = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
COMFY_COMMIT = "5df2ce62f1408572c18a12a3aefee326cd2305a5"
_ONNX = "https://media.githubusercontent.com/media/onnx/models/{ref}/validated/text/machine_comprehension/t5/model/{name}"
_COMFY = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/{ref}/comfy/text_encoders/t5_tokenizer/tokenizer.json"

#: name -> (size in bytes, sha256, urls tried in order)
FILES: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "t5-encoder-12.onnx": (
        438549611,
        "4523122d7cf0f50905694d84995633c8a0cc223da762d3eb2aaffa17251a6f60",
        (_ONNX.format(ref=ONNX_COMMIT, name="t5-encoder-12.onnx"),
         _ONNX.format(ref="main", name="t5-encoder-12.onnx")),
    ),
    "t5-decoder-with-lm-head-12.onnx": (
        650564941,
        "235afca35d3c47a5fd7209844ac11f3157fe7263fd074197a45b7f536e40ea56",
        (_ONNX.format(ref=ONNX_COMMIT, name="t5-decoder-with-lm-head-12.onnx"),
         _ONNX.format(ref="main", name="t5-decoder-with-lm-head-12.onnx")),
    ),
    "tokenizer.json": (
        2424256,
        "6cd1badb7467cdab971403bbeb1945070ca385652385a2b6549f672b08089dfc",
        (_COMFY.format(ref=COMFY_COMMIT), _COMFY.format(ref="master")),
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_name(dest.name + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            fh.write(block)
            done += len(block)
            print(f"\r  {dest.name}: {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch(target: Path, *, force: bool = False) -> bool:
    """Download every file into ``target``. Returns True when all hashes match."""
    target.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (size, digest, urls) in FILES.items():
        dest = target / name
        if dest.exists() and not force and sha256_of(dest) == digest:
            print(f"ok       {name} (already present)")
            continue
        good = False
        for url in urls:
            try:
                download(url, dest, size)
            except Exception as exc:
                print(f"  failed  {url}: {type(exc).__name__}: {exc}")
                continue
            if sha256_of(dest) == digest:
                good = True
                break
            print(f"  hash mismatch for {name} from {url}")
            dest.unlink(missing_ok=True)
        print(f"{'ok      ' if good else 'MISSING '} {name}")
        ok = ok and good
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="models/t5", help="target directory (default: models/t5)")
    ap.add_argument("--force", action="store_true", help="download again even if the files are present")
    args = ap.parse_args(argv)
    return 0 if fetch(Path(args.dir), force=args.force) else 1


if __name__ == "__main__":
    sys.exit(main())
