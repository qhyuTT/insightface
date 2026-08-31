from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

YOLOX_TINY_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_tiny.onnx"
)
YOLOX_TINY_SHA256 = "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
DEFAULT_DESTINATION = Path("models/yolox_tiny.onnx")


@dataclass(frozen=True, slots=True)
class EmbeddingModelManifest:
    model_name: str
    recognition_filename: str
    recognition_sha256: str
    embedding_dimension: int
    input_size: tuple[int, int]


BUFFALO_L_EMBEDDING_MANIFEST = EmbeddingModelManifest(
    model_name="buffalo_l",
    recognition_filename="w600k_r50.onnx",
    recognition_sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    embedding_dimension=512,
    input_size=(112, 112),
)

EMBEDDING_MODEL_MANIFESTS = {
    BUFFALO_L_EMBEDDING_MANIFEST.model_name: BUFFALO_L_EMBEDDING_MANIFEST,
}


def main() -> None:
    destination = DEFAULT_DESTINATION
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == YOLOX_TINY_SHA256:
        print(f"YOLOX-Tiny already present and verified: {destination}")
        return

    temporary = destination.with_suffix(".onnx.part")
    print(f"Downloading official YOLOX-Tiny ONNX to {destination} ...")
    try:
        request = urllib.request.Request(YOLOX_TINY_URL, headers={"User-Agent": "person-search-poc"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        actual = sha256(temporary)
        if actual != YOLOX_TINY_SHA256:
            raise RuntimeError(
                f"YOLOX-Tiny checksum mismatch: expected {YOLOX_TINY_SHA256}, got {actual}"
            )
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"Model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Downloaded and verified {destination} ({YOLOX_TINY_SHA256})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
