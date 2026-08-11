# Model assets

`yolox_tiny.onnx` is downloaded from the official YOLOX `0.1.1rc0` GitHub release and is intentionally ignored by Git because it is a binary model asset.

- URL: `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx`
- SHA-256: `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`
- Input: `images`, float32 `[1, 3, 416, 416]`
- Output: `output`, float32 `[1, 3549, 85]`

Re-download and verify through uv:

```bash
uv run person-search-download-models
```
