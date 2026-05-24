"""Extract video frames to JPG, once, offline (uses the system ffmpeg).

    python scripts/preprocess_videos.py --src data/videos --dst data/frames --fps 0 --size 256

This decouples decoding from training: torchvision 0.27 can no longer decode
video, and full-decoding a long clip would blow up memory. We extract frames to
disk and let the dataset read them with torchvision's (still available) JPEG
decoder. Each input video becomes ``<dst>/<stem>/frame_000001.jpg ...``.

Prefer ``--fps 0`` (native): keep all temporal information on disk and control
the effective sampling rate via ``data.frame_stride`` at train time. Picking a
low ``--fps`` here *and* a ``frame_stride > 1`` subsamples twice (effective fps =
fps / stride), which is usually not what you want.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def extract(video: Path, out_dir: Path, fps: int, size: int, max_seconds: float | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = []
    if fps > 0:
        vf.append(f"fps={fps}")
    if size > 0:
        # Scale so the short side is `size`, keep aspect ratio, even dims.
        vf.append(f"scale='if(gt(iw,ih),-2,{size})':'if(gt(iw,ih),{size},-2)'")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", str(video)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-q:v", "3", str(out_dir / "frame_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("frame_*.jpg")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/videos")
    parser.add_argument("--dst", default="data/frames")
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="extraction fps; 0 = native (recommended). Control the effective "
        "sampling rate via data.frame_stride instead, so you can retune the "
        "temporal span without re-extracting. Setting both --fps<native AND "
        "frame_stride>1 stacks two subsamplings (effective fps = fps/stride).",
    )
    parser.add_argument("--size", type=int, default=256, help="short-side resize; 0 = original")
    parser.add_argument("--max-seconds", type=float, default=None, help="cap per video (quick tests)")
    args = parser.parse_args()

    src = Path(args.src)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise FileNotFoundError(f"no videos under {src}")

    for v in videos:
        out_dir = Path(args.dst) / v.stem
        n = extract(v, out_dir, args.fps, args.size, args.max_seconds)
        print(f"{v.name}: {n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
