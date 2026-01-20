#!/usr/bin/env python3
"""
Export representative reference frames to PNG for inclusion in the paper.

By default, it reads the "top_scored_captions" from:
  results/reference_example_report.json
and extracts those frame indices from:
  data/videos/1.mp4

Outputs PNGs into:
  figures/
"""

import json
from pathlib import Path

import cv2  # type: ignore


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_frame(video_path: Path, frame_index: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Seek and read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write in BGR -> PNG (cv2 handles it)
    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Could not write image: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / "results" / "reference_example_report.json"
    video_path = repo_root / "data" / "videos" / "1.mp4"
    out_dir = repo_root / "figures"

    report = read_json(report_path)
    top = report.get("top_scored_captions", [])[:6]
    if not top:
        raise RuntimeError("No top_scored_captions found in report. Re-run run_complete_pipeline.py.")

    for item in top:
        frame_index = int(item["frame_index"])
        out_path = out_dir / f"ref_frame_{frame_index}.png"
        export_frame(video_path, frame_index, out_path)
        print(f"Wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

