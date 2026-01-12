#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from utils import PathManager
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video file and save as images.")
    p.add_argument("--name", type=str, required=False, help="Name of the data")
    p.add_argument("--video_path", type=str, required=False, help="Video file path")
    p.add_argument("--frames_dir", type=str, required=False, help="Directory to save extracted frames")
    p.add_argument("--every", type=int, default=1, help="Save every N-th frame (default: 1 = all frames)")
    p.add_argument("--start", type=int, default=0, help="Start frame index (inclusive, default: 0)")
    p.add_argument("--end", type=int, default=-1, help="End frame index (inclusive). -1 means until the end.")
    p.add_argument("--max_frames", type=int, default=-1, help="Max number of frames to save. -1 means no limit.")
    p.add_argument("--ext", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image extension")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100). Used only for jpg/jpeg.")
    p.add_argument("--zero_pad", type=int, default=7, help="Zero padding for frame filenames (default: 7 -> 0000000.png)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.name is not None:
        pm = PathManager(args.name)
        video_path = pm.get_video_path()
        frames_dir = pm.get_frames_dir()
    else:
        video_path = Path(args.video_path)
        frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if args.every <= 0:
        raise ValueError("--every must be >= 1")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.end != -1 and args.end < args.start:
        raise ValueError("--end must be -1 or >= --start")
    if not (1 <= args.quality <= 100):
        raise ValueError("--quality must be in [1, 100]")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine end frame
    end_frame = args.end
    if end_frame == -1:
        end_frame = (total_frames - 1) if total_frames != -1 else 10**18  # practically "until read fails"

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    saved = 0
    current = args.start

    # JPEG params
    imwrite_params = []
    ext = args.ext.lower()
    if ext in ("jpg", "jpeg"):
        imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]

    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break  # end of stream or read error

        if (current - args.start) % args.every == 0:
            filename = f"{current:0{args.zero_pad}d}.{ext}"
            out_path = frames_dir / filename

            ok_write = cv2.imwrite(str(out_path), frame, imwrite_params)
            if not ok_write:
                cap.release()
                raise RuntimeError(f"Failed to write image: {out_path}")

            saved += 1
            if args.max_frames != -1 and saved >= args.max_frames:
                break

        current += 1

    cap.release()

    # Simple summary
    print("Done.")
    print(f"Video: {video_path}")
    if fps and fps > 0:
        print(f"FPS: {fps}")
    if total_frames != -1:
        print(f"Total frames (reported): {total_frames}")
    print(f"Saved frames: {saved}")
    print(f"Output dir: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
