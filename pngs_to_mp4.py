import argparse
import re
from pathlib import Path
from utils import PathManager
import cv2
import numpy as np
from PIL import Image


def natural_frame_sort_key(p):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**18


def pngs_to_mp4(frames_dir, video_path, fps):
    frames = sorted(frames_dir.glob("*.png"), key=natural_frame_sort_key)
    if not frames:
        raise FileNotFoundError(f"No png frames found in {frames_dir}")

    # Read first frame to get size
    first = Image.open(frames[0]).convert("RGB")
    w, h = first.size

    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different codec or path.")

    try:
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), resample=Image.NEAREST)

            rgb = np.array(img)  # (H,W,3) RGB
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    print(f"Saved: {video_path}  (frames={len(frames)}, fps={fps}, size={w}x{h})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=False, help="Name of the data")
    p.add_argument("--video_path", type=str, required=False, help="Video file path")
    p.add_argument("--frames_dir", type=str, required=False, help="Directory to save extracted frames")

    p.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    args = p.parse_args()

    if args.name is not None:
        pm = PathManager(args.name)
        video_path = pm.get_video_path()
        frames_dir = pm.get_frames_dir()
    else:
        video_path = Path(args.video_path)
        frames_dir = Path(args.frames_dir)

    pngs_to_mp4(frames_dir, video_path, args.fps)

