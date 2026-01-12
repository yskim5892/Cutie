from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from utils import rgb_to_p, PathManager
from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from pngs_to_mp4 import pngs_to_mp4

# Prefer Cutie's ResultSaver; fall back to local copy if present
try:
    from cutie.inference.utils.results_utils import ResultSaver  # type: ignore
except Exception:  # pragma: no cover
    from results_utils import ResultSaver  # type: ignore

from cutie_state_change_vlm import (
    CutieStateChangePipeline,
    OpenAIVLMAnnotator,
    iter_frames_from_dir,
    iter_frames_from_video,
    save_state_changes_json,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the data")
    parser.add_argument("--input_type", type=str, default="video", help="frames or video")
    parser.add_argument("--obj1_id", type=int, required=True, help="Object ID to detect occlusion for.")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI VLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="VLM temperature (gpt-5.2 supports this).")
    parser.add_argument("--max_internal_size", type=int, default=480, help="Cutie internal resize; -1 keeps original.")
    parser.add_argument("--output_fps", type=float, default=30.0, help="FPS for output mask video.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Resolve paths from --name
    pm = PathManager(args.name)
    args.video_path = pm.get_video_path()
    args.frames_dir = pm.get_frames_dir()
    args.mask_dir = pm.get_input_mask_dir()
    args.output_mask_dir = pm.get_output_mask_dir()
    args.output_video_path = pm.get_output_video_path()
    args.out_json = pm.get_output_state_changes_path()

    # Load Cutie model
    cutie = get_default_model().to(args.device)
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = args.max_internal_size

    # Load initial mask (frame 0). We follow process_video.py's pattern: if RGB, convert to P.
    all_mask_frames = sorted(os.listdir(args.mask_dir))
    if len(all_mask_frames) == 0:
        raise FileNotFoundError(f"No mask frames found in {args.mask_dir}")
    first_mask_frame = all_mask_frames[0]
    first_mask = Image.open(os.path.join(args.mask_dir, first_mask_frame))

    palette = None
    use_long_id = False  # IMPORTANT: keep same colors as input by saving indexed+palette

    if first_mask.mode == "P":
        palette = first_mask.getpalette()
    elif first_mask.mode == "RGB":
        # Convert RGB annotation into indexed mask + palette.
        # For your case (RGB with DAVIS-like colors), we intentionally SAVE as indexed+palette
        # so that the output keeps EXACTLY the same colors as the input.
        first_mask, palette, _ = rgb_to_p(first_mask)
        use_long_id = False
    elif first_mask.mode == "L":
        palette = None
    else:
        raise ValueError(f"Unknown mode {first_mask.mode} in {first_mask_frame}.")

    mask_np = np.array(first_mask).astype(np.int64)
    objects = np.unique(mask_np)
    objects = objects[objects != 0].tolist()
    if args.obj1_id not in objects:
        raise ValueError(f"--obj1_id {args.obj1_id} not present in init_mask object IDs: {objects}")

    init_mask_tensor = torch.from_numpy(mask_np).to(args.device)

    # VLM annotator
    vlm = OpenAIVLMAnnotator(model=args.model, temperature=args.temperature)

    # ResultSaver (same pattern as process_video.py)
    mask_saver = None
    if args.output_mask_dir is not None:
        os.makedirs(args.output_mask_dir, exist_ok=True)
        mask_saver = ResultSaver(
            args.output_mask_dir,
            "",  # no subfolder; write directly under output_mask_dir
            dataset="",
            object_manager=processor.object_manager,
            use_long_id=use_long_id,
            palette=palette,
        )

    pipe = CutieStateChangePipeline(
        processor=processor,
        obj1_id=args.obj1_id,
        vlm=vlm,
        mask_saver=mask_saver,
    )

    # Iterate frames
    all_sc = []
    if args.input_type == "frames":
        frame_iter = enumerate(iter_frames_from_dir(args.frames_dir))
    else:
        frame_iter = enumerate(iter_frames_from_video(args.video_path))

    for ti, frame_rgb in frame_iter:
        img_tensor = to_tensor(frame_rgb).to(args.device).float()

        if ti == 0:
            pipe.feed_frame(ti, frame_rgb, img_tensor, init_mask_tensor=init_mask_tensor, init_objects=objects)
        else:
            pipe.feed_frame(ti, frame_rgb, img_tensor)

        all_sc.extend(pipe.pop_state_changes(image_detail="low"))

    save_state_changes_json(all_sc, args.out_json)
    print(f"[OK] wrote {len(all_sc)} state changes to {args.out_json}")

    if mask_saver is not None:
        mask_saver.end()
        pngs_to_mp4(Path(args.output_mask_dir), Path(args.output_video_path), fps=args.output_fps)


if __name__ == "__main__":
    main()
