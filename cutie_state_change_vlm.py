"""
Cutie + Rule-based Occlusion Detection + OpenAI VLM Query -> State Change tuples.

State change format:
    (start_frame, end_frame, obj1, obj2, description)

This version saves predicted masks via Cutie's ResultSaver (same way as process_video.py),
so that the output mask colors/palette stay consistent with the input annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Any

import os
import json
import time
import base64
import io
import logging

import numpy as np
import torch

import cv2  # used for contours, video IO

from PIL import Image

log = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class OcclusionEvent:
    obj_id: int
    start_frame: int  # first frame judged "occluded"
    end_frame: int    # last frame judged "occluded"
    pre_frame: int    # last visible frame before start_frame
    post_frame: int   # first visible frame after end_frame


@dataclass
class StateChange:
    start_frame: int
    end_frame: int
    obj1: int
    obj2: int
    description: str
    meta: Optional[dict] = None


# -----------------------------
# Utility: masks & geometry
# -----------------------------
def bin_mask_from_id(cls_mask: np.ndarray, obj_id: int) -> np.ndarray:
    return (cls_mask == obj_id).astype(np.uint8)

def mask_area(mask01: np.ndarray) -> int:
    return int(mask01.sum())

def mask_bbox(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)

def bbox_expand(b: Tuple[int,int,int,int], margin: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = b
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)
    return (x1, y1, x2, y2)

def centroid(mask01: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))

def l2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)


# -----------------------------
# Rule-based occlusion detector
# -----------------------------
class OcclusionDetector:
    def __init__(
        self,
        *,
        visible_ratio_th: float = 0.35,
        min_area_px: int = 50,
        ema_alpha: float = 0.05,
        start_patience: int = 2,
        end_patience: int = 2,
        warmup_frames: int = 5,
    ):
        self.visible_ratio_th = float(visible_ratio_th)
        self.min_area_px = int(min_area_px)
        self.ema_alpha = float(ema_alpha)
        self.start_patience = int(start_patience)
        self.end_patience = int(end_patience)
        self.warmup_frames = int(warmup_frames)

        self._baseline_area: Optional[float] = None
        self._frame_count = 0

        self._occluded = False
        self._not_visible_streak = 0
        self._visible_streak = 0

        self._current_start: Optional[int] = None
        self._last_visible_frame: Optional[int] = None

        self._pending_events: List[OcclusionEvent] = []

    def _is_visible(self, area: int) -> bool:
        if area < self.min_area_px:
            return False
        if self._baseline_area is None:
            return True
        ratio = area / max(self._baseline_area, 1.0)
        return ratio >= self.visible_ratio_th

    def update(self, frame_idx: int, mask01: np.ndarray, obj_id: int) -> None:
        area = mask_area(mask01)
        self._frame_count += 1
        visible = self._is_visible(area)

        if visible:
            self._last_visible_frame = frame_idx
            if self._baseline_area is None:
                if self._frame_count >= self.warmup_frames:
                    self._baseline_area = float(area)
            else:
                self._baseline_area = (1.0 - self.ema_alpha) * self._baseline_area + self.ema_alpha * float(area)

        if not self._occluded:
            if visible:
                self._not_visible_streak = 0
            else:
                self._not_visible_streak += 1
                if self._not_visible_streak >= self.start_patience:
                    self._occluded = True
                    self._current_start = frame_idx - self.start_patience + 1
                    self._visible_streak = 0
        else:
            if visible:
                self._visible_streak += 1
                if self._visible_streak >= self.end_patience:
                    end_frame = frame_idx - self.end_patience
                    start_frame = int(self._current_start) if self._current_start is not None else end_frame
                    pre_frame = (start_frame - 1) if start_frame > 0 else 0
                    if self._last_visible_frame is not None and self._last_visible_frame < start_frame:
                        pre_frame = self._last_visible_frame
                    post_frame = frame_idx - self.end_patience + 1
                    self._pending_events.append(OcclusionEvent(
                        obj_id=obj_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        pre_frame=pre_frame,
                        post_frame=post_frame
                    ))
                    self._occluded = False
                    self._not_visible_streak = 0
                    self._visible_streak = 0
                    self._current_start = None
            else:
                self._visible_streak = 0

    def pop_events(self) -> List[OcclusionEvent]:
        ev, self._pending_events = self._pending_events, []
        return ev


# -----------------------------
# Candidate selection (obj2)
# -----------------------------
class InteractionCandidateSelector:
    def __init__(self, *, bbox_margin_px: int = 30, min_intersection_px: int = 20, top_k: int = 3):
        self.bbox_margin_px = int(bbox_margin_px)
        self.min_intersection_px = int(min_intersection_px)
        self.top_k = int(top_k)

    def select(self, cls_mask: np.ndarray, obj1: int, other_obj_ids: Iterable[int]) -> List[int]:
        H, W = cls_mask.shape[:2]
        m1 = bin_mask_from_id(cls_mask, obj1)
        b1 = mask_bbox(m1)
        c1 = centroid(m1)

        scores: List[Tuple[float, int]] = []

        if b1 is not None:
            xb1 = bbox_expand(b1, self.bbox_margin_px, W=W, H=H)
            x1, y1, x2, y2 = xb1
            roi = cls_mask[y1:y2, x1:x2]
            for obj2 in other_obj_ids:
                if obj2 == obj1:
                    continue
                inter = int((roi == obj2).sum())
                if inter >= self.min_intersection_px:
                    scores.append((float(inter), obj2))

        if not scores and c1 is not None:
            for obj2 in other_obj_ids:
                if obj2 == obj1:
                    continue
                m2 = bin_mask_from_id(cls_mask, obj2)
                c2 = centroid(m2)
                if c2 is None:
                    continue
                dist = l2(c1, c2)
                scores.append((-dist, obj2))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [obj for _, obj in scores[: self.top_k]]


# -----------------------------
# VLM helper (TubeletGraph-like)
# -----------------------------
def encode_image_from_np(image_np: np.ndarray) -> Tuple[str, str]:
    assert image_np.ndim == 3 and image_np.shape[2] in (3, 4)
    img = image_np
    if img.max() <= 1.0:
        img = (img * 255.0)
    img = img.astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"

def image_payload(image_np: np.ndarray, detail: str = "low") -> dict:
    b64, mime = encode_image_from_np(image_np)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}

def text_payload(text: str) -> dict:
    return {"type": "text", "text": text}

def overlay_contours(
    image_rgb: np.ndarray,
    mask1: np.ndarray,
    mask2: Optional[np.ndarray] = None,
    *,
    color1_rgb: Tuple[int,int,int] = (255, 0, 0),
    color2_rgb: Tuple[int,int,int] = (0, 128, 255),
    thickness: int = 3,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for overlay_contours().")
    img = image_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    def _draw(mask01: np.ndarray, color_rgb: Tuple[int,int,int]):
        m = (mask01.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.drawContours(img, contours, -1, color_bgr, thickness)

    _draw(mask1, color1_rgb)
    if mask2 is not None:
        _draw(mask2, color2_rgb)
    return img


class OpenAIVLMAnnotator:
    def __init__(self, *, model: str, temperature: float = 0.0, max_retries: int = 3, sleep_between_retries: float = 1.0):
        import openai
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.sleep_between_retries = float(sleep_between_retries)
        self.client = openai.OpenAI()

    @staticmethod
    def system_prompt() -> str:
        return "You are a highly intelligent assistant that can analyze videos and images."

    @staticmethod
    def user_prompt(obj1_id: int, obj2_id: int, start_frame: int, end_frame: int) -> str:
        return (
            f"We are tracking objects in a video.\n"
            f"- Red contour: object {obj1_id}\n"
            f"- Blue contour: object {obj2_id}\n\n"
            f"The first image is the last clear frame before object {obj1_id} becomes occluded.\n"
            f"The second image is the first clear frame after the occlusion interval.\n"
            f"The occlusion interval is frames [{start_frame}, {end_frame}] (inclusive).\n\n"
            f"Task: infer what likely happened between these two objects during the interval.\n"
            f"Return ONLY valid JSON with keys:\n"
            f'  "description": a short sentence (<= 20 words) describing the interaction,\n'
            f'  "action": a short verb phrase describing the interaction (e.g., "covers", "moves behind", "picks up").\n'
        )

    def _parse_json(self, text: str) -> Tuple[str, dict]:
        s = text.strip()
        s = s.replace("```json", "```").replace("```JSON", "```")
        if "```" in s:
            lines = [ln for ln in s.splitlines() if "```" not in ln]
            s = "\n".join(lines).strip()
        try:
            data = json.loads(s)
            return s, data
        except Exception:
            l = s.find("{")
            r = s.rfind("}")
            if l != -1 and r != -1 and r > l:
                sub = s[l:r+1]
                try:
                    data = json.loads(sub)
                    return sub, data
                except Exception:
                    pass
        return s, {"description": s, "action": ""}

    def describe_interaction(
        self,
        *,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        pre_mask_obj1: np.ndarray,
        pre_mask_obj2: np.ndarray,
        post_mask_obj1: np.ndarray,
        post_mask_obj2: np.ndarray,
        obj1_id: int,
        obj2_id: int,
        start_frame: int,
        end_frame: int,
        image_detail: str = "low",
    ) -> Tuple[str, dict]:
        pre_vis = overlay_contours(pre_image, pre_mask_obj1, pre_mask_obj2)
        post_vis = overlay_contours(post_image, post_mask_obj1, post_mask_obj2)

        content = [
            text_payload(self.user_prompt(obj1_id, obj2_id, start_frame, end_frame)),
            text_payload("Image 1 (pre-occlusion):"),
            image_payload(pre_vis, detail=image_detail),
            text_payload("Image 2 (post-occlusion):"),
            image_payload(post_vis, detail=image_detail),
        ]

        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self.system_prompt()},
                        {"role": "user", "content": content},
                    ],
                )
                text = rsp.choices[0].message.content
                _, data = self._parse_json(text)
                return text, data
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(self.sleep_between_retries)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}") from last_err


# -----------------------------
# Pipeline glue
# -----------------------------
class CutieStateChangePipeline:
    def __init__(
        self,
        *,
        processor: Any,  # InferenceCore
        obj1_id: int,
        vlm: Optional[OpenAIVLMAnnotator] = None,
        occlusion: Optional[OcclusionDetector] = None,
        selector: Optional[InteractionCandidateSelector] = None,
        mask_saver: Optional[Any] = None,  # Cutie's ResultSaver
        save_mask_every: int = 1,
    ):
        self.processor = processor
        self.obj1_id = int(obj1_id)
        self.vlm = vlm
        self.occlusion = occlusion or OcclusionDetector()
        self.selector = selector or InteractionCandidateSelector()

        self.mask_saver = mask_saver
        self.save_mask_every = int(save_mask_every)

        self.images: List[np.ndarray] = []
        self.cls_masks: List[np.ndarray] = []

    def _remap_tmp_to_obj(self, mask_tmp: np.ndarray) -> np.ndarray:
        om = getattr(self.processor, "object_manager", None)
        tmp_map = getattr(om, "tmp_id_to_obj", None)
        if not tmp_map:
            return mask_tmp
        out = np.zeros_like(mask_tmp, dtype=np.int32)
        for tmp_id, obj in tmp_map.items():
            out[mask_tmp == int(tmp_id)] = int(obj.id)
        return out

    def feed_frame(
        self,
        frame_idx: int,
        frame_rgb: np.ndarray,
        image_tensor,  # torch 3xHxW float
        *,
        init_mask_tensor=None,  # torch HxW long
        init_objects: Optional[List[int]] = None,
    ) -> None:
        self.images.append(frame_rgb.astype(np.uint8))

        if init_mask_tensor is not None:
            output_prob = self.processor.step(image_tensor, init_mask_tensor, objects=init_objects)
        else:
            output_prob = self.processor.step(image_tensor)

        # Save masks via ResultSaver (same as process_video.py)
        if self.mask_saver is not None and (frame_idx % max(self.save_mask_every, 1) == 0):
            frame_name = f"{frame_idx:07d}.png"
            self.mask_saver.process(
                output_prob,
                frame_name,
                resize_needed=False,
                shape=None,
                last_frame=False,
                path_to_image=None,
            )

        # For occlusion reasoning, use OBJECT IDs (not tmp IDs)
        mask_tmp = torch.argmax(output_prob, dim=0).detach().cpu().numpy().astype(np.int32)
        pred_cls_mask = self._remap_tmp_to_obj(mask_tmp)

        self.cls_masks.append(pred_cls_mask)
        m1 = bin_mask_from_id(pred_cls_mask, self.obj1_id)
        self.occlusion.update(frame_idx, m1, obj_id=self.obj1_id)

    def pop_state_changes(self, *, image_detail: str = "low") -> List[StateChange]:
        events = self.occlusion.pop_events()
        out: List[StateChange] = []

        if not events:
            return out

        if self.vlm is None:
            for ev in events:
                out.append(StateChange(
                    start_frame=ev.start_frame,
                    end_frame=ev.end_frame,
                    obj1=ev.obj_id,
                    obj2=-1,
                    description="occluded",
                    meta={"note": "No VLM annotator configured."}
                ))
            return out

        for ev in events:
            if ev.pre_frame < 0 or ev.post_frame >= len(self.images):
                continue

            pre_img = self.images[ev.pre_frame]
            post_img = self.images[ev.post_frame]
            pre_cls = self.cls_masks[ev.pre_frame]
            post_cls = self.cls_masks[ev.post_frame]

            other_ids = sorted({int(x) for x in np.unique(pre_cls) if int(x) not in (0, ev.obj_id)})
            candidates = self.selector.select(pre_cls, ev.obj_id, other_ids)

            if not candidates:
                out.append(StateChange(
                    start_frame=ev.start_frame,
                    end_frame=ev.end_frame,
                    obj1=ev.obj_id,
                    obj2=-1,
                    description="occlusion (no nearby tracked obj2 found)",
                    meta={"event": ev.__dict__}
                ))
                continue

            for obj2 in candidates:
                pre_m1 = bin_mask_from_id(pre_cls, ev.obj_id)
                pre_m2 = bin_mask_from_id(pre_cls, obj2)
                post_m1 = bin_mask_from_id(post_cls, ev.obj_id)
                post_m2 = bin_mask_from_id(post_cls, obj2)

                raw_text, parsed = self.vlm.describe_interaction(
                    pre_image=pre_img,
                    post_image=post_img,
                    pre_mask_obj1=pre_m1,
                    pre_mask_obj2=pre_m2,
                    post_mask_obj1=post_m1,
                    post_mask_obj2=post_m2,
                    obj1_id=ev.obj_id,
                    obj2_id=obj2,
                    start_frame=ev.start_frame,
                    end_frame=ev.end_frame,
                    image_detail=image_detail,
                )

                desc = parsed.get("description", raw_text)
                out.append(StateChange(
                    start_frame=ev.start_frame,
                    end_frame=ev.end_frame,
                    obj1=ev.obj_id,
                    obj2=obj2,
                    description=str(desc).strip(),
                    meta={"vlm_raw": raw_text, "vlm_parsed": parsed, "event": ev.__dict__}
                ))

        return out


# -----------------------------
# IO helpers (frames/video)
# -----------------------------
def iter_frames_from_dir(frame_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")):
    files = [f for f in os.listdir(frame_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()
    for fn in files:
        path = os.path.join(frame_dir, fn)
        img = np.array(Image.open(path).convert("RGB"))
        yield img

def iter_frames_from_video(video_path: str):
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for iter_frames_from_video().")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = frame_bgr[..., ::-1].copy()
        yield frame_rgb
    cap.release()

def save_state_changes_json(state_changes: List[StateChange], out_path: str) -> None:
    data = [
        {
            "start_frame": sc.start_frame,
            "end_frame": sc.end_frame,
            "obj1": sc.obj1,
            "obj2": sc.obj2,
            "description": sc.description,
            "meta": sc.meta,
        }
        for sc in state_changes
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
