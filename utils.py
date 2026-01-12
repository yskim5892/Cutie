import numpy as np
from PIL import Image
from pathlib import Path

class PathManager:
    def __init__(self, name):
        self.name = name
        self.path = Path.cwd() / 'dataset' / name 

    def get_video_path(self):
        return self.path / (self.name + '.mp4')

    def get_frames_dir(self):
        return self.path / 'frames'

    def get_input_mask_dir(self):
        return self.path / 'input_mask'

    def get_output_mask_dir(self):
        return self.path / 'output_mask'

    def get_output_video_path(self):
        return self.path / (self.name + '_output.mp4')

    def get_output_state_changes_path(self):
        return self.path / (self.name + '_state_changes.json')

def decode_rgb_bit128_mask_to_ids_and_palette(mask_pil):
    """
    RGB mask where each channel is either 0 or 128 encodes IDs as:
      id = (R>=128)*1 + (G>=128)*2 + (B>=128)*4
    Also builds a palette so that output colors exactly match input colors.
    """
    import numpy as np

    rgb = np.array(mask_pil, dtype=np.uint8)  # (H,W,3)
    uniq = np.unique(rgb.reshape(-1, 3), axis=0)

    # validate the (0/128) scheme
    if not np.all(np.isin(uniq, [0, 128])):
        raise ValueError(
            f"init_mask RGB detected but not (0/128) bit-encoded. unique colors (first 20)={uniq[:20]}"
        )

    id_mask = (
        (rgb[..., 0] >= 128).astype(np.uint8) * 1
        + (rgb[..., 1] >= 128).astype(np.uint8) * 2
        + (rgb[..., 2] >= 128).astype(np.uint8) * 4
    )

    # palette: 256 colors * 3
    palette = [0] * (256 * 3)

    # Fill palette entries for IDs that appear, using the exact original RGB triplets.
    for c in uniq:
        rid = (1 if c[0] >= 128 else 0) + (2 if c[1] >= 128 else 0) + (4 if c[2] >= 128 else 0)
        palette[3 * rid + 0] = int(c[0])
        palette[3 * rid + 1] = int(c[1])
        palette[3 * rid + 2] = int(c[2])

    return id_mask, palette


def rgb_to_p(mask: Image.Image, bg_rgb=(0, 0, 0), colors=3):
    """
    Returns:
      mask_id: (H,W) uint8 with values {0..K} where 0 is background
      pal: palette if input was 'RGB' or 'P'
      K: number of objects
    """
    if mask.mode == "RGB":
        rgb = np.array(mask, dtype=np.uint8)
        uniq = np.unique(rgb.reshape(-1, 3), axis=0)
        bg = np.array(bg_rgb, dtype=np.int32)
        d = np.sum((uniq.astype(np.int32) - bg) ** 2, axis=1)
        bg_idx = int(np.argmin(d))
        bg_color = uniq[bg_idx]

        uniq_nonbg = [u for i, u in enumerate(uniq.tolist()) if i != bg_idx]
        out = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for new_id, color in enumerate(uniq_nonbg, start=1):
            color_arr = np.array(color, dtype=np.uint8)
            out[np.all(rgb == color_arr, axis=-1)] = new_id

        pal = [0] * (256 * 3)
        pal[0:3] = bg_color.tolist()
        for new_id, color in enumerate(uniq_nonbg, start=1):
            pal[3 * new_id:3 * new_id + 3] = color

        return out, pal, len(uniq_nonbg)

    if mask.mode == "P":
        pal = mask.getpalette()  # length 768 = 256*3
        pal_rgb = np.array(pal, dtype=np.int32).reshape(-1, 3)  # (256,3)
        mask_np = np.array(mask, dtype=np.uint8)

        uniq = np.unique(mask_np)
        bg = np.array(bg_rgb, dtype=np.int32)

        # choose background index where it is closest to black in palette
        d = np.sum((pal_rgb[uniq] - bg) ** 2, axis=1)
        bg_idx = int(uniq[int(np.argmin(d))])

        # Remap bg_idx -> 0, others -> 1..K
        uniq_nonbg = [u for u in uniq.tolist() if u != bg_idx]
        out = np.zeros_like(mask_np, dtype=np.uint8)
        for new_id, old_id in enumerate(uniq_nonbg, start=1):
            out[mask_np == old_id] = new_id

        new_pal = [0] * (256 * 3)
        new_pal[0:3] = pal_rgb[bg_idx].tolist()
        for new_id, old_id in enumerate(uniq_nonbg, start=1):
            new_pal[3 * new_id:3 * new_id + 3] = pal_rgb[old_id].tolist()

        return out, new_pal, len(uniq_nonbg)

    elif mask.mode == "L":
        mask_np = np.array(mask, dtype=np.uint8)
        uniq = np.unique(mask_np)
        bg_idx = 0 if 0 in uniq else int(uniq[0])
        uniq_nonbg = [u for u in uniq.tolist() if u != bg_idx]
        out = np.zeros_like(mask_np, dtype=np.uint8)
        for new_id, old_id in enumerate(uniq_nonbg, start=1):
            out[mask_np == old_id] = new_id
        return out, None, len(uniq_nonbg)

    else:
        raise ValueError(f"Unsupported mask mode: {mask.mode}")
