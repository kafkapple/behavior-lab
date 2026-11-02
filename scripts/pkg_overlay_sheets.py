"""Overlay contact sheets of the SLEAP 1423/1500 base training labels, one sheet per source video.

    python -I scripts/pkg_overlay_sheets.py <pkg_images dir> [--per-sheet 16]

Reads <dir>/labels_long.csv (from extract_pkg_images.py) and <dir>/images; writes <dir>/overlay_sheets/<NN>_<name>.jpg.
Frames are evenly spaced within each video, so a sheet shows the video's labels, not a hand-picked subset.
Colour = keypoint (legend on each sheet), tile caption = split / frame. Missing points (NaN) are not drawn.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

COLS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
        (70, 240, 240), (240, 50, 230), (210, 245, 60), (128, 128, 0), (255, 255, 255)]
TILE_W = 480


def sheet(g: pd.DataFrame, root: Path, kps: list[str], n: int) -> tuple[Image.Image, int, int]:
    frames = g.drop_duplicates("image").sort_values("image")
    pick = frames.iloc[np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int))]
    tiles, drawn, expect = [], 0, 0
    for _, f in pick.iterrows():
        img = Image.open(root / f.image).convert("RGB")
        s = TILE_W / img.width
        img = img.resize((TILE_W, round(img.height * s)))
        dr = ImageDraw.Draw(img)
        pts = g[g.image == f.image]
        expect += int(pts.x_px.notna().sum())
        for _, p in pts.iterrows():
            if pd.notna(p.x_px):
                x, y, c = p.x_px * s, p.y_px * s, COLS[kps.index(p.keypoint)]
                dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=c, width=2)
                drawn += 1
        dr.rectangle([0, 0, TILE_W, 16], fill=(0, 0, 0))
        dr.text((3, 2), f"{f.split} {Path(f.image).stem}", fill=(255, 255, 255))
        tiles.append(img)
    cols = 4
    rows = -(-len(tiles) // cols)
    h = max(t.height for t in tiles)
    out = Image.new("RGB", (cols * TILE_W, rows * h + 18), (30, 30, 30))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * TILE_W, 18 + (i // cols) * h))
    dr = ImageDraw.Draw(out)
    head = Path(g.source_video.iloc[0]).name + f"  frames {len(frames)}  "
    dr.text((3, 3), head, fill=(255, 255, 255))
    x = 3 + 6 * len(head)
    for k, c in zip(kps, COLS):
        dr.text((x, 3), k, fill=c)
        x += 6 * len(k) + 8
    return out, drawn, expect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-sheet", type=int, default=16)
    a = ap.parse_args()
    d = pd.read_csv(a.root / "labels_long.csv", keep_default_na=False, na_values={"x_px": [""], "y_px": [""]})
    kps = list(dict.fromkeys(d.keypoint))
    assert len(kps) == len(COLS), kps
    out = a.root / "overlay_sheets"
    out.mkdir(exist_ok=True)
    for i, (src, g) in enumerate(d.groupby("source_video", sort=True)):
        im, drawn, expect = sheet(g, a.root, kps, a.per_sheet)
        assert drawn == expect, (src, drawn, expect)
        name = Path(src).parent.name + "_" + Path(src).stem
        im.save(out / f"{i:02d}_{name}.jpg", quality=85)
    print(f"{i + 1} sheets -> {out}; keypoint colours: " + ", ".join(f"{k}={c}" for k, c in zip(kps, COLS)))


if __name__ == "__main__":
    main()
