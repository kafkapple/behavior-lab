"""Overlay contact sheets of the SLEAP 1423/1500 base training labels, one sheet per source video.

    python -I scripts/pkg_overlay_sheets.py <pkg_images dir> [--per-sheet 16]
    python scripts/pkg_overlay_sheets.py <pkg_images dir> --by-group [--per-sheet 12] [--palette keypoint_palette.json]

--by-group: one larger sheet per environment group (scripts/pkg_groups.py) -> <dir>/overlay_groups/<NN>_<group>.jpg,
3 columns x 640 px tiles, viewer-palette filled dots with dark outline, legend bar. Counts per group are printed
(they must sum to the number of frames in the CSV).

Reads <dir>/labels_long.csv (from extract_pkg_images.py) and <dir>/images; writes <dir>/overlay_sheets/<NN>_<name>.jpg.
Frames are evenly spaced within each video, so a sheet shows the video's labels, not a hand-picked subset.
Colour = keypoint (legend on each sheet), tile caption = split / frame. Missing points (NaN) are not drawn.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under python -I too

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pkg_groups import GROUPS, KEYS, group_of

COLS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
        (70, 240, 240), (240, 50, 230), (210, 245, 60), (128, 128, 0), (255, 255, 255)]
TILE_W = 480
G_TILE_W, G_COLS = 640, 3
PALETTE = {  # viewer anatomical colours (tools/subtle_viewer/web/keypoint_palette.json)
    "nose1": "#ffff00", "neck1": "#ff00ff", "earL1": "#ffff00", "earR1": "#ffff00",
    "forelegL1": "#0000ff", "forelegR1": "#00ff00", "tailstart1": "#ffa500",
    "hindlegL1": "#00ffff", "hindlegR1": "#ff0000", "tail1": "#ffa500", "tailend1": "#ffa500",
}


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


def group_sheet(g: pd.DataFrame, root: Path, kps: list[str], n: int, title: str, pal: dict) -> tuple[Image.Image, int, int]:
    """One sheet for an environment group; asserts every drawn point equals the CSV coordinate."""
    frames = g.drop_duplicates("image").sort_values(["source_video", "image"])
    pick = frames.iloc[np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int))]
    font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=16)
    tiles, drawn, expect = [], 0, 0
    for _, f in pick.iterrows():
        img = Image.open(root / f.image).convert("RGB")
        s = G_TILE_W / img.width
        img = img.resize((G_TILE_W, round(img.height * s)))
        dr = ImageDraw.Draw(img)
        pts = g[(g.image == f.image) & g.x_px.notna()]
        seen = {}
        for _, p in pts.iterrows():
            x, y = p.x_px * s, p.y_px * s
            dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=pal[p.keypoint], outline=(20, 20, 20), width=2)
            seen[p.keypoint] = (x, y)
        assert set(seen) == set(pts.keypoint) and all(
            np.allclose(seen[r.keypoint], (r.x_px * s, r.y_px * s)) for r in pts.itertuples()), f.image
        drawn += len(seen)
        expect += int(g[g.image == f.image].x_px.notna().sum())
        dr.rectangle([0, 0, G_TILE_W, 22], fill=(0, 0, 0))
        dr.text((4, 2), f"{f.split} {Path(f.image).stem}  {Path(f.source_video).stem[-14:]}  {img.width / s:.0f}x{img.height / s:.0f}", fill=(255, 255, 255), font=small)
        tiles.append(img)
    rows = -(-len(tiles) // G_COLS)
    h = max(t.height for t in tiles)
    bar = 64
    out = Image.new("RGB", (G_COLS * G_TILE_W, bar + rows * h), (30, 30, 30))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % G_COLS) * G_TILE_W, bar + (i // G_COLS) * h))
    dr = ImageDraw.Draw(out)
    dr.text((8, 4), f"{title}   frames {len(frames)} (showing {len(tiles)})", fill=(255, 255, 255), font=font)
    x = 8
    for k in kps:  # legend: filled dot + name
        dr.ellipse([x, 38, x + 14, 52], fill=pal[k], outline=(20, 20, 20), width=2)
        dr.text((x + 20, 36), k, fill=(255, 255, 255), font=small)
        x += 20 + int(dr.textlength(k, font=small)) + 16
    return out, drawn, expect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-sheet", type=int, default=None)
    ap.add_argument("--by-group", action="store_true", help="one sheet per environment group")
    ap.add_argument("--palette", type=Path, help="keypoint_palette.json (name -> [part, hex])")
    a = ap.parse_args()
    d = pd.read_csv(a.root / "labels_long.csv", keep_default_na=False, na_values={"x_px": [""], "y_px": [""]})
    kps = list(dict.fromkeys(d.keypoint))
    assert len(kps) == len(COLS), kps
    if a.by_group:
        pal = dict(PALETTE)
        if a.palette:
            pal.update({k: v[1] for k, v in json.loads(a.palette.read_text()).items()})
        d["group"] = [group_of(v) for v in d.source_video]
        nfr = d.drop_duplicates("image").group.value_counts()
        assert nfr.sum() == d.image.nunique(), (nfr.sum(), d.image.nunique())
        out = a.root / "overlay_groups"
        out.mkdir(exist_ok=True)
        for i, (key, desc, _) in enumerate(GROUPS):
            im, drawn, expect = group_sheet(d[d.group == key], a.root, kps, a.per_sheet or 12, f"{key}: {desc}", pal)
            assert drawn == expect, (key, drawn, expect)
            im.save(out / f"{i + 1:02d}_{key}.jpg", quality=85)
            print(f"{i + 1:02d}_{key}: frames {nfr[key]}, sheet {im.width}x{im.height}")
        print(f"total frames {nfr.sum()} -> {out}")
        return
    out = a.root / "overlay_sheets"
    out.mkdir(exist_ok=True)
    for i, (src, g) in enumerate(d.groupby("source_video", sort=True)):
        im, drawn, expect = sheet(g, a.root, kps, a.per_sheet or 16)
        assert drawn == expect, (src, drawn, expect)
        name = Path(src).parent.name + "_" + Path(src).stem
        im.save(out / f"{i:02d}_{name}.jpg", quality=85)
    print(f"{i + 1} sheets -> {out}; keypoint colours: " + ", ".join(f"{k}={c}" for k, c in zip(kps, COLS)))


if __name__ == "__main__":
    main()
