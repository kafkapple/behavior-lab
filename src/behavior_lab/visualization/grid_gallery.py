"""Animation pages of the grid: per-cluster GIF gallery and the synchronized playback page."""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np

from ._grid_common import (
    FAMILIES,
    GALLERY_CLUSTERS,
    GALLERY_GIF_FPS,
    GALLERY_MAX_SEC,
    GALLERY_PAD_SEC,
    _family,
    _keypoints,
    _labels,
    _method_color,
    _skeleton,
    _stem,
    _subtle_map,
    _write,
)
from .agreement import stretch_labels
from .cluster_map import pose_embedding
from .grid_slice import MATCH_P, slice_matches
from .player import PLAYER_JS, player_block, player_data

GALLERY_MATCHES = 6


def _bout_clip(mask: np.ndarray, fps: float) -> tuple[int, int, int]:
    """Frame range of the median-length run of ``mask`` plus padding: ``(start, end, bout_len)``.

    One real bout, never several stitched together; the median avoids showing the longest
    (often a resting or tracking-loss tail)."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    order = np.argsort(ends - starts)
    k = order[len(order) // 2]
    pad = int(GALLERY_PAD_SEC * fps)
    length = min(int(ends[k] - starts[k]), int(GALLERY_MAX_SEC * fps))
    return (max(0, int(starts[k]) - pad), min(len(mask), int(starts[k]) + length + pad),
            int(ends[k] - starts[k]))


def render_gallery(batch_dir: str | Path, out_html: str | Path | None = None) -> Path:
    """Per-method, per-cluster skeleton GIFs on the representative slices (those with a map)."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .html_report import image_to_base64
    from .skeleton import animate_skeleton

    def clip(mask, path, caption, title_):
        s, e, bout = _bout_clip(mask, fps)
        hop = max(1, round(fps / GALLERY_GIF_FPS))
        animate_skeleton(kp[s:e:hop], skeleton=skel, fps=fps / hop, figsize=(3, 3),
                         title=title_, save_path=str(path))
        plt.close("all")
        return ('<figure style="display:inline-block;margin:4px;text-align:center">'
                f'<img loading="lazy" src="{image_to_base64(path)}" style="max-width:190px">'
                f"<figcaption>{caption}<br>bout {bout / fps:.1f} s · frames {s}–{e}"
                "</figcaption></figure>")

    batch_dir = Path(batch_dir)
    rows = json.loads((batch_dir / "batch_results.json").read_text())
    slices = {s["name"]: s for s in json.loads((batch_dir / "dataset_slices.json").read_text())}
    methods = sorted({r["method"] for r in rows})
    color = {_stem(m): c for m, c in _method_color(methods).items()}
    chosen = [d for d in slices if (batch_dir / "arrays" / d / "keypoints.npy").exists()
              and _subtle_map(batch_dir, d)]

    title = f"Cluster gallery: {batch_dir.name}"
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<div class="header-meta">{date.today().isoformat()} · slices: '
             f'{html.escape(", ".join(chosen))} · source: <code>{html.escape(str(batch_dir))}'
             "</code></div>",
             '<div class="tldr"><ul><li>Each animation is one real bout of that cluster: the '
             f"bout of median length, at most {GALLERY_MAX_SEC:g} s, with {GALLERY_PAD_SEC:g} s of "
             f"context before and after, played at about {GALLERY_GIF_FPS} fps. Nothing is "
             "stitched.</li>"
             f"<li>Per method the {GALLERY_CLUSTERS} clusters with the most frames are shown; "
             "the heading says how many clusters exist and what share of frames is covered.</li>"
             "<li>Cluster ids are per method: cluster 3 of one method is unrelated to cluster 3 "
             "of another. The colored bar is the method color used in the result grid.</li>"
             "</ul></div>"]
    with tempfile.TemporaryDirectory() as tmp:
        for ds in chosen:
            kp = np.load(batch_dir / "arrays" / ds / "keypoints.npy")
            fps = slices[ds]["fps"]
            skel, skel_note = _skeleton(ds, slices[ds]["notes"].get("node_names"), kp.shape[1])
            parts.append(f"<h2>{html.escape(ds)}</h2><p>{kp.shape[0]} frames at {fps:g} fps; "
                         f"{html.escape(skel_note)}.</p>")
            for m, seq in _labels(batch_dir, ds).items():
                lab = stretch_labels(seq, len(kp))
                ids, counts = np.unique(lab[lab >= 0], return_counts=True)
                keep = ids[np.argsort(-counts)][:GALLERY_CLUSTERS]
                share = counts[np.argsort(-counts)][:GALLERY_CLUSTERS].sum() / len(lab)
                cards = []
                for cid in keep:
                    cards.append(clip(lab == cid, Path(tmp) / f"{ds}_{m}_{cid}.gif",
                                      f"cluster {cid} · {(lab == cid).mean():.0%} of frames",
                                      f"cluster {cid}"))
                parts.append(f'<h3 style="border-left:6px solid {color[m]};padding-left:8px">'
                             f"{html.escape(m)}</h3><p>{len(keep)} of {len(ids)} clusters, "
                             f"{share:.0%} of frames.</p><div>{''.join(cards)}</div>")
            seqs = {m: stretch_labels(v, len(kp)) for m, v in _labels(batch_dir, ds).items()}
            best = sorted(((d, a, b) for (a, b), mm in slice_matches(seqs).items()
                           for d in mm["pairs"] if d["p"] < MATCH_P and not d["largest"]),
                          key=lambda x: -x[0]["jaccard"])[:GALLERY_MATCHES]
            cards = [clip((seqs[a] == d["a"]) & (seqs[b] == d["b"]),
                          Path(tmp) / f"{ds}_match{i}.gif",
                          f"{a} {d['a']} = {b} {d['b']}<br>Jaccard {d['jaccard']:.2f} "
                          f"(expected {d['expected']:.2f})", f"{d['a']} = {d['b']}")
                     for i, (d, a, b) in enumerate(best)]
            parts.append("<h3>Matched across methods</h3><p>Frames that two methods both assign "
                         "to one of their paired clusters (pairs with p &lt; 0.05, pairs of each "
                         f"method's largest cluster left out), top {GALLERY_MATCHES} by Jaccard."
                         f"</p><div>{''.join(cards) or 'No pair clears the null.'}</div>")
    return _write(Path(out_html) if out_html else batch_dir / "batch_gallery.html", title, parts)


def render_player(batch_dir: str | Path, out_html: str | Path | None = None) -> Path:
    """Playback page: one block per slice with keypoints; data is parsed when a block opens."""
    batch_dir = Path(batch_dir)
    slices = json.loads((batch_dir / "dataset_slices.json").read_text())
    title = f"Playback: {batch_dir.name}"
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<div class="header-meta">{date.today().isoformat()} · source: <code>'
             f"{html.escape(str(batch_dir))}</code></div>",
             '<div class="tldr"><ul><li>Left: 3D skeleton; drag to rotate, "follow animal" '
             "keeps it centered. Joint colors are those of the keypoint layout table.</li>"
             "<li>Right: every time step as a point on a 2D PCA of posture, colored by the "
             "chosen method's clusters. The same points serve every method; only the coloring "
             "changes. The ring is the current step, fading dots are the last 2 s.</li>"
             "<li>Legend: click a cluster to switch it off or on. Switched-off clusters fade "
             "on the map and in that method's row, and playback skips them. Every skip shows "
             "a red “cut” mark and resets the trail: the two sides of a cut are not "
             "continuous motion.</li>"
             "<li>Bottom: every method's label sequence with a cursor; click a row to jump. "
             "Colors are per method (12 largest clusters colored, the rest grey): the same "
             "color in two rows is not the same behavior.</li>"
             "<li>Playback is at 10 Hz; pose, map position and labels are all taken from the "
             "same frame.</li></ul></div>"]
    family, first = None, True
    for s in slices:
        ds, kp = s["name"], _keypoints(batch_dir, s["name"])
        seqs = _labels(batch_dir, ds)
        if kp is None or not seqs:
            continue
        if _family(ds) != family:
            family = _family(ds)
            parts.append(f"<h2>{html.escape(FAMILIES.get(family, family))}</h2>")
        skel, _ = _skeleton(ds, s["notes"].get("node_names"), kp.shape[1])
        data = player_data(kp, skel.edges, pose_embedding(kp), seqs, s["fps"])
        parts.append(player_block(f"player-{ds}", f"{ds} ({len(kp)} frames, {len(seqs)} methods)",
                                  data, open_=first))
        first = False
    parts.append(PLAYER_JS)
    return _write(Path(out_html) if out_html else batch_dir / "batch_player.html", title, parts)
