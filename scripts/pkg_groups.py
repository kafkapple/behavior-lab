"""Environment groups of the SLEAP 1423/1500 base labels, defined once from the source-video path.

Rule (first match wins): the path encodes lab, recording session and camera; resolution is a
consequence of the session, so it is not part of the key (checked in pkg_overlay_sheets --by-group).
"""
import re

GROUPS = [  # (key, description, regex on the source-video path, "\" normalised to "/")
    ("avatar_bot", "AVATAR rig, bottom camera (Baseline Ptd005 Fit)", r"AVATAR_split/Baseline.*_bot\.mp4$"),
    ("avatar_side", "AVATAR rig, side1-4 cameras (Baseline Ptd005 Fit)", r"AVATAR_split/Baseline.*_side\d\.mp4$"),
    ("avatar_sci1", "AVATAR rig, 260505 Sci 1 (bot + side1)", r"AVATAR_split/260505 Sci 1_"),
    ("jhl_pted", "JHL cylinder, PTED (vertical video)", r"/JHL/"),
    ("khu_2025_01", "KHU 2025-01 elec_acupuncture (bandicam)", r"/KHU/20250206/elec_acupuncture/"),
    ("khu_20250311", "KHU 20250311", r"/KHU/20250311/"),
    ("khu_20250721_20260213", "KHU 20250721 + 20260213", r"/KHU/(20250721|20260213)/"),
]
KEYS = [g[0] for g in GROUPS]


def group_of(path: str) -> str:
    p = str(path).replace("\\", "/")
    for key, _, rx in GROUPS:
        if re.search(rx, p):
            return key
    raise ValueError(f"no environment group for {path!r}")


# One keypoint palette for every figure and note. Family = anatomy (warm head/neck, blue forelimb,
# green hindlimb, purple tail); lightness = side (light = Left, dark = Right). The SUBTLE viewer palette
# is not used: nose/ears share yellow and the 3 tail points share orange.
KP_COLORS = {
    "nose1": "#E63946", "neck1": "#FFB000", "earL1": "#FF6FB5", "earR1": "#B5179E",
    "forelegL1": "#4EA8DE", "forelegR1": "#1D4ED8", "hindlegL1": "#6BCB77", "hindlegR1": "#1B7F3B",
    "tailstart1": "#C77DFF", "tail1": "#7B2CBF", "tailend1": "#3C096C",
}
# Marker radius relative to the tile width (4.5 px at a 640 px tile, 3.4 px at 480), shared by every overlay image.
MARKER_R_REL = 4.5 / 640
KP_FAMILY = {"nose1": "head", "neck1": "head", "earL1": "head", "earR1": "head",
             "forelegL1": "forelimb", "forelegR1": "forelimb", "hindlegL1": "hindlimb",
             "hindlegR1": "hindlimb", "tailstart1": "tail", "tail1": "tail", "tailend1": "tail"}


def check_palette(min_dist: float = 60.0) -> float:
    """Assert the 11 colours are pairwise distinct (RGB distance > min_dist); return the minimum."""
    rgb = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in KP_COLORS.items()}
    assert len(rgb) == 11 and set(rgb) == set(KP_FAMILY), sorted(rgb)
    d = min((sum((a - b) ** 2 for a, b in zip(rgb[x], rgb[y])) ** 0.5, x, y)
            for i, x in enumerate(rgb) for y in list(rgb)[i + 1:])
    assert d[0] > min_dist, d
    return d[0]
