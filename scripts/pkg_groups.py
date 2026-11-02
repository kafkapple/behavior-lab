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
