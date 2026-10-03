"""Shared shell of the self-contained keypoint report pages (one palette, light and dark)."""
from __future__ import annotations

import json

CSS = """:root{--bg:#fff;--fg:#1c1f23;--mut:#5b6570;--line:#d6dbe0;--card:#f5f7f9;--hi:#cfe3ff}
@media (prefers-color-scheme:dark){:root{--bg:#16191d;--fg:#e6e9ec;--mut:#9aa4ae;--line:#343a41;--card:#1e2328;--hi:#1d3b66}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:22px}h2{font-size:17px;margin:28px 0 4px}.lead{margin:0 0 8px}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}.c{text-align:right;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px}code{background:var(--card);padding:0 4px;border-radius:3px}
select,input{font:inherit}label{margin-right:12px;white-space:nowrap}.vw{display:flex;gap:12px;align-items:flex-start}
@media(max-width:800px){.vw{flex-direction:column}}"""


def page(title: str, body: str, script: str, data: dict, extra_css: str = "") -> str:
    """One HTML file: `script` reads the global `D` (the JSON-embedded `data`)."""
    js = script.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{title}</title><style>{CSS}{extra_css}</style></head><body><main>{body}</main><script>{js}</script></body></html>")
