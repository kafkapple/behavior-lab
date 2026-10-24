"""SUBTLE Lite Viewer server (stdlib only).  usage: server.py [root_dir] [--port 8777] [--no-open]"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import mimetypes
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402
import dek  # noqa: E402
import model  # noqa: E402
import quality  # noqa: E402
import results  # noqa: E402
import stability  # noqa: E402
import sweep  # noqa: E402
import stats  # noqa: E402

HERE = Path(__file__).parent
LOGBUF: collections.deque = collections.deque(maxlen=300)  # debug panel reads this; bounded, no leak


class _Buf(logging.Handler):
    def emit(self, r):
        LOGBUF.append(self.format(r))


def _routes(root: Path):
    def session(q):
        s = core.load(root, q["path"][0])
        stride = max(1, int(q.get("stride", ["1"])[0]))
        xyz = s["xyz"][::stride]
        pick = lambda k: None if s[k] is None else s[k][::stride]  # noqa: E731
        return {"nodes": s["nodes"], "edges": s["edges"], "stride": stride, "n": int(len(s["xyz"])),
                "xyz": [[[round(float(v), 1) for v in j] for j in f] for f in xyz.tolist()],
                "sub": pick("sub"), "super": pick("super"), "emb": pick("emb"), "fit_id": s["fit_id"]}

    def kp2d(q):
        s, cam = core.load(root, q["path"][0]), int(q["cam"][0])
        if s["kp"] is None:
            raise ValueError("no 2D keypoints in this file")
        return {"cam": cam, "kp": np.round(core.kp_cell(s, cam), 4).tolist()}

    docs = HERE.parents[1] / "docs" / "dev"  # manuals shown in the page: docs/dev/subtle_*.md

    def doc(q):
        name = q["name"][0]
        if not re.fullmatch(r"subtle_[\w\-]+\.md", name) or not (docs / name).is_file():
            raise ValueError(f"unknown doc {name}")
        return {"name": name, "text": (docs / name).read_text(encoding="utf-8")}

    def camview(q):
        s = core.load(root, q["path"][0])
        if s["kp"] is None:
            raise ValueError("no 2D keypoints in this file")
        return core.cam_view(s, int(q["cam"][0]))

    def proj_of(q):
        return core.load_project(root, q["name"][0])

    sl = lambda q: int(q["sl"][0]) if "sl" in q else None  # noqa: E731
    return {
        "camview": lambda q, b: camview(q),
        "stats": lambda q, b: stats.group_stats(root, proj_of(q), q.get("level", ["super"])[0], sl(q), q.get("correct", ["1"])[0] == "1"),
        "dek": lambda q, b: dek.group_dek(root, proj_of(q), q.get("level", ["super"])[0], sl(q), q.get("correct", ["1"])[0] == "1"),
        "names": lambda q, b: stats.names_all(root),
        "set_name": lambda q, b: stats.names_set(root, b["key"], b["name"]),
        "stability": lambda q, b: {"job": stability.start(root, b["name"], int(b.get("max_frames", 3000)), [int(x) for x in b.get("seeds", [0, 1, 2])], b.get("clustering"))},
        "stability_result": lambda q, b: json.loads(stability.result_path(root, q["name"][0]).read_text()),
        "sweep": lambda q, b: sweep.project_sweep(root, proj_of(q), q.get("criterion", ["omnibus"])[0], q.get("excl", ["0"])[0] == "1", tuple(filter(None, q.get("drop", [""])[0].split(",")))),
        "quality": lambda q, b: quality.project_quality(root, proj_of(q)),
        "export": lambda q, b: {"job": results.start(root, b["name"], b.get("criterion", "omnibus"))},
        "export_latest": lambda q, b: {"dir": (core.results_dir(root, q["name"][0]) / "latest.txt").read_text()},
        "models": lambda q, b: model.list_models(root),
        "train": lambda q, b: {"job": model.start_train(root, b["name"], b["paths"], int(b.get("max_frames", 0)), b.get("seed"), b.get("clustering"))},
        "map": lambda q, b: {"job": model.start_map(root, b["name"], b["paths"])},
        "docs": lambda q, b: sorted(p.name for p in docs.glob("subtle_*.md")),
        "doc": lambda q, b: doc(q),
        "kp2d": lambda q, b: kp2d(q),
        "sessions": lambda q, b: core.scan(root),
        "session": lambda q, b: session(q),
        "projects": lambda q, b: core.list_projects(root),
        "project": lambda q, b: core.load_project(root, q["name"][0]),
        "save_project": lambda q, b: {"saved": core.save_project(root, b)},
        "frequency": lambda q, b: core.frequency(root, core.load_project(root, q["name"][0]),
                                                  q.get("level", ["super"])[0],
                                                  int(q["sl"][0]) if "sl" in q else None),
        "analyze": lambda q, b: {"job": core.start_job(root, b["paths"], int(b.get("max_frames", 0)), b.get("seed"), b.get("clustering"))},
        "job": lambda q, b: {k: v for k, v in core.JOBS[q["id"][0]].items()} | {
            "elapsed": round(__import__("time").time() - core.JOBS[q["id"][0]]["t0"], 1)},
        "log": lambda q, b: list(LOGBUF),
    }


def make_handler(root: Path):
    routes = _routes(root)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            logging.getLogger("http").debug(fmt, *a)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _video(self, rel):
            p = core.safe(root, rel)
            size, rng = p.stat().st_size, self.headers.get("Range")
            a, b = 0, size - 1
            if rng and rng.startswith("bytes="):
                s, _, e = rng[6:].partition("-")
                a, b = int(s or 0), int(e) if e else size - 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def _dispatch(self, body):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/" or u.path.startswith("/web/"):  # static page: web/index.html, web/style.css, web/js/*.js
                    rel = "index.html" if u.path == "/" else u.path[5:]
                    f = (HERE / "web" / rel).resolve()
                    if (HERE / "web").resolve() not in f.parents or f.suffix not in (".html", ".css", ".js") or not f.is_file():
                        return self._json({"error": "not found"}, 404)
                    data = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}[f.suffix] + "; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                elif u.path == "/video":
                    self._video(q["path"][0])
                elif u.path.startswith("/api/") and u.path[5:] in routes:
                    self._json(routes[u.path[5:]](q, body))
                else:
                    self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed a video range request
            except Exception as e:
                logging.getLogger("http").exception("%s", u.path)
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def do_GET(self):
            self._dispatch({})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self._dispatch(json.loads(self.rfile.read(n) or b"{}"))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "subtle", help="where Export results and quality clips are written (default: <repo>/outputs/subtle)")
    a = ap.parse_args()
    core.RESULTS_BASE = a.results_dir.resolve()
    root = Path(a.root).expanduser().resolve()
    h = _Buf()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[h, logging.StreamHandler()])
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(root))
    url = f"http://127.0.0.1:{a.port}"
    logging.getLogger("subtle_viewer").info("serving %s at %s", root, url)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
