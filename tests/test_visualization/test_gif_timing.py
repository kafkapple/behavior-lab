"""GIF timing must not change scientific frame content or gallery prose."""
import base64
import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, ImageSequence

from behavior_lab.visualization.comparison import _retime_gif, retime_cluster_gallery
from behavior_lab.visualization.skeleton import _gif_durations_ms, animate_skeleton


def _frames(raw):
    return [hashlib.sha256(frame.convert("RGBA").tobytes()).hexdigest()
            for frame in ImageSequence.Iterator(Image.open(io.BytesIO(raw)))]


def _gif():
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "green", "blue")]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=20, loop=0)
    return out.getvalue()


@pytest.mark.parametrize("n,seconds", [(75, 2.5), (90, 3.0)])
def test_actual_writer_distributes_30fps_delay(tmp_path, n, seconds):
    keypoints = np.zeros((n, 1, 2))
    keypoints[:, 0, 0] = np.arange(n)
    path = tmp_path / "fixture.gif"
    try:
        animate_skeleton(keypoints, fps=30, figsize=(2, 2), save_path=str(path))
        im = Image.open(path)
        assert im.n_frames == n
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(im)]
        assert set(durations) == {30, 40}
        assert sum(durations) == seconds * 1000
    finally:
        plt.close("all")


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_rejected(fps):
    with pytest.raises(ValueError, match="finite and positive"):
        animate_skeleton(np.zeros((2, 1, 2)), fps=fps)
    with pytest.raises(ValueError, match="finite and positive"):
        _gif_durations_ms(2, fps)


def test_existing_fps_ceiling_and_gif_delay_limit():
    assert _gif_durations_ms(3, 100) == [20, 20, 20]
    with pytest.raises(ValueError, match="supported frame-delay range"):
        _gif_durations_ms(2, 1e-9)


def test_retime_keeps_all_rgba_and_non_timing_bytes():
    raw = _gif()
    retimed = _retime_gif(raw, 30)
    assert _frames(raw) == _frames(retimed)
    im = Image.open(io.BytesIO(retimed))
    assert [frame.info["duration"] for frame in ImageSequence.Iterator(im)] == [30, 40, 30]
    # Only one low delay byte changes per frame for this fixture.
    assert len(raw) == len(retimed)
    assert sum(a != b for a, b in zip(raw, retimed)) == 3
    assert _retime_gif(retimed, 30) == retimed


def test_republish_changes_only_the_existing_gif(tmp_path):
    uri = "data:image/gif;base64," + base64.b64encode(_gif()).decode()
    before = f'<h1>Existing research result</h1><p>fps=30.</p><img src="{uri}"><p>Keep me.</p>'
    path = tmp_path / "gallery.html"
    path.write_text(before)
    assert retime_cluster_gallery(path) == path
    after = path.read_text()
    encoded = after.split("base64,", 1)[1].split('"', 1)[0]
    replacement = "data:image/gif;base64," + encoded
    assert after.replace(replacement, uri) == before
    assert _frames(base64.b64decode(encoded)) == _frames(_gif())


@pytest.mark.parametrize("raw", [b"", b"GIF89a", b"not a GIF", _gif()[:-3]])
def test_bad_gif_is_rejected(raw):
    with pytest.raises(ValueError):
        _retime_gif(raw, 30)


def test_bad_second_gif_never_changes_gallery(tmp_path):
    good = base64.b64encode(_gif()).decode()
    bad = base64.b64encode(b"bad GIF").decode()
    text = f'<p>fps=30.</p><img src="data:image/gif;base64,{good}"><img src="data:image/gif;base64,{bad}">'
    path = tmp_path / "gallery.html"
    path.write_text(text)
    with pytest.raises(ValueError):
        retime_cluster_gallery(path)
    assert path.read_text() == text


def test_concurrent_gallery_edit_is_preserved(tmp_path, monkeypatch):
    uri = base64.b64encode(_gif()).decode()
    path = tmp_path / "gallery.html"
    original = f'<p>fps=30.</p><img src="data:image/gif;base64,{uri}">'.encode()
    path.write_bytes(original)
    read_bytes, calls = type(path).read_bytes, 0

    def read_with_peer_edit(candidate):
        nonlocal calls
        if candidate == path:
            calls += 1
            if calls == 2:
                path.write_bytes(original + b"<!-- peer edit -->")
        return read_bytes(candidate)

    monkeypatch.setattr(type(path), "read_bytes", read_with_peer_edit)
    with pytest.raises(RuntimeError, match="changed during publication"):
        retime_cluster_gallery(path)
    assert read_bytes(path) == original + b"<!-- peer edit -->"
    assert list(tmp_path.iterdir()) == [path]
