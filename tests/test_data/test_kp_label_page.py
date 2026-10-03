"""Label page generator: the JS export must produce a table the validator accepts. Synthetic template only."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from behavior_lab.pose.labels import load_label_table

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "kp_label_page.py"
spec = importlib.util.spec_from_file_location("kp_label_page", SCRIPT)
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)

KP = ["nose1", "neck1", "tail1"]


def _template(tmp_path: Path) -> Path:
    rows = [{"frame": f, "camera": "cam_1", "image": f"images/f{f:03d}_cam1.jpg", "image_w": 100, "image_h": 80,
             "keypoint": k, "x_px": None, "y_px": None, "visible": None, "note": None} for f in (3, 7) for k in KP]
    d = tmp_path / "gt"
    d.mkdir()
    p = d / "labels_template.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_refuses_a_template_that_already_has_labels(tmp_path):
    t = _template(tmp_path)
    df = pd.read_csv(t)
    df.loc[0, ["x_px", "y_px", "visible"]] = [1, 2, 1]
    df.to_csv(t, index=False)
    with pytest.raises(ValueError, match="already has 1 labelled"):
        page.build(t, tmp_path / "p.html")


def test_page_embeds_spec_and_has_no_prediction_source(tmp_path):
    out = page.build(_template(tmp_path), tmp_path / "p.html")
    html = out.read_text()
    assert '"keypoints": ["nose1", "neck1", "tail1"]' in html
    assert "predict" not in html.split("<script>")[1].lower().replace("predictions are never shown", "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed to run the page logic")
def test_js_export_roundtrips_through_the_validator(tmp_path):
    out = page.build(_template(tmp_path), tmp_path / "p.html")
    js = out.read_text().split("<script>")[1].split("</script>")[0]
    (tmp_path / "page.js").write_text(js)
    driver = (
        "const m = require('./page.js'); m.state = {};\n"
        "m.place('images/f003_cam1.jpg', 10.04, 20, 1); m.place('images/f003_cam1.jpg', 30, 40, 0); m.skip('images/f003_cam1.jpg');\n"
        "m.place('images/f007_cam1.jpg', 5, 6, 1); m.undo('images/f007_cam1.jpg');\n"
        "process.stdout.write(m.csv());"
    )
    (tmp_path / "drive.js").write_text(driver)
    # localStorage / document are absent under node: the module guards them
    res = subprocess.run(["node", "drive.js"], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    (tmp_path / "labels.csv").write_text(res.stdout)
    df = load_label_table(tmp_path / "labels.csv", keypoints=KP)
    first = df[(df.frame == 3)].set_index("keypoint")
    assert first.loc["nose1", "visible"] == 1 and first.loc["neck1", "visible"] == 0
    assert pd.isna(first.loc["tail1", "visible"])               # skipped = blank
    assert df[df.frame == 7]["visible"].isna().all()            # undone
    assert first.loc["nose1", "x_px"] == pytest.approx(10.0)
