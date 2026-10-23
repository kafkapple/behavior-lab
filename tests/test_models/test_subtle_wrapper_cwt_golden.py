"""The scipy>=1.15 shim in subtle_wrapper must reproduce scipy<=1.14 signal.cwt(morlet2) (golden made once from scipy 1.9.3)."""
import pathlib

import numpy as np

from behavior_lab.models.discovery.subtle_wrapper import _cwt_compat, _morlet2_compat

GOLDEN = pathlib.Path(__file__).with_name("cwt_golden_scipy.npz")


def test_shim_matches_scipy_golden():
    d = np.load(GOLDEN)
    fs, omega, n_channels = int(d["fs"]), float(d["omega"]), int(d["n_channels"])
    freq = np.linspace(fs / 2 / 10, fs / 2, n_channels)
    widths = omega * fs / (2 * freq * np.pi)
    out = np.abs(_cwt_compat(d["x"], _morlet2_compat, widths, w=omega))
    np.testing.assert_allclose(out, d["y"], rtol=0, atol=1e-10)
