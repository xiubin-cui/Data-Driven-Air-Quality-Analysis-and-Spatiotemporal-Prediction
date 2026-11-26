"""Unit (UT-*) and functional (FT-*) tests for the PM2.5 model.

Code under test, all in the project root: read_datasets.py, utilize.py,
encoder.py, model.py, and the checkpoint.pt written by main.py.

Run from the project root:
    python test.py -v

Needs torch, numpy, xarray, and netCDF4. Functional tests read datasets/.
FT4 and FT5 also need checkpoint.pt from python main.py.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoder import Diffusivity, SourceSink  # noqa: E402
from model import Model, R_EARTH, d_t  # noqa: E402
from read_datasets import DataSet  # noqa: E402
from utilize import (
    iter_months,
    pm2p5_ug,
    ssrd_wm2,
    test_datasets,
    train_datasets,
)  # noqa: E402

CHECKPOINT = ROOT / "checkpoint.pt"
NLAT, NLON = 241, 480


def dummy_model():
    """Model on two synthetic frames; only the grid and operators are used."""
    z = torch.zeros(2, NLAT, NLON)
    return Model(z + 10, z, z, z + 500, z + 285, z + 280, z + 100)


def area_weights(net):
    return net.mass / net.mass.sum()


# ----------------------------------------------------------------------------
# Unit tests: one function or module, synthetic inputs, no dataset files.
# ----------------------------------------------------------------------------
class UnitTests(unittest.TestCase):

    def test_UT1_iter_months(self):
        """iter_months crosses the year boundary and rejects bad ranges."""
        self.assertEqual(
            list(iter_months(2022, 11, 2023, 2)),
            [(2022, 11), (2022, 12), (2023, 1), (2023, 2)],
        )
        self.assertEqual(list(iter_months(2022, 4, 2022, 4)), [(2022, 4)])
        with self.assertRaises(ValueError):
            list(iter_months(2022, 13, 2023, 1))
        with self.assertRaises(ValueError):
            list(iter_months(2022, 5, 2022, 4))

    def test_UT2_unit_conversion(self):
        """pm2p5_ug: kg/m3 -> ug/m3, negatives and NaN -> 0; ssrd_wm2: J/m2 per 3 h -> W/m2."""
        pm = pm2p5_ug(np.array([1e-9, 2.5e-8, -1.0, np.nan]))
        self.assertTrue(torch.allclose(pm, torch.tensor([1.0, 25.0, 0.0, 0.0])))
        self.assertEqual(pm.dtype, torch.float32)
        rad = ssrd_wm2(np.array([10800.0, 0.0, -5.0]))
        self.assertTrue(torch.allclose(rad, torch.tensor([1.0, 0.0, 0.0])))

    def test_UT3_lumped_mass_covers_sphere(self):
        """Model.mass (lumped node areas) is positive and sums to 4*pi*R^2 (rel. error < 1e-4)."""
        net = dummy_model()
        self.assertEqual(tuple(net.mass.shape), (NLAT, NLON))
        self.assertTrue(bool((net.mass > 0).all()))
        sphere = 4 * math.pi * R_EARTH**2
        self.assertLess(abs(float(net.mass.sum()) - sphere) / sphere, 1e-4)

    def test_UT4_encoders(self):
        """Diffusivity is positive everywhere; SourceSink is exactly 0 before training."""
        blh, t2m, d2m, ssrd = (
            torch.full((NLAT, NLON), x) for x in (500.0, 285.0, 280.0, 100.0)
        )
        K = Diffusivity()(blh, t2m, d2m, ssrd)
        S = SourceSink()(torch.full((NLAT, NLON), 10.0), blh, t2m, d2m, ssrd)
        self.assertEqual(tuple(K.shape), (NLAT, NLON))
        self.assertTrue(bool((K > 0).all()))
        self.assertEqual(tuple(S.shape), (NLAT, NLON))
        self.assertEqual(float(S.detach().abs().max()), 0.0)

    def test_UT5_constant_field_residual(self):
        """residual_vector of a constant field is ~0 for random wind and diffusivity."""
        torch.manual_seed(0)
        net = dummy_model()
        u = torch.randn(NLAT, NLON) * 10
        v = torch.randn(NLAT, NLON) * 10
        K = torch.rand(NLAT, NLON) * 1e4
        flat = net.residual_vector(
            torch.full((NLAT, NLON), 7.0, dtype=torch.float64), u, v, K
        )
        rough = net.residual_vector(
            torch.rand(NLAT, NLON, dtype=torch.float64) * 7, u, v, K
        )
        self.assertLess(float(flat.abs().max()) / float(rough.abs().max()), 1e-12)

    def test_UT6_nan_tensor_input(self):
        """pm2p5_ug should also clean NaN when the input is already a torch tensor."""
        pm = pm2p5_ug(torch.tensor([1e-9, float("nan")]))
        self.assertTrue(bool(torch.isfinite(pm).all()), f"NaN kept: {pm.tolist()}")


# ----------------------------------------------------------------------------
# Functional tests: whole features on the real ERA5/EAC4 files and checkpoint.
# ----------------------------------------------------------------------------
class FunctionalTests(unittest.TestCase):

    def test_FT1_load_and_merge_month(self):
        """DataSet(2022,1): ERA5 + EAC4 merge on the 0.75 deg 3-hourly grid; missing month raises."""
        import xarray as xr

        ds = DataSet(2022, 1)
        merged = xr.merge([ds.get_ERA5_data(), ds.get_EAC4_data()])
        for var in ("u10", "v10", "t2m", "d2m", "blh", "tp", "ssrd", "pm2p5"):
            self.assertIn(var, merged.data_vars)
        self.assertEqual(merged.sizes["latitude"], NLAT)
        self.assertEqual(merged.sizes["longitude"], NLON)
        self.assertEqual(merged.sizes["time"], 31 * 8)
        with self.assertRaises(FileNotFoundError):
            DataSet(2030, 1).read_ERA5()

    def test_FT2_training_dataset(self):
        """train_datasets(2022,1,2022,1): 248 frames -> 247 (input, next-frame target) pairs in ug/m3."""
        data = train_datasets(2022, 1, 2022, 1)
        self.assertEqual(tuple(data.pm2p5.shape), (248, NLAT, NLON))
        self.assertEqual(len(data), 247)
        inputs, target = data[10]
        self.assertEqual(len(inputs), 7)
        for field in inputs:
            self.assertEqual(tuple(field.shape), (NLAT, NLON))
        self.assertTrue(torch.equal(target, data.pm2p5[11]))
        self.assertTrue(bool(torch.isfinite(data.pm2p5).all()))
        self.assertGreaterEqual(float(data.pm2p5.min()), 0.0)
        with self.assertRaises(IndexError):
            data[len(data)]

    def test_FT3_one_step_forecast(self):
        """Untrained Model.C_predict on real January data: one 3-hour step is finite.

        The explicit step has no positivity limiter, so an untrained forecast may
        go negative. Non-negativity is checked on the trained model in FT5.
        """
        data = train_datasets(2022, 1, 2022, 1)
        net = Model(*data())
        with torch.no_grad():
            pred = net.C_predict(d_t, 0)
        self.assertEqual(tuple(pred.shape), (NLAT, NLON))
        self.assertTrue(bool(torch.isfinite(pred).all()))

    @unittest.skipUnless(
        CHECKPOINT.exists(), "checkpoint.pt is missing; run python main.py first"
    )
    def test_FT4_checkpoint_predict(self):
        """checkpoint.pt loads into Model; April forecasts are finite and beat persistence (area-weighted MAE)."""
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        data = test_datasets(2022, 4, 2022, 4)
        net = Model(*data())
        try:
            net.load_state_dict(ckpt["model"])
        except RuntimeError as err:
            self.fail(
                f"checkpoint cannot be loaded: {str(err).splitlines()[1].strip()}"
            )
        weight = area_weights(net)
        model_mae = persist_mae = 0.0
        steps = 40
        with torch.no_grad():
            for t in range(steps):
                pred = net.C_predict(d_t, t)
                target = data.pm2p5[t + 1].double()
                self.assertTrue(bool(torch.isfinite(pred).all()))
                model_mae += float(((pred - target).abs() * weight).sum()) / steps
                persist_mae += (
                    float(((data.pm2p5[t].double() - target).abs() * weight).sum())
                    / steps
                )
        self.assertLess(model_mae, persist_mae)

    @unittest.skipUnless(
        CHECKPOINT.exists(), "checkpoint.pt is missing; run python main.py first"
    )
    def test_FT5_trained_forecast_non_negative(self):
        """Trained model (lon_grid buffers skipped) never predicts negative PM2.5 on April data."""
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state = {k: v for k, v in ckpt["model"].items() if not k.endswith("lon_grid")}
        data = test_datasets(2022, 4, 2022, 4)
        net = Model(*data())
        net.load_state_dict(state, strict=False)
        with torch.no_grad():
            lowest = min(float(net.C_predict(d_t, t).min()) for t in range(40))
        self.assertGreaterEqual(lowest, 0.0, f"lowest prediction {lowest:.1f} ug/m3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
