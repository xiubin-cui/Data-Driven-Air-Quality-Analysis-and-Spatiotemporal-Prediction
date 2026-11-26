import numpy as np
import xarray as xr
import torch
import torch.nn as nn

from read_datasets import DataSet

DT = 10800.0


def as_tensor(value):
    if torch.is_tensor(value):
        value = value.detach().to(dtype=torch.float32).contiguous()
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if hasattr(value, "values") and not callable(getattr(value, "values")):
        value = value.values
    array = np.nan_to_num(np.asarray(value), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)


def pm2p5_ug(pm2p5):
    # EAC4 stores kg/m³. The source network uses μg/m³.
    return as_tensor(pm2p5).clamp_min(0) * 1e9


def ssrd_wm2(ssrd):
    # ERA5 ssrd is joules per square metre over 3 hours.
    return as_tensor(ssrd).clamp_min(0) / DT


def source_inputs(pm2p5, blh, t2m, d2m, ssrd):
    return (
        pm2p5_ug(pm2p5),
        as_tensor(blh),
        as_tensor(t2m),
        as_tensor(d2m),
        ssrd_wm2(ssrd),
    )


def iter_months(start_year, start_month, end_year, end_month):
    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        raise ValueError("month must be from 1 to 12")
    start = start_year * 12 + start_month
    end = end_year * 12 + end_month
    if start > end:
        raise ValueError("start is after end")
    year, month = start_year, start_month
    while year * 12 + month <= end:
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def load_range(start_year, start_month, end_year, end_month):
    pm2p5, u, v, blh, t2m, d2m, ssrd = [], [], [], [], [], [], []
    for year, month in iter_months(start_year, start_month, end_year, end_month):
        ds = DataSet(year, month)
        era = ds.get_ERA5_data(feature=("u10", "v10", "t2m", "d2m", "blh", "ssrd"))
        eac = ds.get_EAC4_data(feature=("pm2p5",))
        if float(era.latitude[0]) < float(era.latitude[-1]):
            era = era.isel(latitude=slice(None, None, -1))
        if float(eac.latitude[0]) < float(eac.latitude[-1]):
            eac = eac.isel(latitude=slice(None, None, -1))
        if era.sizes["time"] != eac.sizes["time"]:
            raise ValueError(f"{year}-{month:02d} ERA5 and EAC4 time lengths differ")
        c, h, t, d, s = source_inputs(
            eac["pm2p5"], era["blh"], era["t2m"], era["d2m"], era["ssrd"]
        )
        pm2p5.append(c)
        u.append(as_tensor(era["u10"]))
        v.append(as_tensor(era["v10"]))
        blh.append(h)
        t2m.append(t)
        d2m.append(d)
        ssrd.append(s)
    return tuple(torch.cat(part, dim=0) for part in (pm2p5, u, v, blh, t2m, d2m, ssrd))


class train_datasets(nn.Module):
    def __init__(self, start_year, start_month, end_year, end_month):
        super().__init__()
        self.pm2p5, self.u, self.v, self.blh, self.t2m, self.d2m, self.ssrd = (
            load_range(start_year, start_month, end_year, end_month)
        )

    def __len__(self):
        return self.pm2p5.shape[0] - 1

    def __getitem__(self, t):
        t = int(t)
        if t < 0 or t >= len(self):
            raise IndexError(t)
        inputs = (
            self.pm2p5[t],
            self.u[t],
            self.v[t],
            self.blh[t],
            self.t2m[t],
            self.d2m[t],
            self.ssrd[t],
        )
        return inputs, self.pm2p5[t + 1]

    def forward(self):
        return self.pm2p5, self.u, self.v, self.blh, self.t2m, self.d2m, self.ssrd


class test_datasets(train_datasets):
    def __init__(self, start_year, start_month, end_year, end_month):
        super().__init__(start_year, start_month, end_year, end_month)
