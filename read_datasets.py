from pathlib import Path
import xarray as xr
import pandas as pd

ROOT = Path(__file__).resolve().parent
ERA5_DIR = ROOT / "datasets" / "era5_3h_0p75"
EAC4_DIR = ROOT / "datasets" / "eac4_pm2p5"


class DataSet:
    def __init__(self, Year: int, Month: int):
        self.Year = Year
        self.Month = Month

    def read_EAC4(self):
        path = EAC4_DIR / f"pm2p5_{self.Year:04d}{self.Month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        return xr.open_dataset(path)

    def read_ERA5(self):
        path = ERA5_DIR / f"era5_3h_0p75_{self.Year:04d}{self.Month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        return xr.open_dataset(path)

    def get_ERA5_data(self, feature=("u10", "v10", "t2m", "d2m", "blh", "tp", "ssrd")):
        """
        Load selected ERA5 variables.

        u10   : 10 m eastward wind component
        v10   : 10 m northward wind component
        t2m   : 2 m air temperature
        d2m   : 2 m dewpoint temperature
        blh   : Boundary layer height
        tp    : Total precipitation
        ssrd  : Surface solar radiation downwards
        """

        ds = self.read_ERA5()
        try:
            return ds[list(feature)].load()
        finally:
            ds.close()

    def get_EAC4_data(self, feature=("pm2p5",)):
        """
        Load PM2.5 data.

        pm2p5 : PM2.5 concentration
        """

        ds = self.read_EAC4()
        try:
            return ds[list(feature)].load()
        finally:
            ds.close()


if __name__ == "__main__":
    ds = DataSet(2022, 1)
    era5_data = ds.get_ERA5_data(
        feature=("u10", "v10", "t2m", "d2m", "blh", "tp", "ssrd")
    )
    eac4_data = ds.get_EAC4_data(feature=("pm2p5",))
    table = xr.merge([era5_data, eac4_data]).to_dataframe()
    print(table.head(5))
