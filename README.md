## Global PM2.5 Advection–Diffusion Model

This is course project for UCSD DSC-200 courses.

A weak-form advection–diffusion model for surface PM2.5 on a global 0.75° grid. Wind-driven transport is taken from ERA5. Local source/sink and diffusivity are predicted by two small networks and scaled by learned per-cell coefficients. The training script steps forward one 3-hour interval at a time and fits the next observed concentration.

### Setup

Python 3.13 is required. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

`requirements.txt` installs the PyPI build of PyTorch. This project was developed with the CUDA 13.0 wheel (`torch==2.14.1+cu130`). To match that build:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

Training uses CUDA when it is available and falls back to CPU.

### Train

Run from the project root. Dataset paths are resolved from the directory that contains `read_datasets.py`.

```bash
python main.py
```

The script prints the device, the number of steps, the chosen learning rates, and a running loss and mass-weighted MAE every 50 steps. It writes `checkpoint.pt` next to `main.py`, with the model state dict and the month range `(2022, 1, 2022, 4)`.

To load another month range, change the `train_datasets(2022, 1, 2022, 4)` call in `main.py`. The matching NetCDF files must already be in `datasets/`.

### Inspect one month

```bash
python read_datasets.py
```

This loads January 2022 and prints the first rows of the merged ERA5 and PM2.5 table.
