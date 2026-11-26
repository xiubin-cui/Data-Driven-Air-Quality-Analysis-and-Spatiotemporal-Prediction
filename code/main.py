import torch
import torch.nn.functional as F
from pathlib import Path

from model import Model, d_t
from utilize import train_datasets


def rms(value):
    return value.detach().float().pow(2).mean().sqrt().clamp_min(1e-3)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm2p5, u, v, blh, t2m, d2m, ssrd = train_datasets(2022, 1, 2022, 4)()
    net = Model(pm2p5, u, v, blh, t2m, d2m, ssrd).to(device)
    for name in ("pm2p5", "u", "v", "blh", "t2m", "d2m", "ssrd"):
        setattr(net, name, getattr(net, name).to(device))
    del pm2p5, u, v, blh, t2m, d2m, ssrd

    # Adam's first step is about one learning rate. Choose rates from the
    # encoder output so one step changes concentration and K only slightly.
    with torch.no_grad():
        src_y = net.source.encoder(
            torch.stack(
                (net.pm2p5[0], net.blh[0], net.t2m[0], net.d2m[0], net.ssrd[0]), -1
            ).to(net.source.encoder[0].weight)
        ).squeeze(-1)
        diff_y = net.diffusivity.encoder(
            torch.stack((net.blh[0], net.t2m[0], net.d2m[0], net.ssrd[0]), -1).to(
                net.diffusivity.encoder[0].weight
            )
        ).squeeze(-1)
    source_lr = min(0.05 / (d_t * float(rms(src_y))), 1e-4)
    k_lr = min(0.01 / float(rms(diff_y)), 1e-3)
    encoder_params = list(net.source.encoder.parameters()) + list(
        net.diffusivity.encoder.parameters()
    )
    opt = torch.optim.Adam(
        [
            {"params": [net.source.source_scale], "lr": source_lr},
            {"params": [net.diffusivity.k_scale], "lr": k_lr},
            {"params": encoder_params, "lr": 1e-5},
        ]
    )

    weight = net.mass / net.mass.sum()
    nstep = net.pm2p5.shape[0] - 1
    print(
        f"device {device}  steps {nstep}  "
        f"source_lr {source_lr:.3e}  k_lr {k_lr:.3e}",
        flush=True,
    )

    net.train()
    total_loss = 0.0
    total_mae = 0.0
    for t in range(nstep):
        pred = net.C_predict(d_t, t)
        target = net.pm2p5[t + 1].to(pred.dtype)
        error = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
        loss = (error * weight).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {t}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [net.source.source_scale, net.diffusivity.k_scale, *encoder_params],
            1.0,
        )
        opt.step()

        mae = ((pred.detach() - target).abs() * weight).sum()
        total_loss += float(loss.detach())
        total_mae += float(mae)
        if t % 50 == 0 or t + 1 == nstep:
            print(
                f"step {t:4d}  loss {total_loss / (t + 1):.4f}  "
                f"mae {total_mae / (t + 1):.4f}",
                flush=True,
            )

    ncell = net.source.source_scale.numel()
    source_changed = int((net.source.source_scale.detach() != 0).sum())
    k_changed = int((net.diffusivity.k_scale.detach() != 0).sum())
    print(
        f"updated source {source_changed}/{ncell}  " f"K {k_changed}/{ncell}",
        flush=True,
    )
    ckpt = Path(__file__).with_name("checkpoint.pt")
    torch.save(
        {"model": net.state_dict(), "months": (2022, 1, 2022, 4)},
        ckpt,
    )
    print(f"saved {ckpt}", flush=True)
    return net


if __name__ == "__main__":
    main()
