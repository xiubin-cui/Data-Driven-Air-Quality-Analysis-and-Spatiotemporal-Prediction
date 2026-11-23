import torch
import torch.nn as nn
import torch.functional as F
import xarray as xr
import math
import numpy as np
from encoder import SourceSink, Diffusivity

# This is no AI model, don't change the code without my permission

R_EARTH = 6.371e6
d_lat = 0.75
d_lon = 0.75
d_t = 10800


class EarthGrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.nlat = 241
        self.nlon = 480

        lat = 90 - d_lat * torch.arange(self.nlat, dtype=torch.float64)
        lon = d_lon * torch.arange(self.nlon, dtype=torch.float64)

        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        phi = torch.deg2rad(lat_grid)
        theta = torch.deg2rad(lon_grid)

        x = R_EARTH * torch.cos(phi) * torch.cos(theta)
        y = R_EARTH * torch.cos(phi) * torch.sin(theta)
        z = R_EARTH * torch.sin(phi)

        x[0, :] = x[-1, :] = 0.0
        y[0, :] = y[-1, :] = 0.0
        z[0, :] = R_EARTH
        z[-1, :] = -R_EARTH

        self.register_buffer("x", x)
        self.register_buffer("y", y)
        self.register_buffer("z", z)

        self.register_buffer("lon_grid", lon_grid)
        self.register_buffer("phi", phi)
        self.register_buffer("theta", theta)

    def triangle_area(self, vertex_lat, base_lat):
        h = vertex_lat - base_lat
        dlon = math.radians(d_lon)
        return (
            R_EARTH**2
            * dlon
            * (torch.cos(base_lat) - torch.cos(vertex_lat) - h * torch.sin(base_lat))
            / h.abs()
        )

    def triangles(self):
        # One latitude band at a time. Longitude stays in the tensor, so this
        # does not walk the globe cell by cell.
        nlon = self.nlon
        nband = self.nlat - 1
        j = torch.arange(nlon, device=self.x.device)
        je = (j + 1) % nlon
        point_1 = self.x.new_empty(nband * nlon * 2, 3)
        point_2 = self.x.new_empty(nband * nlon * 2, 3)
        point_3 = self.x.new_empty(nband * nlon * 2, 3)
        area = self.x.new_empty(nband * nlon * 2)
        for i in range(nband):
            sl = slice(i * nlon * 2, (i + 1) * nlon * 2)
            area_n = self.triangle_area(self.phi[i + 1, 0], self.phi[i, 0])
            area_s = self.triangle_area(self.phi[i, 0], self.phi[i + 1, 0])
            north_1 = torch.stack((self.x[i, j], self.y[i, j], self.z[i, j]), -1)
            north_2 = torch.stack(
                (self.x[i + 1, j], self.y[i + 1, j], self.z[i + 1, j]), -1
            )
            north_3 = torch.stack((self.x[i, je], self.y[i, je], self.z[i, je]), -1)
            south_2 = torch.stack(
                (self.x[i + 1, je], self.y[i + 1, je], self.z[i + 1, je]), -1
            )
            point_1[sl] = torch.stack((north_1, north_2), 1).reshape(nlon * 2, 3)
            point_2[sl] = torch.stack((north_2, south_2), 1).reshape(nlon * 2, 3)
            point_3[sl] = torch.stack((north_3, north_3), 1).reshape(nlon * 2, 3)
            area[sl] = torch.stack(
                (area_n.expand(nlon), area_s.expand(nlon)), 1
            ).reshape(nlon * 2)
        self.register_buffer("point_1", point_1)
        self.register_buffer("point_2", point_2)
        self.register_buffer("point_3", point_3)
        self.register_buffer("Area", area)
        return (self.point_1, self.point_2, self.point_3), self.Area


class weak_pde(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = EarthGrid()
        (self.point_1, self.point_2, self.point_3), self.Area = self.grid.triangles()

    def test_function(self, vertex_lat, base_lat):
        # this is N=2, p=3 weak-form test function
        gap = float(vertex_lat) - float(base_lat)
        dtheta = math.radians(d_lon)
        q = int(gap > 0)

        N = self.grid.phi.new_tensor(
            [
                [0.0, 0.5, 0.5],  # BC midpoint
                [0.5, 0.0, 0.5],  # CA midpoint
                [0.5, 0.5, 0.0],  # AB midpoint
            ],
            dtype=torch.float64,
        )

        grad = N.new_tensor(
            [
                [-1 / dtheta, (q - 1) / gap],
                [1 / dtheta, -q / gap],
                [0.0, 1 / gap],
            ]
        )
        return N, grad

    def lumped_mass(self, vertex_lat, base_lat):
        # Lumped mass (M_A, M_B, M_C) on one right triangle, M_i = ∫ N_i dA.
        # A and B lie on base_lat, C on vertex_lat. Edge-midpoint quadrature.
        delta_theta = math.radians(d_lon)
        delta_phi = (vertex_lat - base_lat).abs()
        weight = delta_theta * delta_phi / 6
        mid = (vertex_lat + base_lat) / 2
        cos_mid = torch.cos(mid)
        cos_base = torch.cos(base_lat)
        factor = weight * R_EARTH**2
        mass_ab = factor * 0.5 * (cos_mid + cos_base)
        mass_c = factor * cos_mid
        return torch.stack((mass_ab, mass_ab, mass_c))

    def advection_matrix(self, vertex_lat, base_lat, u, v):
        vertex_lat = torch.as_tensor(
            vertex_lat, dtype=torch.float64, device=self.device
        )
        base_lat = torch.as_tensor(base_lat, dtype=torch.float64, device=self.device)
        u = torch.as_tensor(u, dtype=torch.float64, device=self.device)
        v = torch.as_tensor(v, dtype=torch.float64, device=self.device)

        delta_theta = math.radians(d_lon)
        weight = delta_theta * (vertex_lat - base_lat).abs() / 6

        cos_mid = torch.cos((vertex_lat + base_lat) / 2)
        cos_base = torch.cos(base_lat)
        cos_edge = torch.stack((cos_mid, cos_mid, cos_base))

        value, grad = self.test_function(vertex_lat, base_lat)
        g_theta = grad[:, 0]
        g_phi = grad[:, 1]

        u_A, u_B, u_C = u.unbind(dim=-1)
        v_A, v_B, v_C = v.unbind(dim=-1)
        u_mid = torch.stack((u_B + u_C, u_C + u_A, u_A + u_B), dim=-1) / 2
        v_mid = torch.stack((v_B + v_C, v_C + v_A, v_A + v_B), dim=-1) / 2

        speed = R_EARTH * (
            u_mid.unsqueeze(-1) * g_theta + (v_mid * cos_edge).unsqueeze(-1) * g_phi
        )
        speed_BC, speed_CA, speed_AB = speed.unbind(dim=-2)

        row_A = weight * (speed_CA + speed_AB) / 2
        row_B = weight * (speed_BC + speed_AB) / 2
        row_C = weight * (speed_BC + speed_CA) / 2

        return torch.stack((row_A, row_B, row_C), dim=-2)

    def diffusion_matrix(self, vertex_lat, base_lat, K):
        value, grad = self.test_function(vertex_lat, base_lat)

        vertex_lat = torch.as_tensor(vertex_lat, dtype=grad.dtype, device=grad.device)
        base_lat = torch.as_tensor(base_lat, dtype=grad.dtype, device=grad.device)
        K = torch.as_tensor(K, dtype=grad.dtype, device=grad.device)
        K_A, K_B, K_C = K.unbind(dim=-1)

        K_AB = (K_A + K_B) / 2
        K_BC = (K_B + K_C) / 2
        K_CA = (K_C + K_A) / 2

        cos_base = torch.cos(base_lat)
        cos_mid = torch.cos((vertex_lat + base_lat) / 2)
        weight = math.radians(d_lon) * abs(vertex_lat - base_lat) / 6

        alpha = weight * (K_AB / cos_base + (K_BC + K_CA) / cos_mid)
        beta = weight * (K_AB * cos_base + (K_BC + K_CA) * cos_mid)

        g_theta = grad[:, 0]
        g_phi = grad[:, 1]

        Diffusion = alpha * g_theta.unsqueeze(-1) * g_theta.unsqueeze(
            -2
        ) + beta * g_phi.unsqueeze(-1) * g_phi.unsqueeze(-2)
        return Diffusion


class assemble(nn.Module):
    def __init__(self, pm2p5, u, v, blh, t2m, d2m, ssrd):
        super().__init__()
        self.pm2p5 = pm2p5
        self.u = u
        self.v = v
        self.blh = blh
        self.t2m = t2m
        self.d2m = d2m
        self.ssrd = ssrd
        self.grid = EarthGrid()
        self.diffusivity = Diffusivity()
        self.source = SourceSink()
        self.weak_pde = weak_pde()
        # Mass depends only on the grid, so build it once.
        self.register_buffer("mass", self._build_mass())

    def _frame(self, value, t):
        if not torch.is_tensor(value):
            if hasattr(value, "values") and not callable(getattr(value, "values")):
                value = value.values
            value = torch.as_tensor(value)
        if value.ndim >= 3:
            return value[t]
        return value

    def _build_mass(self):
        nlat, nlon = self.grid.nlat, self.grid.nlon
        phi = self.grid.phi[:, 0]
        m = torch.zeros(nlat, nlon, dtype=phi.dtype, device=phi.device)

        for i in range(nlat - 1):
            if i > 0:
                m_A, m_B, m_C = self.weak_pde.lumped_mass(phi[i + 1], phi[i])
                m[i, :] = m[i, :] + m_A + m_B
                m[i + 1, :] = m[i + 1, :] + m_C
            if i < nlat - 2:
                m_A, m_B, m_C = self.weak_pde.lumped_mass(phi[i], phi[i + 1])
                m[i + 1, :] = m[i + 1, :] + m_A + m_B
                m[i, :] = m[i, :] + m_C
        return m

    def mass_vector(self):
        return self.mass

    def residual_vector(self, C, u, v, K):
        # Latitude spacing is uniform, so each orientation has one gradient.
        # All longitudes and latitude bands are one batched matmul.
        pde = self.weak_pde
        phi = pde.grid.phi[:, 0]
        nlat = C.shape[0]
        _, grad_north = pde.test_function(phi[2], phi[1])
        _, grad_south = pde.test_function(phi[0], phi[1])

        def nodes(row_a, row_c):
            # A and B lie on row_a; B is one longitude east. C is on row_c.
            return torch.stack((row_a, torch.roll(row_a, -1, -1), row_c), -1)

        def local(vertex, base, grad, uu, vv, kk, cc):
            # uu, vv, kk, cc: (band, longitude, 3 vertices)
            weight = math.radians(d_lon) * abs(float(vertex[0]) - float(base[0])) / 6
            uu, vv, kk, cc = (x.to(torch.float64) for x in (uu, vv, kk, cc))
            cos_mid = torch.cos((vertex + base) / 2)
            cos_base = torch.cos(base)
            cos_edge = torch.stack((cos_mid, cos_mid, cos_base), -1).unsqueeze(1)
            g_theta, g_phi = grad[:, 0], grad[:, 1]

            uA, uB, uC = uu.unbind(-1)
            vA, vB, vC = vv.unbind(-1)
            u_mid = torch.stack((uB + uC, uC + uA, uA + uB), -1) / 2
            v_mid = torch.stack((vB + vC, vC + vA, vA + vB), -1) / 2
            speed = R_EARTH * (
                u_mid.unsqueeze(-1) * g_theta + (v_mid * cos_edge).unsqueeze(-1) * g_phi
            )
            sBC, sCA, sAB = speed.unbind(-2)
            adv = torch.stack(
                (
                    weight * (sCA + sAB) / 2,
                    weight * (sBC + sAB) / 2,
                    weight * (sBC + sCA) / 2,
                ),
                -2,
            )

            kA, kB, kC = kk.unbind(-1)
            kAB, kBC, kCA = (kA + kB) / 2, (kB + kC) / 2, (kC + kA) / 2
            alpha = weight * (kAB / cos_base[:, None] + (kBC + kCA) / cos_mid[:, None])
            beta = weight * (kAB * cos_base[:, None] + (kBC + kCA) * cos_mid[:, None])
            diff = (
                alpha[:, :, None, None]
                * g_theta[None, None, :, None]
                * g_theta[None, None, None, :]
                + beta[:, :, None, None]
                * g_phi[None, None, :, None]
                * g_phi[None, None, None, :]
            )
            return (-(adv + diff) @ cc.unsqueeze(-1)).squeeze(-1)

        def place(acc, start, values):
            index = torch.arange(start, start + values.shape[0], device=acc.device)
            return torch.index_add(acc, 0, index, values)

        north = local(
            phi[2:nlat],
            phi[1 : nlat - 1],
            grad_north,
            nodes(u[1 : nlat - 1], u[2:nlat]),
            nodes(v[1 : nlat - 1], v[2:nlat]),
            nodes(K[1 : nlat - 1], K[2:nlat]),
            nodes(C[1 : nlat - 1], C[2:nlat]),
        )
        # South triangle: A,B on the southern row, C one longitude east on the northern row.
        south = local(
            phi[0 : nlat - 2],
            phi[1 : nlat - 1],
            grad_south,
            nodes(u[1 : nlat - 1], torch.roll(u[0 : nlat - 2], -1, -1)),
            nodes(v[1 : nlat - 1], torch.roll(v[0 : nlat - 2], -1, -1)),
            nodes(K[1 : nlat - 1], torch.roll(K[0 : nlat - 2], -1, -1)),
            nodes(C[1 : nlat - 1], torch.roll(C[0 : nlat - 2], -1, -1)),
        )
        acc = torch.zeros(nlat, C.shape[1], dtype=torch.float64, device=C.device)
        acc = place(acc, 1, north[:, :, 0] + torch.roll(north[:, :, 1], 1, -1))
        acc = place(acc, 2, north[:, :, 2])
        acc = place(acc, 1, south[:, :, 0] + torch.roll(south[:, :, 1], 1, -1))
        acc = place(acc, 0, torch.roll(south[:, :, 2], 1, -1))
        return acc.to(dtype=C.dtype)

    def forward_step(self, h, t=0):
        pm2p5 = self._frame(self.pm2p5, t)
        u = self._frame(self.u, t)
        v = self._frame(self.v, t)
        blh = self._frame(self.blh, t)
        t2m = self._frame(self.t2m, t)
        d2m = self._frame(self.d2m, t)
        ssrd = self._frame(self.ssrd, t)
        K = self.diffusivity(blh, t2m, d2m, ssrd)
        S = self.source(pm2p5, blh, t2m, d2m, ssrd)
        return h * (S + self.residual_vector(pm2p5, u, v, K) / self.mass_vector())


class Model(assemble):
    def C_predict(self, h, t=0):
        return self._frame(self.pm2p5, t) + self.forward_step(h, t)
