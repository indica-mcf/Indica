"""Tests for line-of-sight coordinate transforms."""

from copy import deepcopy

import numpy as np
import pytest
from xarray import DataArray

from indica.converters import line_of_sight
from indica.equilibrium import fake_equilibrium


class TestHelike:
    def setup_class(self):
        self.machine_dims = ((0.15, 0.85), (-0.75, 0.75))

        nchannels = 3
        self.x1 = np.arange(nchannels)

        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.17
        los_end[:, 1] = 0.0
        los_end[:, 2] = np.linspace(0.53, -0.53, nchannels)
        los_start = np.array([[1.0, 0, 0]] * los_end.shape[0])
        origin = los_start
        direction = los_end - los_start

        self.los_transform = line_of_sight.LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            machine_dimensions=self.machine_dims,
            name="los_test",
        )

        equil = fake_equilibrium()
        self.los_transform.set_equilibrium(equil)

        _profile_1d = np.abs(np.linspace(-1, 0))
        coords = [("rhop", np.linspace(0, 1.0))]
        self.profile_1d = (
            DataArray(_profile_1d, coords=coords)
            .expand_dims({"t": equil.t.size})
            .assign_coords(t=equil.t)
        )
        self.profile_2d = self.profile_1d.interp(rhop=equil.rho).drop("rhop")

    def test_convert_to_xy(self):
        x1 = self.x1[0]
        x2 = self.los_transform.x2[0]
        t = self.los_transform.equilibrium.t[0]

        x, y = self.los_transform.convert_to_xy(x1, x2, t)
        _, z = self.los_transform.convert_to_Rz(x1, x2, t)

        assert np.all(
            x.values <= np.max([self.los_transform.x_start, self.los_transform.x_end])
        )
        assert np.all(
            x >= np.min([self.los_transform.x_start, self.los_transform.x_end])
        )
        assert np.all(
            y <= np.max([self.los_transform.y_start, self.los_transform.y_end])
        )
        assert np.all(
            y >= np.min([self.los_transform.y_start, self.los_transform.y_end])
        )
        assert np.all(
            z <= np.max([self.los_transform.z_start, self.los_transform.z_end])
        )
        assert np.all(
            z >= np.min([self.los_transform.z_start, self.los_transform.z_end])
        )

    def test_convert_to_Rz(self):
        x1 = self.x1[0]
        x2 = self.los_transform.x2[0]
        t = self.los_transform.equilibrium.t[0]

        # Test method
        R_, z_ = self.los_transform.convert_to_Rz(x1, x2, t)

        x, y = self.los_transform.convert_to_xy(x1, x2, t)
        R = np.sqrt(x**2 + y**2)

        assert R == R_

    def test_distance(self):
        x1 = self.x1[0]
        x2 = self.los_transform.x2
        t = self.los_transform.equilibrium.t[0]

        dist = self.los_transform.distance("los_position", x1, x2, t)
        for beamlet in dist.beamlet:
            _dist = dist.sel(beamlet=beamlet).values
            dls = [_dist[i + 1] - _dist[i] for i in range(len(_dist) - 1)]
            print(dls)

            assert all(np.abs(dls - dls[0]) / dls[0] < (dls[0] * 1.0e-6))

    def test_set_dl(self):
        dl = 0.002

        self.los_transform.set_dl(dl)
        dl_out = self.los_transform.dl

        print(dl, dl_out, np.abs(dl - dl_out) / dl)

        assert pytest.approx(np.abs(dl - dl_out) / dl, abs=1.0e-2) == 0

    def test_missing_los(self):
        # TODO: substitute print with better testing assertion
        origin = np.array(
            [
                [4.0, -2.0, 0.5],
            ]
        )
        direction = np.array(
            [
                [0.0, 1.0, 0.0],
            ]
        )

        try:
            _ = line_of_sight.LineOfSightTransform(
                origin[:, 0],
                origin[:, 1],
                origin[:, 2],
                direction[:, 0],
                direction[:, 1],
                direction[:, 2],
                machine_dimensions=self.machine_dims,
                name="los_test",
            )
        except ValueError:
            print("LOS initialisation failed with ValueError as expected")

    def test_2d_los_mapping(self):
        los_transform_1d = self.los_transform
        los_transform_2d = deepcopy(self.los_transform)

        time = los_transform_2d.equilibrium.rho.t.values[0:2]

        los_int_1d = los_transform_1d.integrate_on_los(self.profile_1d, t=time)
        los_int_2d = los_transform_2d.integrate_on_los(self.profile_2d, t=time)

        along_los_1d = los_transform_1d.along_los
        along_los_2d = los_transform_2d.along_los

        assert (
            pytest.approx(
                np.mean((along_los_1d - along_los_2d) / along_los_1d), abs=1.0e-2
            )
            == 0
        )
        assert (
            pytest.approx(np.mean((los_int_1d - los_int_2d) / los_int_1d), abs=1.0e-2)
            == 0
        )
