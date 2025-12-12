import numpy as np
from omfit_classes import omfit_eqdsk
import indica
import os
from indica.configs.operators.aurora_config import AuroraConfig, AuroraSteadyStateConfig
import aurora
import pytest
from dataclasses import asdict



def example_profiles():
    T_core = 2000  # eV
    T_edge = 50  # eV
    T_alpha = 1.0
    n_core = 1e14  # cm^-3
    n_edge = 0.1e14  # cm^-3
    n_alpha = 2
    rhop = np.linspace(0, 1, 100)
    ne = (n_core - n_edge) * (1 - rhop ** n_alpha) + n_edge
    Te = (T_core - T_edge) * (1 - rhop ** T_alpha) + T_edge
    return rhop, Te, ne


@pytest.fixture()
def namelist():
    _namelist = AuroraConfig.copy()
    rhop, Te, ne = example_profiles()
    _namelist["kin_profs"]["Te"]["rhop"] = _namelist["kin_profs"]["ne"]["rhop"] = rhop
    _namelist["kin_profs"]["Te"]["vals"] = Te
    _namelist["kin_profs"]["ne"]["vals"] = ne
    yield _namelist


@pytest.fixture()
def geqdsk_content():
    head = os.path.dirname(indica.__file__)
    geqdsk_content = omfit_eqdsk.OMFITgeqdsk(head + "/defaults/ST40_EFIT.geqdsk")
    yield geqdsk_content



class TestAurora:
    def test_configs_initialise(self):
        _config = AuroraConfig
        _config = AuroraSteadyStateConfig(D_z=0, V_z=0)


    def test_example_geqdsk_reading(self, geqdsk_content):
        properties = ["FPOL", "PRES", "FFPRIM", "PPRIME", "PSIRZ", "QPSI", "RBBBS", "ZBBBS", "RLIM", "ZLIM", "RHOVN"]
        for key in properties:
            assert geqdsk_content[key].size > 0


    def test_namelist_is_dict_and_not_empty(self, namelist):
        assert isinstance(namelist, dict)
        assert namelist


    def test_namelist_has_kinetic_profiles(self, namelist):
        assert "vals" in namelist["kin_profs"]["Te"].keys()
        assert "vals" in namelist["kin_profs"]["ne"].keys()
        assert "rhop" in namelist["kin_profs"]["Te"].keys()
        assert "rhop" in namelist["kin_profs"]["ne"].keys()


    def test_aurora_steady_state_sim(self, geqdsk_content, namelist,):
        asim = aurora.aurora_sim(namelist=AuroraConfig, geqdsk=geqdsk_content)
        D_z = 2e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
        V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s
        config = AuroraSteadyStateConfig(D_z=D_z, V_z=V_z)
        N_z = asim.run_aurora_steady(**config.__dict__)
        assert isinstance(N_z, np.ndarray)
        assert N_z.size > 0

