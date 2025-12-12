from omfit_classes import omfit_eqdsk
import indica
import os
from indica.configs.operators.aurora_config import AuroraConfig, AuroraSteadyStateConfig
from st40_database.formatted_data_writers import geqdsk
import aurora


def example_geqdsk():
    head = os.path.dirname(indica.__file__)
    geqdsk_content = omfit_eqdsk.OMFITgeqdsk(head + "/defaults/ST40_EFIT.geqdsk")
    return geqdsk_content


class TestAurora:
    def test_configs_initialise(self):
        _config = AuroraConfig
        _config = AuroraSteadyStateConfig(D_z=0, V_z=0)


    def test_example_geqdsk_reading(self):
        geqdsk_content = example_geqdsk()
        properties = ["FPOL", "PRES", "FFPRIM", "PPRIME", "PSIRZ", "QPSI", "RBBBS", "ZBBBS", "RLIM", "ZLIM", "RHOVN"]
        for key in properties:
            assert geqdsk_content[key].size > 0

    def test_aurora_sim(self):
        geqdsk_content = example_geqdsk()
        asim = aurora.aurora_sim(namelist=AuroraConfig, geqdsk=geqdsk_content)
