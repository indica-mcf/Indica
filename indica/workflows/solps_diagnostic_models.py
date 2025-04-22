from indica import Equilibrium
from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.readers import SOLPSReader
from indica.readers import ST40Reader


def example_pinhole_camera_2d(
    pulse: int = 11419,
    time: float = 0.12,
    machine: str = "st40",
    instrument: str = "blom_dv1",
):

    # pulse = 11419  # 11890
    # time = 0.120  # 0.105

    solps = SOLPSReader(pulse, time)
    data = solps.get()

    st40 = ST40Reader(pulse, 0, 10)
    equilibrium_data = st40.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)

    transforms = load_default_objects(machine, "geometry")
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)

    transform.beamlets = 5**2
    transform.focal_length = 0.05
    transform.spot_width = 0.005  # meter
    transform.spot_height = 0.005  # meter
    transform.spot_shape = "round"
    # transform.origin_z = np.full_like(transform.origin_z, 0.5)
    transform.distribute_beamlets()
    transform.set_dl(transform.dl)

    _, power_loss = default_atomic_data(data["nion"].element.values)

    model = PinholeCamera(name=instrument, power_loss=power_loss)
    model.set_transform(transform)

    _ = model(
        Te=data["te"],
        Ne=data["ne"],
        Nion=data["nion"],
        fz=data["fz"],
        t=data["te"].t,
        sum_beamlets=False,
    )

    model.plot()

    return model, data
