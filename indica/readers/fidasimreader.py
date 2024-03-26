import numpy as np
import h5py
from xarray import DataArray


# Reader function to get FIDASIM data
def get_fidasim_data(
    pulse: int,
    beam: str,
    times: np.ndarray,
    filepath: str = "/home/jonathan.wood/fidasim_output",
    user: str = "jonathan.wood",
):
    # Get Beam origin in simulation
    time_str = f"t_{times[0]:0.6f}"
    file_to_read = f"{filepath}/{pulse}/{time_str}/{beam.lower()}/"
    with open(file_to_read + "jonathan.wood_inputs.dat", "r") as f:
        lines = f.readlines()
    origin_x_str = lines[78]
    origin_y_str = lines[79]
    origin_z_str = lines[80]

    origin_x_str = origin_x_str.replace(" ", "")
    origin_x_lst = origin_x_str.split("=")
    origin_x = float(origin_x_lst[1].split('!')[0]) * 1e-2

    origin_y_str = origin_y_str.replace(" ", "")
    origin_y_lst = origin_y_str.split("=")
    origin_y = float(origin_y_lst[1].split('!')[0]) * 1e-2

    origin_z_str = origin_z_str.replace(" ", "")
    origin_z_lst = origin_z_str.split("=")
    origin_z = float(origin_z_lst[1].split('!')[0]) * 1e-2
    source = [origin_x, origin_y, origin_z]

    geom = h5py.File(file_to_read + "jonathan.wood_geometry.h5")
    axis = list(geom['nbi']['axis'])
    beam_angle = np.arctan2(axis[1], axis[0])

    # Get Neutral density map
    fdens_list: list = []
    hdens_list: list = []
    tdens_list: list = []
    for i_time in range(len(times)):
        # Construct full file path
        time = times[i_time]
        time_str = f"t_{time:0.6f}"
        file_to_read = f"{filepath}/{pulse}/{time_str}/{beam.lower()}/"

        # Read Neutral density results from FIDASIM
        h5 = h5py.File(file_to_read + f"{user}_neutrals.h5")

        # Get densities
        fdens = np.array(h5["fdens"]) * 1e6  # Full energy neutral density (z, y, x, excitation level), m-3
        hdens = np.array(h5["hdens"]) * 1e6  # 1/2 energy neutral density (z, y, x, excitation level), m-3
        tdens = np.array(h5["tdens"]) * 1e6  # 1/3 energy neutral density (z, y, x, excitation level), m-3

        # Get neutral density grid
        grid = h5["grid"]
        xgrid = np.array(grid["x"]) * 1e-2  # (m)
        ygrid = np.array(grid["y"]) * 1e-2  # (m)
        zgrid = np.array(grid["z"]) * 1e-2  # (m)

        fdens_list.append(fdens)
        hdens_list.append(hdens)
        tdens_list.append(tdens)

    # Convert to xarray arrays
    data = {
        'fdens': DataArray(
            fdens_list, coords=[
                ("t", times),
                ("height_grid", zgrid),
                ("width_grid", ygrid),
                ("depth_grid", xgrid),
                ("excitations", np.arange(0, 6))
            ]
        ),
        'hdens': DataArray(
            hdens_list, coords=[
                ("t", times),
                ("height_grid", zgrid),
                ("width_grid", ygrid),
                ("depth_grid", xgrid),
                ("excitations", np.arange(0, 6))
            ]
        ),
        'tdens': DataArray(
            tdens_list, coords=[
                ("t", times),
                ("height_grid", zgrid),
                ("width_grid", ygrid),
                ("depth_grid", xgrid),
                ("excitations", np.arange(0, 6))
            ]
        ),
        'height_grid': zgrid,
        'width_grid': ygrid,
        'depth_grid': xgrid,
        'angle': beam_angle,
        'source': source,
    }

    return data
