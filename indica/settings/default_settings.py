
MACHINE_DIMS = {"st40":((0.15, 0.85), (-0.75, 0.75)),
                "jet":((1.83, 3.9), (-1.75, 2.0))}


def default_geometries(machine: str, pulse: int = None):
    """
    Load default geometries for specified machine
    or
    if pulse is not None --> save new default geometry file

    Parameters
    ----------
    machine
        Machine name
    pulse
        Pulse number

    Returns
    -------
    dictionary of LOS geometries for all available diagnostics

    TODO: refactor once ReadST40 has been fixed
    """
    import pickle
    from pathlib import Path
    from indica.readers.read_st40 import ReadST40

    project_path = Path(__file__).parent.parent
    geometries_file = f"{project_path}/data/{machine}_default_geometries.pkl"

    if pulse is None:
        return pickle.load(open(geometries_file, "rb"))

    st40 = ReadST40(pulse)
    st40()
    raw_data = st40.raw_data
    geometry: dict = {}
    for instr, instr_data in raw_data.items():
        quant = list(instr_data)[0]
        quant_data = instr_data[quant]
        if not hasattr(quant_data, "transform"):
            continue

        if "Transect" in str(quant_data.transform):
            geometry[instr] = {
                "x_positions": quant_data.transform.x.values,
                "y_positions": quant_data.transform.y.values,
                "z_positions": quant_data.transform.z.values,
            }
        else:
            geometry[instr] = {
                "origin_x": quant_data.transform.origin_x,
                "origin_y": quant_data.transform.origin_y,
                "origin_z": quant_data.transform.origin_z,
                "direction_x": quant_data.transform.direction_x,
                "direction_y": quant_data.transform.direction_y,
                "direction_z": quant_data.transform.direction_z,
            }

    pickle.dump(geometry, open(geometries_file, "wb"))

    return geometry
