import os
from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot as plt
from omfit_classes import omfit_eqdsk, omfit_gapy
import aurora
from st40_database.formatted_data_writers import geqdsk


AuroraConfig = dict(
    imp="Ar",
    main_element="D",
    K=6.0,
    dr_0=0.3,
    dr_1=0.05,
    lim_sep=1,
    bound_sep=2.0,
    decay_length_boundary=0.05,
    acd=None,
    scd=None,
    ccd=None,
    cxr_flag=False,
    nbi_cxr_flag=False,
    source_type="const",
    source_rate=1e22,
    source_file=None,
    explicit_source_vals=None,
    explicit_source_time=None,
    explicit_source_rhop=None,
    source_width_in=0.0,
    source_width_out=0.0,
    source_cm_out_lcfs=1.0,
    imp_source_energy_eV=3.0,
    imp_recycling_energy_eV=3.0,
    prompt_redep_flag=False,
    clen_divertor=17.0,
    clen_limiter=0.5,
    SOL_mach=0.1,
    div_recomb_ratio=1.0,
    recycling_flag=False,
    tau_div_SOL_ms=50.0,
    div_neut_screen=0.0,
    wall_recycling=0.0,
    tau_rcl_ret_ms=50.0,
    phys_surfaces=False,
    surf_mainwall=100000.0,
    surf_divwall=10000.0,
    mainwall_roughness=1.0,
    divwall_roughness=1.0,
    phys_volumes=False,
    tau_pump_ms=500.0,
    vol_div=1000000.0,
    S_pump=5000000.0,
    pump_chamber=False,
    vol_pump=1000000.0,
    L_divpump=10000000.0,
    L_leak=0.0,
    device='ST40',
    shot=99999,
    time=1250,

    superstages=[],
    full_PWI={'main_wall_material': 'W', 'div_wall_material': 'W', 'background_mode': 'manual',
              'background_species': ['D'],
              'background_main_wall_fluxes': [0], 'background_div_wall_fluxes': [0],
              'background_files': ['file/location'],
              'characteristic_impact_energy_main_wall': 200, 'characteristic_impact_energy_div_wall': 500,
              'n_main_wall_sat': 1e+20, 'n_div_wall_sat': 1e+20, 'energetic_recycled_neutrals': False,
              'Te_div': 30.0,
              'Te_lim': 10.0, 'Ti_over_Te': 1.0, 'gammai': 2.0
              },
    saw_model={
        'crash_width': 1.0, 'mixing_radius': 1000.0, 'saw_flag': False, 'times': [1.0]
    },
    LBO={
        'n_particles': 1e+18, 't_fall': 0.3, 't_rise': 0.05, 't_start': 0.01},
    nbi_cxr={
        "rhop": None, "vals": None
    },

    timing={
        "dt_increase": [1.005, 1.],
        "dt_start": [1e-5, 1e-3],
        "steps_per_cycle": [1, 1],
        "times": [0, 0.1],
    },
    kin_profs={
        "Te": {
            "decay": [1.0], "fun": "interpa", "times": [1.0]
        },
        "Ti": {
            "decay": [1.0], "fun": "interpa", "times": [1.0]
        },
        "ne": {
            "fun": "interpa", "times": [1.0]
        },
        "n0": {
            "fun": "interpa", "times": [1.0]
        },
    }
)


@dataclass
class AuroraSteadyStateConfig:
    D_z: np.ndarray
    V_z: np.ndarray
    nz_init: np.ndarray = None
    unstage: bool = False
    alg_opt: int = 1
    evolneut: bool = False
    use_julia: bool = False
    tolerance: float = 0.001
    max_sim_time: int = 1000
    dt: float = 1e-3
    dt_increase: float = 1.05
    n_steps: int = 10
    plot: bool = False
    plot_radial_coordinate: str = 'rho_pol'

