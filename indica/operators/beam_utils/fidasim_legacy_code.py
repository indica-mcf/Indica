"""Archived plotting/debug snippets previously in fidasim_utils.py.

These blocks are preserved for reference only and are not executed.
"""

# Might be used later.
FIDASIM_POSTPROCESING=r"""
def postproc_fidasim(
    shot: int,
    time: float,
    nbiconfig: dict,
    specconfig: dict,
    plasmaconfig: dict,
    save_dir: str = FIDASIM_OUTPUT_DIR,
    process_spec=True,
    block=False,
    debug=False,
    los_type="center",
):

    Collect fidasim hdf5 results from each pini.
    Optionally fit CXS spectra and save processed output to a JSON dictionary.

    Parameters
    ----------
    process_spec : bool
        Flag for collecting and fitting CXS spectra for each pini, as well as
        the total of all pinis.



    out_dict = {}  # Ouptut dictionary containing combined pini results
    time = time
    st40_beams = nbiconfig
    beam_amu = st40_beams["ab"]
    beam_name = st40_beams["name"]
    st40_spec = specconfig
    runid = pwd.getpwuid(os.getuid())[0]
    spec_name = st40_spec["name"]
    cross_section_corr = False
    if "cross_section_corr" in st40_spec:
        cross_section_corr = st40_spec["cross_section_corr"]
    plasma_ion_amu = plasmaconfig["plasma_ion_amu"]

    out_dict["amu"] = plasma_ion_amu

    # Configure spec dictionary compatible with fidasim format.
    spec = None
    if spec_name in st40_spec["name"]:
        pi_spec = CxsSpec(
            shot,
            chord_IDs=st40_spec["chord_IDs"],
            amu=plasma_ion_amu,
            beam_amu=beam_amu,
            beam_name=beam_name,
            spec_name=spec_name,
            cross_section_corr=cross_section_corr,
            custom_geo_dict=st40_spec["geom_dict"],
        )
        nchan = len(st40_spec["chord_IDs"])

        ids = []
        for id in st40_spec["chord_IDs"]:
            ids.append(id.encode(encoding="utf_8"))

        ids = []
        radius = []
        lens = []
        axis = []
        _spot_radius = 1.25  # TODO: estimate spot radius on Princeton foreoptic
        spot_size = []
        _sigma_pi_ratio = 1.0  # default sigma/pi ratio
        sigma_pi = []

        ## import LOS data from local pickle file (J Wood 29/07/22)
        # import pickle
        # los_data = pickle.load(open('PI_LOS_geometry_processed.p', 'rb'))
        # los_data = los_data['3POINT_AV']

        for index, chord in enumerate(pi_spec.chords):

            ids.append(chord.id.encode(encoding="utf_8"))
            radius.append(chord.tang_rad)
            lens.append(chord.origin)
            axis.append(chord.diruvec)
            spot_size.append(_spot_radius)
            sigma_pi.append(_sigma_pi_ratio)

    # run directory
    time_str = "t_{:8.6f}".format(time)
    run_dir = save_dir + "/" + str(shot) + "/" + time_str
    plasma_file = run_dir + "/TE-fidasim_plasma.json"

    # Collect fidasim results for each beam and store in output dictionary
    # icnt = 0
    # for beam_id, beam_detail in st40_beams.items():

    beam_save_dir = run_dir + "/" + beam_name

    if not os.path.exists(beam_save_dir):
        raise FileNotFoundError(f"Results directory path not found: {beam_save_dir}")

    if spec_name in st40_spec["name"] and process_spec:
        spec_file = beam_save_dir + "/" + runid + "_spectra.h5"
        geo_file = beam_save_dir + "/" + runid + "_geometry.h5"
        # dcx_file = beam_save_dir + '/' + runid + '_dcx.h5'
        neut_file = beam_save_dir + "/" + runid + "_neutrals.h5"

        try:
            open(spec_file, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(f"Results spectra file not found: {spec_file}")

        # Collect results from fidasim
        pi_spec.collect_pini_spectra(beam_name, spec_file, geo_file, neut_file)

        # Using fidasim DCX and halo density, manually perform line-integration
        # as a sanity check against fidasim.
        pi_spec.los_integrate_pini_brightness(
            beam_name, beam_save_dir, plasma_file, neut_file
        )

        # Using fidasim full-energy neutral beam density, manually perform CVI
        # line-integration.
        # Assume constant C_6+ concetration
        pi_spec.los_integrate_CVI_brightness(
            beam_name, beam_save_dir, plasma_file, neut_file, block=block
        )

    export_dict = dict()
    if spec_name in st40_spec["name"] and process_spec:
        # Fit fidasim spectra from individual pini and sum of pinis for Ti, v_tor
        pi_spec.fit_spectra(block=block)

        # Calculate Doppler shifts for full, half, and third-energy components
        # of each pini.
        pi_spec.calc_bes_dopp_shifts()

        # Also fit manually line-integrated spectra from each beam and the
        # beam-summed spectra for Ti, v_tor.
        # Spectra are generated using fidasim 3D density plots and 2D poloidal
        # plasma Ti contours.
        pi_spec.fit_spectra(fit_manual_los_integral=True, block=block)
        pi_spec.fit_spectra(fit_manual_cvi_integral=True, block=block, run_dir=run_dir)

        # Save results to JSON dictionary and append to main output dictionary
        out_dict[spec_name] = pi_spec.serialize()
        # Extract fit data, export as dictionary
        Ti = np.zeros(len(out_dict[spec_name].keys()))
        Ti_err = np.zeros(len(out_dict[spec_name].keys()))
        cwl = np.zeros(len(out_dict[spec_name].keys()))
        cwl_err = np.zeros(len(out_dict[spec_name].keys()))
        vtor = np.zeros(len(out_dict[spec_name].keys()))
        vtor_err = np.zeros(len(out_dict[spec_name].keys()))
        for i_chord, id in enumerate(out_dict[spec_name].keys()):
            Ti[i_chord] = out_dict[spec_name][id]["res"][beam_name]["man_los_integral"][
                "fit_cvi"
            ]["Ti"]
            Ti_err[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["Ti_err"]
            cwl[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["cwl"]
            cwl_err[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["cwl_err"]

            # Convert Doppler shift to toroidal rotation
            vtor[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]["origin"],
                np.array(out_dict[spec_name][id]["beam_intersect_pos"][beam_name]),
                529.059 - cwl[i_chord],
                529.059,
            )
            vtor_err[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]["origin"],
                np.array(out_dict[spec_name][id]["beam_intersect_pos"][beam_name]),
                cwl_err[i_chord],
                529.059,
            )

        export_dict["chord_id"] = list(out_dict[spec_name].keys())
        export_dict["Ti"] = Ti
        export_dict["Ti_err"] = Ti_err
        export_dict["cwl"] = cwl
        export_dict["cwl_err"] = cwl_err
        export_dict["vtor"] = vtor
        export_dict["vtor_err"] = vtor_err

    # Write output dictionary in JSON format and save to run directory
    savefile = run_dir + "/TE-fidasim_output.json"
    with open(savefile, mode="w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    # Plotting removed (see fidasim_utils_plotting_legacy.py).

    # Export temperature and velocity results from simulated data
    return export_dict
    """




PLOTTING_SNIPPETS = r"""
# Geometry plot for inspection
ax = None
if plot_geo:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    Rmaj = plt.Circle((0,0), 40, color='k', fill=False)
    Rsep = plt.Circle((0,0), 40+26, color='darkgrey', fill=False)
    ax.add_patch(Rmaj)
    ax.add_patch(Rsep)
    art3d.pathpatch_2d_to_3d(Rmaj, z=0, zdir="z")
    art3d.pathpatch_2d_to_3d(Rsep, z=0, zdir="z")

if plot_geo:
    plt.show()

debugging_shape = False
if debugging_shape:
    plt.figure()
    plt.imshow(mask)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.contourf(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        plasma['ti'],
    )
    plt.contour(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        plasma['ti'],
        [1.0*1e-3],
        colors='k',
    )
    plt.ylim([rmin, rmax])
    plt.xlim([zmin, zmax])
    plt.subplot(122)
    plt.contourf(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        rhogrid,
    )
    plt.contour(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        rhogrid,
        [1.0],
        colors='k',
    )
    plt.ylim([rmin, rmax])
    plt.xlim([zmin, zmax])

    plt.figure()
    plt.subplot(131)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['br'],
    )
    plt.colorbar()
    plt.subplot(132)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['bz'],
    )
    plt.colorbar()
    plt.subplot(133)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['bt'],
    )
    plt.colorbar()

    plt.show(block=True)

if (not block) and debug:
    plt.show(block=True)
else:
    plt.close('all')
"""
