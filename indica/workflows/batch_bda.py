from indica.workflows.bayes_workflow_example import BayesWorkflowExample, DEFAULT_PRIORS, DEFAULT_PROFILE_PARAMS

PARAMS_SET_TS =[
            # "Ne_prof.y1",
            # "Ne_prof.y0",
            # "Ne_prof.peaking",
            # "Ne_prof.wcenter",
            # "Ne_prof.wped",
            # "Nimp_prof.y1",
            "Nimp_prof.y0",
            "Nimp_prof.wcenter",
            # "Nimp_prof.wped",
            "Nimp_prof.peaking",
            # "Te_prof.y0",
            # "Te_prof.wped",
            # "Te_prof.wcenter",
            # "Te_prof.peaking",
            "Ti_prof.y0",
            # # "Ti_prof.wped",
            "Ti_prof.wcenter",
            "Ti_prof.peaking",
            "xrcs.pixel_offset",
        ]

PARAMS_DEFAULT =[
            "Ne_prof.y1",
            "Ne_prof.y0",
            "Ne_prof.peaking",
            # "Ne_prof.wcenter",
            "Ne_prof.wped",
            # "Nimp_prof.y1",
            "Nimp_prof.y0",
            # "Nimp_prof.wcenter",
            # "Nimp_prof.wped",
            "Nimp_prof.peaking",
            "Te_prof.y0",
            # "Te_prof.wped",
            # "Te_prof.wcenter",
            "Te_prof.peaking",
            "Ti_prof.y0",
            # "Ti_prof.wped",
            "Ti_prof.wcenter",
            "Ti_prof.peaking",
            "xrcs.pixel_offset",
        ]

PARAMS_SET_ALL =[
            # "Ne_prof.y1",
            # "Ne_prof.y0",
            # "Ne_prof.peaking",
            # "Ne_prof.wcenter",
            # "Ne_prof.wped",
            # "Nimp_prof.y1",
            # "Nimp_prof.y0",
            # "Nimp_prof.wcenter",
            # "Nimp_prof.wped",
            # "Nimp_prof.peaking",
            # "Te_prof.y0",
            # "Te_prof.wped",
            # "Te_prof.wcenter",
            # "Te_prof.peaking",
            # "Ti_prof.y0",
            # "Ti_prof.wped",
            # "Ti_prof.wcenter",
            # "Ti_prof.peaking",
            "xrcs.pixel_offset",
        ]


DIAG_DEFAULT = [
                "xrcs",
                "ts",
                "efit",
                "cxff_pi",
                "cxff_tws_c"
                # "smmh1",
            ]

DIAG_DEFAULT_CHERS = [
                "xrcs",
                "ts",
                "efit",
                # "cxff_pi",
                "cxff_tws_c"
                # "smmh1",
            ]

DIAG_DEFAULT_PI = [
                "xrcs",
                "ts",
                "efit",
                "cxff_pi",
                # "cxff_tws_c"
                # "smmh1",
            ]


DIAG_SET_TS = [
                "xrcs",
                # "ts",
                "efit",
                "cxff_pi",
                "cxff_tws_c"
                # "smmh1",
            ]

OPT_DEFAULT = [
              "xrcs.spectra",
              "ts.ne",
              "ts.te",
              "efit.wp",
              "cxff_pi.ti",
              "cxff_tws_c.ti",
              # "smmh1.ne"
          ]

OPT_SET_TS = [
              "xrcs.spectra",
              # "ts.ne",
              # "ts.te",
              "efit.wp",
              "cxff_pi.ti",
              "cxff_tws_c.ti",
              # "smmh1.ne"
          ]

def pulse_example(pulse, diagnostics, param_names, opt_quantity, t_opt, iterations=100, run="RUN01",
                  fast_particles=False, astra_run="RUN602", astra_pulse_range=13000000,
                  astra_equilibrium=False, efit_revision=0, set_ts_profiles=False, set_all_profiles=True,
                  astra_wp=False, **kwargs):

    workflow = BayesWorkflowExample(
        pulse=pulse,
        diagnostics=diagnostics,
        param_names=param_names,
        opt_quantity=opt_quantity,
        priors=DEFAULT_PRIORS,
        profile_params=DEFAULT_PROFILE_PARAMS,
        phantoms=False,
        fast_particles=fast_particles,
        tstart=0.06,
        tend=0.10,
        dt=0.005,
        astra_run=astra_run,
        astra_pulse_range=astra_pulse_range,
        astra_equilibrium=astra_equilibrium,
        efit_revision=efit_revision,
        set_ts_profiles = set_ts_profiles,
        set_all_profiles=set_all_profiles,
        astra_wp = astra_wp,
    )

    workflow.setup_plasma(
        tsample=t_opt,
        # n_rad=50
    )
    workflow.setup_opt_data(phantoms=workflow.phantoms)
    workflow.setup_optimiser(nwalkers=40, sample_method="high_density", model_kwargs=workflow.model_kwargs, nsamples=100)
    results = workflow(
        filepath=f"./results/{workflow.pulse}.{run}/",
        pulse_to_write=25000000 + workflow.pulse,
        run=run,
        mds_write=False,
        plot=True,
        burn_frac=0.20,
        iterations=iterations,
    )

# (11089, DIAG_DEFAULT_CHERS, PARAMS_DEFAULT, [
        #     "xrcs.spectra",
        #     "ts.ne",
        #     "ts.te",
        #     "efit.wp",
        #     # "cxff_pi.ti",
        #     "cxff_tws_c.ti",
        #     # "smmh1.ne"
        # ], 0.100,
        #  dict(run="RUN01", fast_particles=True, astra_run="RUN601", astra_pulse_range=13000000, astra_equilibrium=False,
        #       efit_revision=2)),



if __name__ == "__main__":

    pulse_info = [

        # (11211, DIAG_DEFAULT, PARAMS_DEFAULT, OPT_DEFAULT, 0.084),
        # (11215, DIAG_DEFAULT_CHERS, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       # "cxff_pi.ti",
        #       "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.070, dict(fast_particles=True, astra_run="RUN603", astra_pulse_range=13000000, astra_equilibrium=True, astra_wp=True)),
        #
        # # (11224, DIAG_DEFAULT, PARAMS_DEFAULT, OPT_DEFAULT, 0.090),
        # (11225, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.090, dict(efit_revision=2, fast_particles=True, astra_run="RUN603", astra_pulse_range=13000000, astra_equilibrium=True, astra_wp=True)),
        #
        # # (11226, DIAG_DEFAULT, PARAMS_DEFAULT, OPT_DEFAULT, 0.070),
        # (11227, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.070, dict(fast_particles=True, astra_run="RUN603", astra_pulse_range=13000000, astra_equilibrium=True, astra_wp=True)),
        #
        # (11228, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.080, dict(run="RUN02", fast_particles=True, astra_run="RUN603", astra_pulse_range=13000000, astra_equilibrium=True, astra_wp=True)),
        # (11238, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.075, dict(efit_revision=2, fast_particles=True, astra_run="RUN603", astra_pulse_range=13000000, astra_equilibrium=True, astra_wp=True)),
        #
        # (11312, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.080, dict(run="RUN01", fast_particles=True, astra_run="RUN14", astra_pulse_range=33000000, astra_equilibrium=True, set_ts_profiles=False, set_all_profiles=False, astra_wp=True)),
        #
        # (11314, DIAG_DEFAULT_PI, PARAMS_DEFAULT, [
        #       "xrcs.spectra",
        #       "ts.ne",
        #       "ts.te",
        #       "efit.wp",
        #       "cxff_pi.ti",
        #       # "cxff_tws_c.ti",
        #       # "smmh1.ne"
        #   ], 0.080, dict(run="RUN01", fast_particles=True, astra_run="RUN3", astra_pulse_range=33000000, astra_equilibrium=True, set_ts_profiles=False, astra_wp=True)),
        (11312, DIAG_SET_TS, PARAMS_SET_TS, [
            "xrcs.spectra",
            # "ts.ne",
            # "ts.te",
            # "efit.wp",
            "cxff_pi.ti",
            # "cxff_tws_c.ti",
            # "smmh1.ne"
        ], 0.080,
         dict(run="TEST_TI", fast_particles=True, astra_run="RUN14", astra_pulse_range=33000000, astra_equilibrium=True,
              set_ts_profiles=True, set_all_profiles=False, astra_wp=True)),

        # Tree doesnt exist
        ## (11317, 0.080, dict(run="RUN01", fast_particles=True, astra_run="RUN11", astra_pulse_range=33000000, astra_equilibrium=True, set_ts_profiles=True, astra_wp=True)),
    ]

    # for info in pulse_info:
    #     if len(info) < 6:
    #         info = list(info)
    #         info.append(dict())
    #         info = tuple(info)
    #     try:
    #         pulse_example(*info[0:5], iterations=2, **info[5])
    #     except Exception as e:
    #         print(f"pulse: {info[0]}")
    #         print(repr(e))
    #         continue

    for info in pulse_info:
        pulse_example(*info[0:5], iterations=2000, **info[5])