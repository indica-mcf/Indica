from indica.workflows.bayes_workflow import BayesWorkflow, DEFAULT_PRIORS, DEFAULT_PROFILE_PARAMS, BayesBBSettings, \
    ReaderSettings, \
    ExpData, PhantomData, PlasmaSettings, PlasmaContext, ModelContext, EmceeOptimiser, OptimiserEmceeSettings, \
    ModelSettings

PARAMS_DEFAULT =[
            "Ne_prof.y1",
            "Ne_prof.y0",
            "Ne_prof.peaking",
            # "Ne_prof.wcenter",
            "Ne_prof.wped",
            # "Niz1_prof.y1",
            "Niz1_prof.y0",
            # "Niz1_prof.wcenter",
            # "Niz1_prof.wped",
            "Niz1_prof.peaking",
            "Te_prof.y0",
            "Te_prof.wped",
            "Te_prof.wcenter",
            "Te_prof.peaking",
            "Ti_prof.y0",
            # "Ti_prof.wped",
            "Ti_prof.wcenter",
            "Ti_prof.peaking",
        ]

PARAMS_SET_TS = [
            # "Ne_prof.y1",
            # "Ne_prof.y0",
            # "Ne_prof.peaking",
            # "Ne_prof.wcenter",
            # "Ne_prof.wped",
            # "Niz1_prof.y1",
            "Niz1_prof.y0",
            # "Niz1_prof.wcenter",
            # "Niz1_prof.wped",
            "Niz1_prof.peaking",
            # "Te_prof.y0",
            # "Te_prof.wped",
            # "Te_prof.wcenter",
            # "Te_prof.peaking",
            "Ti_prof.y0",
            # "Ti_prof.wped",
            "Ti_prof.wcenter",
            "Ti_prof.peaking",
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
                # "efit",
                # "cxff_pi",
                "cxff_tws_c"
                # "smmh1",
            ]

DIAG_DEFAULT_PI = [
                "xrcs",
                "ts",
                # "efit",
                "cxff_pi",
                # "cxff_tws_c"
                # "smmh1",
            ]

OPT_DEFAULT = [
              "xrcs.spectra",
              "ts.ne",
              "ts.te",
              # "efit.wp",
              "cxff_pi.ti",
              "cxff_tws_c.ti",
              # "smmh1.ne"
          ]


def bda_run(pulse, diagnostics, param_names, opt_quantity, phantom=False,
            tstart=0.02, tend=0.05, dt=0.01, revisions = {}, filters={},
            iterations=500, nwalkers=50, stopping_criteria_factor=0.002,
            mds_write=False, plot=False, run="RUN01", dirname=None, set_ts=False, **kwargs):

    if dirname is None:
        dirname = f"{pulse}"

    # BlackBoxSettings
    bayes_settings = BayesBBSettings(diagnostics=diagnostics, param_names=param_names,
                                     opt_quantity=opt_quantity, priors=DEFAULT_PRIORS, )

    data_settings = ReaderSettings(filters=filters,
                                   revisions=revisions)
    if phantom:
        data_context = PhantomData(pulse=pulse, diagnostics=diagnostics,
                                   tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    else:
        data_context = ExpData(pulse=pulse, diagnostics=diagnostics,
                               tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    data_context.read_data()

    plasma_settings = PlasmaSettings(main_ion="h", impurities=("ar", "c"), impurity_concentration=(0.001, 0.04),
                                     n_rad=20)
    plasma_context = PlasmaContext(plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS)

    plasma_context.init_plasma(data_context.equilibrium, tstart=tstart, tend=tend, dt=dt)
    plasma_context.save_phantom_profiles(phantoms=data_context.phantoms)

    if set_ts:
        plasma_context.set_ts_profiles(data_context, split="LFS")

    model_settings = ModelSettings(call_kwargs={"xrcs": {"pixel_offset": 0.0}})

    model_context = ModelContext(diagnostics=diagnostics,
                                 plasma_context=plasma_context,
                                 equilibrium=data_context.equilibrium,
                                 transforms=data_context.transforms,
                                 model_settings=model_settings,
                                 )
    model_context.update_model_kwargs(data_context.binned_data)
    model_context.init_models()

    data_context.process_data(model_context._build_bckc, )

    optimiser_settings = OptimiserEmceeSettings(param_names=bayes_settings.param_names, nwalkers=nwalkers, iterations=iterations,
                                                sample_method="high_density", starting_samples=100, burn_frac=0.20,
                                                stopping_criteria="mode", stopping_criteria_factor=stopping_criteria_factor,
                                                stopping_criteria_debug=True, priors=bayes_settings.priors)
    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(tstart=tstart, tend=tend, dt=dt,
                             blackbox_settings=bayes_settings, data_context=data_context,
                             optimiser_context=optimiser_context,
                             plasma_context=plasma_context, model_context=model_context)

    workflow(pulse_to_write=43000000+pulse, run=run, mds_write=mds_write, plot=plot, filepath=f"./results/{dirname}/")


if __name__ == "__main__":

    pulse_info = [

        # [(11226,
        #   ["xrcs", "cxff_pi", "cxff_tws_c", "ts"],
        #   PARAMS_DEFAULT,
        #   [
        #       "xrcs.spectra",
        #       "cxff_pi.ti",
        #       "cxff_tws_c.ti",
        #       "ts.ne",
        #       "ts.te",
        #   ]),
        #  dict(
        #      phantom=True,
        #      tstart=0.05,
        #      tend=0.10,
        #      dt=0.01,
        #      iterations=5000,
        #      nwalkers=50,
        #      stopping_criteria_factor=0.002,
        #      mds_write=True,
        #      plot=True,
        #      run="RUN01",
        #      dirname=f"{11226}",
        #  )],

        # [(10009,
        # ["xrcs", "cxff_pi"],
        # PARAMS_DEFAULT,
        #   [
        #       "xrcs.spectra",
        #       "cxff_pi.ti",
        #   ]),
        #  dict(
        #      phantom=True,
        #      tstart=0.07,
        #      tend=0.08,
        #      dt=0.01,
        #      iterations=100,
        #      nwalkers=50,
        #      mds_write=True,
        #      plot=True,
        #      run="RUN01",
        #      dirname=f"{11336}_phantom",
        #  )],

        [(11089,
          ["xrcs", "cxff_tws_c", "ts", "efit"],
          PARAMS_SET_TS,
          [
              "xrcs.spectra",
              "cxff_tws_c.ti",
              "ts.ne",
              "ts.te",
          ]),
         dict(
             phantom=False,
             tstart=0.05,
             tend=0.10,
             dt=0.01,
             iterations=1000,
             nwalkers=20,
             stopping_criteria_factor=0.002,
             mds_write=True,
             plot=True,
             revisions={},
             run="RUN01",
             # dirname="test",
             set_ts=True,
         )],
        # RFX low ti/te pulse
    ]


    for info in pulse_info:
        print(f"pulse: {info[0][0]}")
        bda_run(*info[0], **info[1])