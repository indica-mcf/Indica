from indica.workflows.bayes_workflow import BayesWorkflow, DEFAULT_PRIORS, DEFAULT_PROFILE_PARAMS, BayesSettings, ReaderSettings, \
    ExpData, MockData, PhantomData, PlasmaSettings, PlasmaContext, ModelContext, EmceeOptimiser, OptimiserEmceeSettings

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

OPT_DEFAULT = [
              "xrcs.spectra",
              "ts.ne",
              "ts.te",
              "efit.wp",
              "cxff_pi.ti",
              "cxff_tws_c.ti",
              # "smmh1.ne"
          ]


def bda_run(pulse, diagnostics, param_names, opt_quantity,
            tstart=0.02, tend=0.03, dt=0.01, revisions = {}, filters={},
            iterations=500, nwalkers=50, mds_write=False, plot=False,
            run="RUN01", **kwargs):

    # BlackBoxSettings
    bayes_settings = BayesSettings(diagnostics=diagnostics, param_names=param_names,
                                   opt_quantity=opt_quantity, priors=DEFAULT_PRIORS, )

    data_settings = ReaderSettings(filters=filters,
                                   revisions=revisions)

    data_context = ExpData(pulse=pulse, diagnostics=diagnostics,
                           tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    data_context.read_data()

    plasma_settings = PlasmaSettings(main_ion="h", impurities=("ar", "c"), impurity_concentration=(0.001, 0.04),
                                     n_rad=20)
    plasma_context = PlasmaContext(plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS)

    model_init_kwargs = {
        "cxff_pi": {"element": "ar"},
        "cxff_tws_c": {"element": "c"},
        "xrcs": {
            "window_masks": [slice(0.394, 0.396)],
        },
    }

    model_context = ModelContext(diagnostics=diagnostics,
                                 plasma_context=plasma_context,
                                 equilibrium=data_context.equilibrium,
                                 transforms=data_context.transforms,
                                 model_kwargs=model_init_kwargs,
                                 )

    optimiser_settings = OptimiserEmceeSettings(param_names=bayes_settings.param_names, nwalkers=nwalkers, iterations=iterations,
                                                sample_method="high_density", starting_samples=100, burn_frac=0.20,
                                                stopping_criterion="mode", stopping_criterion_factor=10,
                                                priors=bayes_settings.priors)
    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(tstart=tstart, tend=tend, dt=dt,
                             bayes_settings=bayes_settings, data_context=data_context,
                             optimiser_context=optimiser_context,
                             plasma_context=plasma_context, model_context=model_context)

    workflow(pulse_to_write=25000000+pulse, run=run, mds_write=mds_write, plot=plot, filepath=f"./results/{pulse}/")


if __name__ == "__main__":

    pulse_info = [

        [(11366,
         ["ts"],
         ["Te_prof.y0",
          "Te_prof.peaking",
          "Te_prof.wped",
          "Te_prof.wcenter",
          ],
         [
            # "xrcs.spectra",
            # "ts.ne",
            "ts.te",
            # "efit.wp",
            # "cxff_pi.ti",
            # "cxff_tws_c.ti",
            # "smmh1.ne"
        ]),
         dict(
             tstart=0.07,
             tend=0.08,
             dt=0.01,
             iterations=500,
             nwalkers=50,
             mds_write=True,
             plot=True,
             run="RUN01",
              )],

    ]

    for info in pulse_info:
        bda_run(*info[0], **info[1])