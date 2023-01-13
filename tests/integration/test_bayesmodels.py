from indica.bayesmodels import BayesModels, uniform

from indica.models.plasma import example_run
from indica.models.helike_spectroscopy import Helike_spectroscopy
from tests.integration.models.test_helike_spectroscopy import helike_LOS_example

import flatdict
import numpy as np
import emcee

class TestBayesModels():

    def setup_class(self):
        self.plasma = example_run()
        self.plasma.time_to_calculate = self.plasma.t[1]
        self.los_transform = helike_LOS_example(nchannels=1)
        self.los_transform.set_equilibrium(self.plasma.equilibrium)

    def test_simple_run_bayesmodels_with_xrcs(self):
        xrcs = Helike_spectroscopy(name="xrcs", )
        xrcs.plasma = self.plasma
        xrcs.set_los_transform(self.los_transform)

        priors = {
            "Te_prof_y0": lambda x: uniform(x, 500, 1e4),
            # "Te_prof_peaking": lambda x: uniform(x, 1, 10),
            "Ti_prof_y0": lambda x: uniform(x, 500, 2e4),
            # "Ti_prof_peaking": lambda x: uniform(x, 1, 10),
        }

        bckc = {}
        bckc = dict(bckc, **{xrcs.name: {**xrcs(calc_spectra=False)}})
        flat_phantom_data = flatdict.FlatDict(bckc, delimiter=".")

        bm = BayesModels(
            plasma=self.plasma,
            data=flat_phantom_data,
            diagnostic_models=[xrcs],
            quant_to_optimise=["xrcs.ti_w", "xrcs.te_kw"],
            priors=priors,
        )

        # Setup Optimiser
        params_names = [
            "Te_prof_y0",
            # "Te_prof_peaking",
            "Ti_prof_y0",
            # "Ti_prof_peaking",
        ]
        nwalk = 4

        Te_y0 = np.random.uniform(1000, 5e3, size=(nwalk,1,), )
        Te_peaking = np.random.uniform(2, 5, size=(nwalk,1,), )
        Ti_y0 = np.random.uniform(1000, 5e3, size=(nwalk,1,), )
        Ti_peaking = np.random.uniform(2, 5, size=(nwalk,1,), )

        start_points = np.concatenate([
                Te_y0,
                # Te_peaking,
                Ti_y0,
                # Ti_peaking,
            ], axis=1,)

        nwalkers, ndim = start_points.shape

        move = [emcee.moves.StretchMove()]
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn=bm.ln_posterior,
            parameter_names=params_names,
            moves=move,
        )
        sampler.run_mcmc(start_points, 10, progress=False)

if __name__ == "__main__":
    test = TestBayesModels()
    test.setup_class()
    test.test_simple_run_bayesmodels_with_xrcs()