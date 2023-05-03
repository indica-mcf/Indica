import emcee
import flatdict

from indica.bayesmodels import BayesModels
from indica.bayesmodels import get_uniform
from indica.models.helike_spectroscopy import helike_LOS_example
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.plasma import example_run


class TestBayesModels:
    def setup_class(self):
        self.plasma = example_run()
        self.plasma.time_to_calculate = self.plasma.t[1]
        self.los_transform = helike_LOS_example(nchannels=1)
        self.los_transform.set_equilibrium(self.plasma.equilibrium)

    def test_simple_run_bayesmodels_with_xrcs(self):
        xrcs = Helike_spectroscopy(
            name="xrcs",
        )
        xrcs.plasma = self.plasma
        xrcs.set_los_transform(self.los_transform)

        priors = {
            "Te_prof.y0": get_uniform(2e3, 5e3),
            "Ti_prof.y0": get_uniform(2e3, 8e3),
        }

        bckc = {}
        bckc = dict(bckc, **{xrcs.name: {**xrcs(calc_spectra=True)}})
        flat_phantom_data = flatdict.FlatDict(bckc, delimiter=".")
        flat_phantom_data["xrcs.spectra"] = flat_phantom_data[
            "xrcs.spectra"
        ].expand_dims(dim={"t": [self.plasma.time_to_calculate]})

        bm = BayesModels(
            plasma=self.plasma,
            data=flat_phantom_data,
            diagnostic_models=[xrcs],
            quant_to_optimise=["xrcs.spectra"],
            priors=priors,
        )

        # Setup Optimiser
        param_names = [
            "Te_prof.y0",
            "Ti_prof.y0",
        ]

        ndim = param_names.__len__()
        nwalkers = ndim * 2
        start_points = bm.sample_from_priors(param_names, size=nwalkers)

        move = [emcee.moves.StretchMove()]
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn=bm.ln_posterior,
            parameter_names=param_names,
            moves=move,
            kwargs={},
        )
        sampler.run_mcmc(start_points, 10, progress=False)


if __name__ == "__main__":
    test = TestBayesModels()
    test.setup_class()
    test.test_simple_run_bayesmodels_with_xrcs()
