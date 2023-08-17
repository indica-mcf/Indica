import copy
import sys
from unittest.mock import Mock

import pytest

from indica.workflows.bayes_workflow_example import BayesWorkflowExample
from indica.workflows.bayes_workflow_example import DEFAULT_PRIORS
from indica.workflows.bayes_workflow_example import DEFAULT_PROFILE_PARAMS
from indica.workflows.bayes_workflow_example import OPTIMISED_PARAMS
from indica.workflows.bayes_workflow_example import OPTIMISED_QUANTITY

"""
TODO:
Mock reader for testing experimental data reading
"""


class TestBayesWorkflowExample:
    def setup_class(self):
        self.init_settings = dict(
            pulse=None,
            phantoms=True,
            diagnostics=["xrcs", "efit", "smmh1", "cxff_pi"],
            opt_quantity=OPTIMISED_QUANTITY,
            param_names=OPTIMISED_PARAMS,
            profile_params=DEFAULT_PROFILE_PARAMS,
            priors=DEFAULT_PRIORS,
            tstart=0.02,
            tend=0.10,
            dt=0.005,
        )
        self.plasma_settings = dict(
            tsample=0.060,
        )

        self.optimiser_settings = dict(
            model_kwargs={
                "xrcs_moment_analysis": False,
            },
            nwalkers=20,
            sample_high_density=False,
        )

        self.call_settings = dict(
            filepath=None,
            pulse_to_write=23000101,
            run="RUN01",
            mds_write=False,
            plot=False,
            iterations=1,
            burn_frac=0.10,
        )
        self.sampler_settings = dict(
            iterations=1,
            burn_frac=0.10,
        )

        self.workflow_untouched = BayesWorkflowExample(**self.init_settings)
        self.workflow = None

    def setup_method(self):
        self.workflow = copy.deepcopy(self.workflow_untouched)

    def teardown_method(self):
        self.workflow = None

    def test_workflow_initializes(self):
        attributes_to_check = ["data", "reader", "models", "equilibrium"]
        for attribute in attributes_to_check:
            if not hasattr(self.workflow, attribute):
                raise ValueError(f"missing {attribute} in workflow object")
        assert True

    def test_init_phantoms_false_with_example_plasma(self):
        with pytest.raises(ValueError):
            BayesWorkflowExample(dict(self.init_settings, **{"phantoms": False}))

    def test_init_not_including_all_required_inputs(self):
        with pytest.raises(ValueError):
            BayesWorkflowExample(dict(self.init_settings, **{"param_names": None}))

    # def test_reader_has_read_all_diagnostic_data(self):
    #     assert all(diag_name in self.workflow.reader.keys()
    #     for diag_name in self.workflow.diagnostics)

    def test_plasma_has_equilibrium(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        assert hasattr(self.workflow.plasma, "equilibrium")

    def test_phantom_profiles_are_not_mutatable(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        phantoms = copy.deepcopy(self.workflow.phantom_profiles)
        self.workflow.plasma.electron_temperature += 1
        assert phantoms is not self.workflow.phantom_profiles

    def test_setup_models_with_wrong_diagnostic_names(self):
        with pytest.raises(ValueError):
            self.workflow.setup_models(["foo", "bar", "xrcs"])

    def test_opt_data_without_plasma(self):
        with pytest.raises(ValueError):
            self.workflow.setup_opt_data(phantoms=True)

    def test_phantom_data_exists(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(phantoms=True)
        assert self.workflow.opt_data

    # def test_experimental_data_exists(self):
    #     self.workflow._exp_data()
    #     assert self.workflow.opt_data

    def test_phantom_data_has_time_dim(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(phantoms=True)
        for key, value in self.workflow.opt_data.items():
            assert "t" in value.dims

    # def test_experimental_data_has_time_dim(self):
    #     self.workflow._exp_data()
    #     for key, value in self.workflow.opt_data.items():
    #       assert "t" in value.dims

    def test_phantom_data_runs_with_noise_added(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(phantoms=True, noise=True)
        assert self.workflow.opt_data

    def test_sampling_from_priors(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(
            phantoms=True,
        )
        self.workflow.setup_optimiser(
            **dict(self.optimiser_settings, **{"sample_high_density": False})
        )
        assert True

    def test_sampling_from_high_density(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(
            phantoms=True,
        )
        self.workflow.setup_optimiser(
            **dict(self.optimiser_settings, **{"sample_high_density": True})
        )
        assert True

    def test_worklow_has_results_after_run(self):
        self.workflow.setup_plasma(**self.plasma_settings)
        self.workflow.setup_opt_data(phantoms=True)
        self.workflow.setup_optimiser(**self.optimiser_settings)
        self.workflow.run_sampler(**self.sampler_settings)
        if not hasattr(self.workflow, "result"):
            raise ValueError("missing result in workflow object")
        assert True


if __name__ == "__main__":
    test = TestBayesWorkflowExample()
    test.setup_class()
