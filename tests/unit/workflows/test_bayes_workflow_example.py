import pytest
import copy
import numpy as np

from indica.workflows.bayes_workflow_example import (
    BayesWorkflowExample,
    DEFAULT_PRIORS,
    DEFAULT_PROFILE_PARAMS,
    OPTIMISED_PARAMS,
    OPTIMISED_QUANTITY,
)

"""
TODO:

test kwarg handling

Mock reader for testing experimental data methods
"""


class TestBayesWorkflowExample:
    def setup_class(self):
        self.init_settings = dict(
            pulse=None,
            phantoms = True,
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

         tsample=0.060,)


        self.opt_settings = dict(
            model_kwargs = {
                           "xrcs_moment_analysis": False,
                       },
        iterations = 1,
        nwalkers = 20,
        burn_frac = 0.10,
        sample_high_density = False,
        )

        self.call_settings = dict(
            filepath=None,
            pulse_to_write=23000101,
            run="RUN01",
            mds_write = False,
            plot = False,)

        self.untouched_workflow = BayesWorkflowExample(**self.init_settings)
        self.workflow = copy.deepcopy(self.untouched_workflow)

    def setup_method(self):
        self.workflow = copy.deepcopy(self.untouched_workflow)

    def teardown_method(self):
        self.workflow = None

    def test_workflow_initializes(self):
        attributes_to_check = [
            "data",
            "reader",
            "models",
            "equilibrium"
        ]
        for attribute in attributes_to_check:
            if not hasattr(self.workflow, attribute):
                raise ValueError(f"missing {attribute} in workflow object")
        assert True

    def test_init_phantoms_false_with_example_plasma(self):
        with pytest.raises(ValueError):
            example = BayesWorkflowExample(
                dict(self.default_settings, **{"phantoms": False})
            )

    def test_init_not_including_all_required_inputs(self):
        with pytest.raises(ValueError):
            example = BayesWorkflowExample(
                dict(self.default_settings, **{"param_names": None})
            )

    # def test_reader_has_read_all_diagnostic_data(self):
    #     assert all(diag_name in self.workflow.reader.keys() for diag_name in self.workflow.diagnostics)

    def test_plasma_has_equilibrium(self):
        self.workflow.setup_plasma()
        assert hasattr(self.workflow.plasma, "equilibrium")

    def test_phantom_profiles_are_not_mutatable(self):
        self.workflow.setup_plasma()
        phantoms = copy.deepcopy(self.workflow.phantom_profiles)
        self.workflow.plasma.electron_temperature += 1
        assert phantoms is not self.workflow.phantom_profiles

    def test_setup_models_with_wrong_diagnostic_names(self):
        with pytest.raises(Exception):
            self.workflow.setup_models(["foo", "bar", "xrcs"])

    def test_phantom_data_exists(self):
        self.workflow._phantom_data()
        assert self.workflow.opt_data

    # def test_experimental_data_exists(self):
    #     self.workflow._exp_data()
    #     assert self.workflow.opt_data

    def test_phantom_data_has_time_dim(self):
        self.workflow._phantom_data()
        for key, value in self.workflow.opt_data.items():
            assert "t" in value.dims

    # def test_experimental_data_has_time_dim(self):
    #     self.workflow._exp_data()
    #     for key, value in self.workflow.opt_data.items():
    #       assert "t" in value.dims

    def test_phantom_data_runs_with_noise_added(self):
        self.workflow._phantom_data(noise=True)
        assert self.workflow.opt_data

    def test_sampling_from_high_density(self):
        self.workflow._sample_start_points(sample_high_density=True)
        assert True

    def test_sampling_from_priors(self):
        self.workflow._sample_start_points(sample_high_density=False)
        assert True

    def test_worklow_has_results_after_run(self):
        self.workflow.setup_plasma()
        self.workflow.setup_opt_data(self.workflow.phantoms)
        self.workflow.setup_optimiser()
        self.workflow.run_sampler()
        if not hasattr(self.workflow, "result"):
            raise ValueError(f"missing result in workflow object")
        assert True


if __name__ == "__main__":
    test = TestBayesWorkflowExample()
    test.setup_class()
