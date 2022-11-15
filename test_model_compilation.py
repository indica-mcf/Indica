import os

import cmdstanpy


def test_model_compilation():
    model_file = os.path.join("emissivity.stan")
    cmdstanpy.CmdStanModel(stan_file=model_file)


if __name__ == "__main__":
    test_model_compilation()
