import os
from typing import List

from indica.operators import adas_nbi_utils
from indica.operators import analytic_nbi_utils
from indica.operators import fidasim_utils
from indica.operators import nbi_utils



PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))
print(f'PATH_TO_TE_FIDASIM = {PATH_TO_TE_FIDASIM}')


from .abstractoperator import Operator

# Flow/Config map:
# 1) NBIOperator takes a transform + nbispecs (=beam specs).
# 2) nbispecs (from nbi_configs.DEFAULT_NBI_SPECS or test overrides) supplies
#    beam operating params (einj/pinj/current_fractions/ab).
# 3) fidasim_utils.prepare_fidasim builds FIDASIM inputs by combining:
#    - nbispecs (beam params; also picks beam name for geometry),
#    - plasmaconfig (equilibrium + profiles),
#    - global settings in nbi_configs.py (paths, MC settings, grids, switches),
#    - beam geometry from get_hnbi_geo/get_rfx_geo via create_st40_beam_grid.
# 4) Resulting inputs are written to FIDASIM_OUTPUT_DIR and run.



class NBIOperator(Operator):

    """This operator should be operating on a standard plasma+profiles, and spit out
        fast neutral density and fast particle pressure. I believe it does.
    """



    def __init__(
        self,
        name: str,
        einj: float,
        pinj: float,
        current_fractions: List[float],
        ab: float,
        pulse: int = None,
        plasma_ion_amu: float = 2.014,
    ):
        # Initialized with beam related info; transform is set later.
        self.transform = None
        self.pulse = pulse

        # NBI config
        self.name = name
        self.einj = einj
        self.pinj = pinj
        self.current_fractions = current_fractions
        self.ab = ab
        self.nbispecs = {
            "name": self.name,
            "einj": self.einj,
            "pinj": self.pinj,
            "current_fractions": self.current_fractions,
            "ab": self.ab,
        }

        self.plasma_ion_amu = plasma_ion_amu

    def __call__(
        self,
        nbi_model="FIDASIM",
        ion_temperature=None,
        electron_temperature=None,
        electron_density=None,
        neutral_density=None,
        toroidal_rotation=None,
        zeff=None,
        t=None,
        pulse: int = None,
        plasma=None,
    ) -> dict:




        # Resolve which NBI model runner to use for this call.
        model = nbi_model  or nbi_model
        model_key = str(model).strip().upper()
        model_handler = self._get_model_handler(model_key)

        # Build the context dictionary (profiles + equilibrium geometry).
        ctx = nbi_utils.build_nbi_context(
            self,
            ion_temperature=ion_temperature,
            electron_temperature=electron_temperature,
            electron_density=electron_density,
            neutral_density=neutral_density,
            toroidal_rotation=toroidal_rotation,
            zeff=zeff,
            t=t,
            pulse=pulse,
            plasma=plasma,
        )
        # Execute the selected model once and collect results.
        neutrals_by_time = {}
        result = model_handler(self, ctx)
        if isinstance(result, dict):
            neutrals_by_time.update(result)
        elif result is not None:
            neutrals_by_time[float(ctx["time"])] = result

        # Return all neutrals indexed by time.
        return neutrals_by_time

    def _get_model_handler(self, model_key: str):
        handlers = {
            "FIDASIM": fidasim_utils._run_fidasim,
            "ANALYTIC": analytic_nbi_utils._run_analytic,
            "ADAS": adas_nbi_utils._run_adas,
        }
        if model_key not in handlers:
            supported = ", ".join(handlers.keys())
            raise ValueError(
                f"Unknown nbi_model '{model_key}'. Supported models: {supported}"
            )
        return handlers[model_key]
