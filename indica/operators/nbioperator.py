import os
from typing import List

from indica.operators.beam_utils import nbi_utils



PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))
print(f'PATH_TO_TE_FIDASIM = {PATH_TO_TE_FIDASIM}')


from .abstractoperator import Operator




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
        model = nbi_model
        model_key = str(model).strip().upper()
        model_handler = nbi_utils.get_model_handler(model_key)

        # Build per-time context dictionaries (profiles + equilibrium geometry).
        contexts = nbi_utils.build_nbi_contexts(
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
        # Execute the selected model for each time slice and collect results.
        neutrals_by_time = {}
        for ctx in contexts:
            result = model_handler(self, ctx)
            if isinstance(result, dict):
                neutrals_by_time.update(result)
            elif result is not None:
                neutrals_by_time[float(ctx["time"])] = result

        # Return all neutrals indexed by time.
        return neutrals_by_time
