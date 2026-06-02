from __future__ import annotations

from typing import Optional

import numpy as np
from xarray import DataArray

from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.operators.abstract_nbioperator import NbiOperator


class NbiOperatorModel(AbstractDiagnostic):
    """Lightweight model wrapper for NBI operators (e.g. NbiFidasim).

    Mirrors the diagnostic-model pattern by allowing an attached Plasma object
    to provide the required profiles automatically.
    """

    def __init__(
        self,
        operator: NbiOperator,
        target_element: str = "d",
        file_name: str = "nbiop_model",
        pulse: int = 0,
        machine: str = "tokamak",
    ):
        self.operator = operator
        self.target_element = target_element
        self.file_name = file_name
        self.pulse = pulse
        self.machine = machine
        self.result: dict = {}
        self.bckc: dict = {}

    def set_transform(self, transform):
        super().set_transform(transform)
        self.operator.set_transform(transform)

    def _build_bckc_dictionary(self):
        self.bckc = self.result

    @staticmethod
    def _as_scalar_time(t) -> float:
        t_array = np.asarray(t)
        if t_array.size != 1:
            raise ValueError(
                "NBI operator model supports a single time point per call. "
                "Pass a scalar `t`."
            )
        return float(t_array.reshape(-1)[0])

    def __call__(
        self,
        t: Optional[float] = None,
        target_element: Optional[str] = None,
        file_name: Optional[str] = None,
        pulse: Optional[int] = None,
        machine: Optional[str] = None,
        prepare_kwargs: Optional[dict] = None,
        run_kwargs: Optional[dict] = None,
        Ti: Optional[DataArray] = None,
        Te: Optional[DataArray] = None,
        Ne: Optional[DataArray] = None,
        Nn: Optional[DataArray] = None,
        Vtor: Optional[DataArray] = None,
        Zeff: Optional[DataArray] = None,
        MeanZ: Optional[DataArray] = None,
    ) -> dict:
        if hasattr(self, "plasma"):
            if t is None:
                t = self.plasma.time_to_calculate
            if Ti is None:
                Ti = self.plasma.ion_temperature
            if Te is None:
                Te = self.plasma.electron_temperature
            if Ne is None:
                Ne = self.plasma.electron_density
            if Nn is None:
                Nn = self.plasma.neutral_density
            if Vtor is None:
                Vtor = self.plasma.toroidal_rotation
            if Zeff is None:
                Zeff = self.plasma.zeff
            if MeanZ is None:
                MeanZ = self.plasma.meanz

        missing = [
            name
            for name, value in (
                ("Ti", Ti),
                ("Te", Te),
                ("Ne", Ne),
                ("Nn", Nn),
                ("Vtor", Vtor),
                ("Zeff", Zeff),
                ("MeanZ", MeanZ),
                ("t", t),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Missing NBI inputs: "
                + ", ".join(missing)
                + ". Provide these directly or attach a Plasma with set_plasma()."
            )

        result = self.operator(
            Ti=Ti,
            Te=Te,
            Ne=Ne,
            Nn=Nn,
            Vtor=Vtor,
            Zeff=Zeff,
            MeanZ=MeanZ,
            target_element=target_element or self.target_element,
            t=self._as_scalar_time(t),
            file_name=file_name or self.file_name,
            pulse=pulse if pulse is not None else self.pulse,
            machine=machine or self.machine,
            prepare_kwargs=prepare_kwargs or {},
            run_kwargs=run_kwargs or {},
        )
        self.result = result
        self._build_bckc_dictionary()
        return self.bckc
