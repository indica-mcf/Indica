from indica.models.charge_exchange_spectrometer import ChargeExchangeSpectrometer
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.helike_spectrometer import HelikeSpectrometer
from indica.models.interferometer import Interferometer
from indica.models.passive_spectrometer import PassiveSpectrometer
from indica.models.pinhole_camera import PinholeCamera
from indica.models.thomson_scattering import ThomsonScattering

MODELS_METHODS = {
    "get_interferometry": Interferometer,
    "get_charge_exchange": ChargeExchangeSpectrometer,
    "get_spectrometer": PassiveSpectrometer,
    "get_radiation": PinholeCamera,
    "get_thomson_scattering": ThomsonScattering,
    "get_helike_spectroscopy": HelikeSpectrometer,
    "get_diode_filters": BremsstrahlungDiode,
}

__all__ = [
    "PinholeCamera",
    "PassiveSpectrometer",
    "ChargeExchangeSpectrometer",
    "BremsstrahlungDiode",
    "EquilibriumReconstruction",
    "HelikeSpectrometer",
    "Interferometer",
    "ThomsonScattering",
]
