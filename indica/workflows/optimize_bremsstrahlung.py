
import indica.physics as ph
from indica.readers.manage_data import initialize_bckc

def bremsstrahlung(
    self,
    data,
    bckc={},
    diagnostic="lines",
    quantity="brems",
    wavelength=532.0,
    cal=2.5e-5,
):
    """
    Estimate back-calculated Bremsstrahlung measurement from plasma quantities

    Parameters
    ----------
    data
        diagnostic data as returned by build_data()
    bckc
        back-calculated data
    diagnostic
        name of diagnostic usef for bremsstrahlung measurement
    quantity
        Measurement to be used for the bremsstrahlung
    wavelength
        Wavelength of measurement
    cal
        Calibration factor for measurement
        Default value calculated to match Zeff before Ar puff from
        LINES.BREMS_MP for pulse 9408

    Returns
    -------
    bckc
        dictionary with back calculated value(s)

    """
    zeff = self.zeff
    brems = ph.zeff_bremsstrahlung(
        self.el_temp, self.el_dens, wavelength, zeff=zeff.sum("element")
    )
    if diagnostic in data.keys():
        if quantity in data[diagnostic].keys():
            bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

            bckc[diagnostic][quantity].values = self.calc_los_int(
                data[diagnostic][quantity], brems * cal
            ).values

    bckc[diagnostic][quantity].attrs["calibration"] = cal
    return bckc

