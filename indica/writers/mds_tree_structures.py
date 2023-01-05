import getpass


user = getpass.getuser()


def astra():
    psu_dict = {
        "I": ("SIGNAL", "Current, MA"),
        "V": ("SIGNAL", "Voltage, V"),
    }
    constraints_dict = {
        "INDEX": ("NUMERIC", "x vector(i) = i"),
        "CVALUE": ("SIGNAL", "simulated"),
        "MVALUE": ("SIGNAL", "experimental"),
        "WEIGHT": ("SIGNAL", ""),
    }
    nodes = {
        "CODE_VERSION": {
            "USER": ("TEXT", "User responsible of run"),
        },
        "": {
            "TIME": ("NUMERIC", "time vector, s"),
        },  # (type, description)
        ".GLOBAL": {
            "IPL": ("SIGNAL", "Plasma current, MA"),
            "BTVAC": ("SIGNAL", "BT_vacuum at R=0.5m, T"),
            "DF": ("SIGNAL", "simulated diamagnetic flux, Wb"),
            "CR0": ("SIGNAL", "MINOR RAD=(Rmax-Rmin)/2, m"),
            "RGEO": ("SIGNAL", "MAJOR R = (Rmax+Rmin)/2, m"),
            "ZGEO": ("SIGNAL", "Geom vertical position, m "),
            "RC": ("SIGNAL", "Current density center R_IP, m "),
            "ZC": ("SIGNAL", "Current density center Z_IP, m "),
            "ELON": ("SIGNAL", "ELONGATION BOUNDARY"),
            "TRIL": ("SIGNAL", "LOWER TRIANGULARITY"),
            "TRIU": ("SIGNAL", "UPPER TRIANGULARITY"),
            "QWL": ("SIGNAL", "Q(PSI) AT the LCFS"),
            "Q95": ("SIGNAL", "Q(PSI) AT 95% of full poloidal flux inside LCFS"),
            "TE0": ("SIGNAL", "Central electron temp, keV"),
            "TI0": ("SIGNAL", "Central ion temp, keV"),
            "NE0": ("SIGNAL", "Central electron density, 10E19m^-3 "),
            "NEL": ("SIGNAL", "Line aver electron density 10E19m^-3 "),
            "NEV": ("SIGNAL", "Volume aver electron density 10E19m^-3 "),
            "TEV": ("SIGNAL", "Volume aver electron temp, keV"),
            "TIV": ("SIGNAL", "Volume aver ion temp, keV"),
            "TAUE": ("SIGNAL", "Energy confinement time, s "),
            "P_OH": ("SIGNAL", "Total Ohmic power, MW"),
            "IEXC": ("SIGNAL", "Ion-electron exchange power, MW"),
            "UPL": ("SIGNAL", "Loop Voltage,V"),
            "WTH": ("SIGNAL", "Thermal energy, MJ"),
            "Li3": ("SIGNAL", "Internal inductance"),
            "BetP": ("SIGNAL", "Poloidal beta"),
            "BetT": ("SIGNAL", "Toroidal beta"),
            "BetN": ("SIGNAL", "Beta normalized  "),
            "Hoh": ("SIGNAL", "Neo-alcator H-factor"),
            "H98": ("SIGNAL", "ITER IPB(y,2) H-factor"),
            "HNSTX": ("SIGNAL", "NSTX scaling H-factor"),
            "HPB": ("SIGNAL", "Peter Buxton H-factor"),
            "ZEFF": ("SIGNAL", "Z effective at the plasma center"),
            "Res": ("SIGNAL", "Total plasma resistance Qj/Ipl^2, Ohm"),
            "Rmag": ("SIGNAL", "Magnetic axis hor position, m"),
            "Zmag": ("SIGNAL", "Magnetic axis vert position, m"),
            "Vol": ("SIGNAL", "Plasma volume, m^3"),
            "ROC": ("SIGNAL", "Effective plasma radius, m"),
            "P_NBI_E": ("SIGNAL", "Total power from NBI to electrons, MW"),
            "P_NBI_I": ("SIGNAL", "Total power from NBI to ions, MW"),
            "P_RF": ("SIGNAL", "Total RF power to electrons,MW"),
            "I_BS": ("SIGNAL", "Total bootstrap current,MA"),
            "F_BS": ("SIGNAL", "Bootstrap current fraction"),
            "I_NBI": ("SIGNAL", "Total NB driven current,MA"),
            "NBI_NAMES": ("SIGNAL", "NBI names, RFX,HNBI1=00,01,10,11"),
            "I_RF": ("SIGNAL", "Total RF driven current,MA"),
            "I_OH": ("SIGNAL", "Total Ohmic current,MA"),
            "F_NI": ("SIGNAL", "Non-inductive current fraction"),
            "P_FUS_THERM": ("SIGNAL", "Thermal fusion power,MW"),
            "P_FUS_TOT": ("SIGNAL", "Total fusion power: thermal+NBI,MW"),
            "P_AUX": ("SIGNAL", "Total external heating power,MW"),
            "Q_FUS": ("SIGNAL", "Fusion energy gain"),
            "P_TOT_E": ("SIGNAL", "Total alpha power to electrons,MW"),
            "P_TOT_I": ("SIGNAL", "Total alpha power to ions,MW"),
            "FBND": ("SIGNAL", "boundary poloidal flux,Wb"),
            "FAXS": ("SIGNAL", "axis poloidal flux,Wb"),
            "QE": ("SIGNAL", "electron power flux through LCFS, MW"),
            "QI": ("SIGNAL", "ion power flux through LCFS, MW"),
            "QN": ("SIGNAL", "electron flux through LCFS, 10^19/s"),
            "STOT": ("SIGNAL", "Total electron source, 10^19/s"),
            "SPEL": ("SIGNAL", "Pellet electron source, 10^19/s"),
            "SWALL": ("SIGNAL", "Boundary electron source, 10^19/s"),
            "SBM": ("SIGNAL", "Neutral beam electron source, 10^19/s"),
            "NNCL": ("SIGNAL", "Wall cold neutral density, 10^19/m^3"),
            "TAUP": ("SIGNAL", "Particle confinement time ,s"),
            "GAMMA": ("SIGNAL", "KINX growth rate, 1/s"),
            "VTOR0": ("SIGNAL", "Central toroidal velocity, m/s"),
            "TORQ": ("SIGNAL", "Total torque from NB, N*m"),
            "TORQ_BE": ("SIGNAL", "Collisional to electron torque from NB, N*m"),
            "TORQ_BI": ("SIGNAL", "Collisional to ions torque from NB, N*m"),
            "TORQ_BTH": ("SIGNAL", "Beam thermalisation torque from NB, N*m"),
            "TORQ_JXB": ("SIGNAL", "JXB torque from NB, N*m"),
            "TORQ_BCX": ("SIGNAL", "CX losses torque from NB, N*m"),
            "TAU_PHI": ("SIGNAL", "Momentum confinement time, s"),
        },
        ".PSI2D:": {
            "RGRID": ("NUMERIC", "Major radius coordinate m"),
            "ZGRID": ("NUMERIC", "Vertical coordinate m"),
            "PSI": ("SIGNAL", "Poloidal flux W"),
        },
        ".PSU.CS": psu_dict,
        ".PSU.MC": psu_dict,
        ".PSU.DIV": psu_dict,
        ".PSU.PSH": psu_dict,
        ".PSU.BVU": psu_dict,
        ".PSU.BVUT": psu_dict,
        ".PSU.BVUB": psu_dict,
        ".PSU.BVL": psu_dict,
        ".PSU.PSH": psu_dict,
        ".CONSTRAINTS.BP": constraints_dict,
        ".CONSTRAINTS.DF": constraints_dict,
        ".CONSTRAINTS.FLUX": constraints_dict,
        ".CONSTRAINTS.IP": constraints_dict,
        ".CONSTRAINTS.PFC_DOF": constraints_dict,
        ".CONSTRAINTS.PRESSURE": constraints_dict,
        ".CONSTRAINTS.ROGC": constraints_dict,
        ".CONSTRAINTS.ULOOP": constraints_dict,
        ".PROFILES.PSI_NORM": {
            "XPSN": ("NUMERIC", "x vector -sqrt(fi_normalized)"),
            "Q": ("SIGNAL", "Q_PROFILE(PSI_NORM)"),
            "P": ("SIGNAL", "PRESSURE(PSI_NORM)"),
            "PSI": ("SIGNAL", "PSI"),
            "PPRIME": ("SIGNAL", "PPRIME"),
            "FFPRIME": ("SIGNAL", "FFPRIME"),
            "FTOR": ("SIGNAL", "Toroidal flux, Wb"),
            "SIGMAPAR": ("SIGNAL", "Parallel conductivity,1/(Ohm*m)"),
            "AREAT": ("SIGNAL", "Toroidal cross section,m2"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m3"),
        },
        ".PROFILES.ASTRA": {
            "RHO": ("SIGNAL", "rho - toroidal flux coordinate, m"),
            "TE": ("SIGNAL", "Electron temperature, keV"),
            "NE": ("SIGNAL", "Electron density, 10^19 m^-3"),
            "NI": ("SIGNAL", "Main ion density, 10^19 m^-3"),
            "TI": ("SIGNAL", "Ion temperature, keV"),
            "ZEFF": ("SIGNAL", "Effective ion charge"),
            "QNBE": ("SIGNAL", "Beam power density to electrons, MW/m3"),
            "QNBI": ("SIGNAL", "Beam power density to ions, MW/m3"),
            "Q_OH": ("SIGNAL", "Ohmic heating power profile, MW/m3"),
            "STOT": ("SIGNAL", "Total electron source,10^19/s/m3"),
            "CHI_E": ("SIGNAL", "Total electron heat conductivity, m^2/s"),
            "CHI_I": ("SIGNAL", "Total ion heat conductivity, m^2/s"),
            "CHI_E_ANOM": ("SIGNAL", "anomalous electron heat conductivity, m^2/s"),
            "CHI_I_ANOM": ("SIGNAL", "anomalous ion heat conductivity, m^2/s"),
            "CHI_E_NEO": ("SIGNAL", "neoclassical electron heat conductivity, m^2/s"),
            "CHI_I_NEO": ("SIGNAL", "neoclassical ion heat conductivity, m^2/s"),
            "DIFF": ("SIGNAL", "diffusion coefficient, m^2/s"),
            "QE": ("SIGNAL", "electron power flux, MW"),
            "QI": ("SIGNAL", "ion power flux, MW"),
            "QN": ("SIGNAL", "total electron flux, 10^19/s"),
            "PEGN": ("SIGNAL", "electron convective heat flux, MW"),
            "PIGN": ("SIGNAL", "ion convective heat flux, MW"),
            "CC": ("SIGNAL", "Parallel current conductivity, 1/(Ohm*m),"),
            "ELON": ("SIGNAL", "Elongation profile"),
            "TRI": ("SIGNAL", "Triangularity (up/down symmetrized), profile"),
            "RMID": ("SIGNAL", "Centre of flux surfaces, m"),
            "RMINOR": ("SIGNAL", "minor radius, m"),
            "N_D": ("SIGNAL", "Deuterium density,10E19/m3"),
            "N_T": ("SIGNAL", "Tritium density	,10E19/m3"),
            "T_D": ("SIGNAL", "Deuterium temperature,keV"),
            "T_T": ("SIGNAL", "Tritium temperature,keV"),
            "Q_RF": ("SIGNAL", "RF power density to electron,MW/m3"),
            "Q_ALPHA_E": ("SIGNAL", "Alpha power density to electrons,MW/m3"),
            "Q_ALPHA_I": ("SIGNAL", "Alpha power density to ions,MW/m3"),
            "J_NBI": ("SIGNAL", "NB driven current density,MA/m2"),
            "J_RF": ("SIGNAL", " EC driven current density,MA/m2"),
            "J_BS": ("SIGNAL", "Bootstrap current density,MA/m2"),
            "J_OH": ("SIGNAL", "Ohmic current density,MA/m2"),
            "J_TOT": ("SIGNAL", "Total current density,MA/m2"),
            "PSIN": ("SIGNAL", "Normalized poloidal flux -"),
            "CN": ("SIGNAL", "Particle pinch velocity , m/s"),
            "SBM": ("SIGNAL", "Particle source from beam, 10^19/m^3/s "),
            "SPEL": ("SIGNAL", "Particle source from pellets, 10^19/m^3/s"),
            "SWALL": ("SIGNAL", "Particle source from wall neutrals, 10^19/m^3/s"),
            "OMEGA_TOR": ("SIGNAL", "Toroidal rotation frequency, 1/s"),
            "TORQ_DEN": ("SIGNAL", "Total torque density from NB, N*m/m3"),
            "TORQ_DEN_BE": (
                "SIGNAL",
                "Collisional to electron torque density from NB, N*m/m3",
            ),
            "TORQ_DEN_BI": (
                "SIGNAL",
                "Collisional to ions torque density from NB, N*m/m3",
            ),
            "TORQ_DEN_BTH": (
                "SIGNAL",
                "Beam thermalisation torque density from NB, N*m/m3",
            ),
            "TORQ_DEN_JXB": ("SIGNAL", "JXB torque density from NB, N*m/m3"),
            "TORQ_DEN_BCX": ("SIGNAL", "CX losses torque density from NB, N*m/m3"),
            "CHI_PHI": ("SIGNAL", "Momentum transport coefficient, m2/s"),
        },
        "P_BOUNDARY": {
            "INDEX": ("NUMERIC", ""),
            "RBND": ("SIGNAL", "R OF PLASMA_BOUNDARY"),
            "ZBND": ("SIGNAL", "Z OF PLASMA_BOUNDARY"),
        },
        "P_BOUNDARY.XPOINTS": {
            "FXPM": ("SIGNAL", "x-point poloidal flux, Wb"),
            "RXPM": ("SIGNAL", "r-position,m"),
            "ZXPM": ("SIGNAL", "z-position,m"),
            "ACTIVE": ("SIGNAL", "=1 if divertor"),
        },
        "P_BOUNDARY.LIMITER.VESSEL": {
            "INDEX": ("NUMERIC", ""),
            "R": ("NUMERIC", "r-position,m"),
            "Z": ("NUMERIC", "z-position,m"),
        },
        "P_BOUNDARY.LIMITER.PLASMA": {
            "ACTIVE": ("SIGNAL", "=1 if limiter"),
            "FBND": ("SIGNAL", "Limiter poloidal flux, Wb"),
            "R": ("SIGNAL", "r-position,m"),
            "Z": ("SIGNAL", "z-position,m"),
        },
        "P_BOUNDARY.TARGETS": {
            "RTORX": ("SIGNAL", "Geometric axis r-position from ASTRA exp file,m"),
            "ZX": ("SIGNAL", "Geometric axis z-position from ASTRA exp file,m"),
            "ELONGX": ("SIGNAL", "Elongation from ASTRA exp file"),
            "TRIANX": ("SIGNAL", "Triangularity from ASTRA exp file"),
            "ABCX": ("SIGNAL", "Minor radius at midplane from ASTRA exp file,m"),
        },
    }

    return nodes


def hda():

    # TODO: create upper and lower bounds for global quantities

    nodes = {
        "": {
            "TIME": ("NUMERIC", "time vector, s"),
        },  # (type, description)
        ".METADATA": {
            "USER": ("TEXT", "Username of owner"),
            "PULSE": ("NUMERIC", "Pulse number analysed"),
            "EQUIL": ("TEXT", "Equilibrium used"),
            "EL_DENS": ("TEXT", "Electron density diagnostic used for optimization"),
            "EL_TEMP": (
                "TEXT",
                "Electron temperature diagnostic used for optimization",
            ),
            "ION_TEMP": ("TEXT", "Ion temperature diagnostic used for optimization"),
            "STORED_EN": ("TEXT", "Stored energy diagnostic used for optimization"),
            "MAIN_ION": ("TEXT", "Main ion element"),
            "IMPURITY1": ("TEXT", "Impurity element chosen for Z1"),
            "IMPURITY2": ("TEXT", "Impurity element chosen for Z2"),
            "IMPURITY3": ("TEXT", "Impurity element chosen for Z3"),
        },
        ".GLOBAL": {
            "CR0": ("SIGNAL", "Minor radius = (R_LFS - R_HFS)/2 at midplane, m"),
            "RMAG": ("SIGNAL", "Magnetic axis R, m"),
            "ZMAG": ("SIGNAL", "Magnetic axis z, m"),
            "VOLM": ("SIGNAL", "Plasma volume z, m^3"),
            "IP": ("SIGNAL", "Plasma current, A"),
            "TE0": ("SIGNAL", "Central electron temp, eV"),
            "TI0": ("SIGNAL", "Central main ion temp, eV"),
            "TI0_Z1": ("SIGNAL", "Central impurity1 ion temp, eV"),
            "TI0_Z2": ("SIGNAL", "Central impurity2 ion temp, eV"),
            "TI0_Z3": ("SIGNAL", "Central impurity3 ion temp, eV"),
            "NE0": ("SIGNAL", "Central electron density, m^-3 "),
            "NI0": ("SIGNAL", "Central main ion density, m^-3 "),
            "TEV": ("SIGNAL", "Volume average electron temp, eV"),
            "TIV": ("SIGNAL", "Volume average ion temp, eV"),
            "NEV": ("SIGNAL", "Volume average electron density m^-3"),
            "NIV": ("SIGNAL", "Volume average main ion density m^-3"),
            "WP": ("SIGNAL", "Total stored energy, J"),
            "WTH": ("SIGNAL", "Thermal stored energy, J"),
            "UPL": ("SIGNAL", "Loop Voltage, V"),
            "P_OH": ("SIGNAL", "Total Ohmic power, W"),
            "ZEFF": ("SIGNAL", "Effective charge at the plasma center"),
            "ZEFFV": ("SIGNAL", "Volume averaged effective charge"),
            "CION": ("SIGNAL", "Average concentration of main ion"),
            "CIM1": ("SIGNAL", "Average concentration of impurity IMP1"),
            "CIM2": ("SIGNAL", "Average concentration of impurity IMP2"),
            "CIM3": ("SIGNAL", "Average concentration of impurity IMP3"),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": ("NUMERIC", "Radial vector, Sqrt of normalised poloidal flux"),
            "XPSN": ("NUMERIC", "x vector - fi_normalized"),
            "P": ("SIGNAL", "Pressure,Pa"),
            "P_HI": ("SIGNAL", "Pressure upper bound,Pa"),
            "P_LO": ("SIGNAL", "Pressure lower bound,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NE_HI": ("SIGNAL", "Electron density upper limit, m^-3"),
            "NE_LO": ("SIGNAL", "Electron density lower limit, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "NI_HI": ("SIGNAL", "Ion density upper limit, m^-3"),
            "NI_LO": ("SIGNAL", "Ion density lower limit, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TE_HI": ("SIGNAL", "Electron temperature upper limit, eV"),
            "TE_LO": ("SIGNAL", "Electron temperature lower limit, eV"),
            "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
            "TI_HI": ("SIGNAL", "Ion temperature of main ion upper limit, eV"),
            "TI_LO": ("SIGNAL", "Ion temperature of main ion lower limit, eV"),
            "TIZ1": ("SIGNAL", "Ion temperature of impurity IMP1, eV"),
            "TIZ1_HI": ("SIGNAL", "Ion temperature of impurity IMP1 upper limit, eV"),
            "TIZ1_LO": ("SIGNAL", "Ion temperature of impurity IMP1 lower limit, eV"),
            "TIZ2": ("SIGNAL", "Ion temperature of impurity IMP2, eV"),
            "TIZ2_HI": ("SIGNAL", "Ion temperature of impurity IMP2 upper limit, eV"),
            "TIZ2_LO": ("SIGNAL", "Ion temperature of impurity IMP2 lower limit, eV"),
            "TIZ3": ("SIGNAL", "Ion temperature of impurity IMP3, eV"),
            "TIZ3_HI": ("SIGNAL", "Ion temperature of impurity IMP3 upper limit, eV"),
            "TIZ3_LO": ("SIGNAL", "Ion temperature of impurity IMP3 lower limit, eV"),
            "NIZ1": ("SIGNAL", "Density of impurity IMP1, m^-3"),
            "NIZ1_HI": ("SIGNAL", "Density of impurity IMP1 upper limit, m^-3"),
            "NIZ1_LO": ("SIGNAL", "Density of impurity IMP1 lower limit, m^-3"),
            "NIZ2": ("SIGNAL", "Density of impurity IMP2, m^-3"),
            "NIZ2_HI": ("SIGNAL", "Density of impurity IMP2 upper limit, m^-3"),
            "NIZ2_LO": ("SIGNAL", "Density of impurity IMP2 lower limit, m^-3"),
            "NIZ3": ("SIGNAL", "Density of impurity IMP3, m^-3"),
            "NIZ3_HI": ("SIGNAL", "Density of impurity IMP3 upper limit, m^-3"),
            "NIZ3_LO": ("SIGNAL", "Density of impurity IMP3 lower limit, m^-3"),
            "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
            "NNEUTR_HI": ("SIGNAL", "Density of neutral main ion upper limit, m^-3"),
            "NNEUTR_LO": ("SIGNAL", "Density of neutral main ion lower limit, m^-3"),
            "ZI": ("SIGNAL", "Average charge of main ion, "),
            "ZI_HI": ("SIGNAL", "Average charge of main ion upper bound, "),
            "ZI_LO": ("SIGNAL", "Average charge of main ion lower bound, "),
            "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
            "ZIM1_HI": ("SIGNAL", "Average charge of impurity IMP1 upper bound, "),
            "ZIM1_LO": ("SIGNAL", "Average charge of impurity IMP1 lower bound, "),
            "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
            "ZIM2_HI": ("SIGNAL", "Average charge of impurity IMP2 upper bound, "),
            "ZIM2_LO": ("SIGNAL", "Average charge of impurity IMP2 lower bound, "),
            "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
            "ZIM3_HI": ("SIGNAL", "Average charge of impurity IMP3 upper bound, "),
            "ZIM3_LO": ("SIGNAL", "Average charge of impurity IMP3 lower bound, "),
            "ZEFF": ("SIGNAL", "Effective charge, "),
            "ZEFF_HI": ("SIGNAL", "Effective charge upper limit, "),
            "ZEFF_LO": ("SIGNAL", "Effective charge lower limit, "),
        },
        ".PROFILES.R_MIDPLANE": {
            "RPOS": ("NUMERIC", "Major radius position of measurement"),
            "ZPOS": ("NUMERIC", "Z position of measurement"),
            "P": ("SIGNAL", "Pressure,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
            "TIZ1": ("SIGNAL", "Ion temperature of impurity IMP1, eV"),
            "TIZ2": ("SIGNAL", "Ion temperature of impurity IMP2, eV"),
            "TIZ3": ("SIGNAL", "Ion temperature of impurity IMP3, eV"),
            "NIZ1": ("SIGNAL", "Density of impurity IMP1, m^-3"),
            "NIZ2": ("SIGNAL", "Density of impurity IMP2, m^-3"),
            "NIZ3": ("SIGNAL", "Density of impurity IMP3, m^-3"),
            "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
            "ZI": ("SIGNAL", "Average charge of main ion, "),
            "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
            "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
            "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
            "ZEFF": ("SIGNAL", "Effective charge, "),
        },
    }
    # "RHOT": ("SIGNAL", "Sqrt of normalised toroidal flux, xpsn"),

    return nodes
