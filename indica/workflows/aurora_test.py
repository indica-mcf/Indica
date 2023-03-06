import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
import aurora
from matplotlib import cm
from scipy.interpolate import interp1d, interp2d
from omfit_classes import omfit_eqdsk, omfit_gapy
from st40_phys_viewer.utility.make_geqdsk_MDSplus import make_geqdsk_MDSplus


def read_strahl(astra_shot=25009780, run=4, suffix=0):

    factor_astra = 1e13

    path_file = '/home/alsu.sladkomedova/ASTRA-STRAHL model results/{}_RUN{}/dat/'.format(astra_shot, run)
    fn10 = 'st40_9780hdaRUN60_eqAr.show_st40_strahl_ArC-9780.{}'.format(run+suffix)
    fn1 = path_file + fn10
    nskip = 13
    nint = 86
    d = pd.read_csv(fn1, skiprows=nskip, nrows=nint - nskip, delim_whitespace=True)

    for row in range(1, 8):
        d_new = pd.read_csv(fn1, skiprows=nskip+row*(nint-nskip+1), nrows=nint-nskip, delim_whitespace=True)
        # remove duplicate columns
        dup = [*set(d.keys()) & set(d_new.keys())]
        d_new = d_new.drop(columns=dup)  # Drop first instance of D1 and -V1

        d = pd.concat([d, d_new], axis=1)


    argon_names = ["Ar" + str(x) for x in range(0,19)]
    argon = d[argon_names]
    Dnc_names = ["D" + str(x) for x in range(1, 19)]
    Dnc_names[0] = "D1.1"
    Dnc = d[Dnc_names]
    Dnc.insert(0, "D0", Dnc.iloc[:,0].values*0)
    Vnc_names = ["-V" + str(x) for x in range(1, 19)]
    Vnc_names[0] = "-V1.1"
    Vnc = d[Vnc_names]
    Vnc.insert(0, "-V0", Vnc.iloc[:, 0].values*0)
    result = {"d":d,
             "argon":argon,
              "Dnc":Dnc,
              "Vnc":Vnc,
              }
    return result


def main(filehead):
    strahl_run = {
              #   1:return_strahl(25009780, 1, 0),
              # 2: return_strahl(25009780, 2, 0),
              #   3: return_strahl(25009780, 3, 0),
              #   4: return_strahl(25009780, 4, 0),
              #   5: return_strahl(25009780, 5, 0),
              #   6: return_strahl(25009780, 6, 0),
              #   7: return_strahl(25009780, 7, 0),
              #   8: return_strahl(25009780, 8, 0),
              #   9: return_strahl(25009780, 9, 0),
              10: read_strahl(33009780, 2, 8),
              # 11: return_strahl(33009780, 3, 8),
              #   12: return_strahl(33009780, 4, 8),
              # 13: return_strahl(33009780, 5, 8),
              #   14: return_strahl(33009780, 6, 8),
              # 15: return_strahl(33009780, 7, 8),
    }


    ##### read equilbrium
    filename, content = make_geqdsk_MDSplus(9780, 0.078, "EFIT#BEST")
    filepath = f"{filehead}/Downloads/eqdsks/" + filename
    with open(filepath, 'w') as f:
       f.write(content)
    geqdsk = omfit_eqdsk.OMFITgeqdsk(filepath)


    ##### Input dictionary for aurora sim
    namelist = aurora.load_default_namelist()
    inputgacode = omfit_gapy.OMFITgacode(f"{filehead}/python/Aurora/examples/example.input.gacode")
    kp = namelist['kin_profs']

    R_sep = geqdsk["RBBBS"].max()
    for key in strahl_run.keys():

        kp['Te']['rhop'] = kp['ne']['rhop'] = kp['n0']['rhop'] = kp['Ti']['rhop'] = np.sqrt(strahl_run[key]["d"]["psiN"]).values
        rhop = kp['Te']['rhop']
        kp['Te']['rhop'][0] = kp['ne']['rhop'][0] = kp['n0']['rhop'][0] = kp['Ti']['rhop'][0] = 0
        kp['ne']['vals'] = strahl_run[key]["d"]["ne"].values*1e13    # n19 --> cm^-3
        kp['n0']['vals'] = strahl_run[key]["d"]["Nh"].values*1e13    # n19 --> cm^-3
        kp['Te']['vals'] = strahl_run[key]["d"]["Te"].values*1e3
        kp['Ti']['vals'] = strahl_run[key]["d"]["Ti"].values*1e3

        imp = namelist['imp'] = 'Ar'
        imp = namelist['main_element'] = 'D'
        namelist["timing"]["times"] = np.array([0, 0.001, 1.0, 5.0])
        namelist["timing"]["dt_increase"] = np.array([1.0, 1.01, 1.05, 1.0])
        namelist["timing"]["steps_per_cycle"] = np.array([1, 1, 1, 1])
        namelist["timing"]["dt_start"] = np.array([1e-5, 1e-4, 1e-4, 1e-4])

        namelist["dr_0"] = 0.5
        namelist["dr_1"] = 0.1

        namelist["cxr_flag"] = True
        namelist["recycling_flag"] = True
        namelist["wall_recycling"]=1
        namelist["source_type"] = "const"
        namelist["source_rate"] = 0.5e19
        namelist["source_width_in"]= R_sep * 0.01
        namelist["source_width_out"]= R_sep * 0.01
        namelist["imp_source_energy_eV"]= 3
        namelist["source_cm_out_lcfs"]= R_sep * 0.001

        namelist["bound_sep"]=0
        namelist["lim_sep"]=0
        namelist["clen_limiter"]=1200
        namelist["clen_divertor"]=1200
        namelist["SOL_decay"]=1

        namelist["tau_div_SOL_ms"] = 1e4
        namelist["tau_pump_ms"] = 1e4
        namelist["SOL_mach"] = 0.5

        # namelist["saw_model"]["saw_flag"]=True
        # namelist["saw_model"]["rmix"]=20
        # namelist["saw_model"]["times"]= [2.0,]

        asim = aurora.aurora_sim(namelist, geqdsk=geqdsk)

        # Transport Settings
        charge_states = np.arange(0,19)
        D_z = (strahl_run[key]["d"]["Dara"].values[:,None] * 1e4) + (strahl_run[key]["Dnc"].values * 1e4)  # cm^2/s
        V_z = (strahl_run[key]["d"]["-Var"].values[:,None] * -1e2) + (strahl_run[key]["Vnc"].values * -1e2)

        D_z_interp = interp2d(charge_states, kp['Te']['rhop'], D_z)(charge_states, asim.rhop_grid)
        V_z_interp = interp2d(charge_states, kp['Te']['rhop'], V_z)(charge_states, asim.rhop_grid)
        # Add time in
        D_z_interp = np.expand_dims(D_z_interp, 1)
        V_z_interp = np.expand_dims(V_z_interp, 1)

        # # initial guess for steady state Ar charge state densities
        nz_init = strahl_run[key]["argon"].values * 1e13
        # Now get aurora setup
        nz_init = interp1d(rhop, nz_init, bounds_error=False, fill_value=0.0, axis=0)(
            asim.rhop_grid
        )

        out = asim.run_aurora(D_z_interp, V_z_interp, times_DV=np.array([1]), nz_init=nz_init)
        nz = out[0]  # charge state densities are stored first in the output of the run_aurora method

        fa_aurora = nz[:,:,-2]
        NAr = np.sum(fa_aurora, axis=1)

        plt.figure()
        plt.plot(rhop, strahl_run[key]["d"]['nAr'].values, label="STRAHL")
        plt.plot(asim.rhop_grid, NAr*1e-13, "--", label="Aurora")
        plt.title("Argon profile ")
        plt.xlabel("rho_poloidal")
        plt.yscale("log")
        plt.ylabel("NAr (n19)")
        plt.legend()
        # plt.plot(asim.rhop_grid, n_z_init[:,0] / n_z_init[:,0].max(), "x")

        nzar = 19
        color = [cm.nipy_spectral(x) for x in np.linspace(0, 1, nzar)]

        leg = []
        fig, ax = plt.subplots(figsize=(14,6))
        title = 'Ar fractional abundancies, run' + str(key)
        for i in range(nzar):
            p1,=ax.plot(rhop, strahl_run[key]["argon"]['Ar{}'.format(i)]/strahl_run[key]["d"]['nAr'].values,
                            c=color[i], label=i)
            ax.plot(asim.rhop_grid, fa_aurora[:,i]/NAr, "--", c=color[i], )
            leg.append(p1)
        ax.legend(handles=leg, bbox_to_anchor=[1.05, 1], ncol=2)
        ax.set_ylim(bottom=1e-3)
        ax.set_title(title)
        ax.set_xlabel('rho_poloidal')
        ax.set_ylabel('F.a.')
        # ax.set_yscale("log")
        plt.tight_layout()

        plt.figure()
        plt.plot(asim.rhop_grid, D_z_interp[:, 0, 16], label="RUN"+str(key) + "_D16")
        plt.plot(asim.rhop_grid, V_z_interp[:, 0, 16], label="RUN"+str(key) + "_V16")
        plt.title("Transport Coefficients")
        plt.xlabel("rho_poloidal")
        plt.ylabel(" (cm^-2)")
        plt.legend()


    # aurora.slider_plot(asim.rhop_grid, asim.time_out, nz.transpose(1,0,2),
    #                    xlabel='rho_pol [-]', ylabel='time [s]', zlabel='Charge State Densities',
    #                    labels=[str(i) for i in np.arange(0,nz.shape[1])],
    #                    plot_sum=True,  )


    #
    # leg=[]
    # fig,ax=plt.subplots(figsize=(14,6))
    # title='Ar fractional abundancies, 0.5Dan'
    # for i in range(nzar):
    #     p1,=ax.semilogy(psin1, argon_2['Ar{}'.format(i)]/nar2, c=color[i], label=i)
    #     leg.append(p1)
    # ax.legend(handles=leg, bbox_to_anchor=[1.05, 1], ncol=2)
    # ax.set_ylim(1e-3, 1)
    # ax.set_title(title)
    # ax.set_xlabel('psiN')
    # ax.set_ylabel('F.a.')
    # plt.tight_layout()
    # plt.savefig(save_path+title + '.png')
    #
    # leg=[]
    # fig,ax=plt.subplots(figsize=(14,6))
    # title='Ar fractional abundancies, 0.3Dan'
    # for i in range(nzar):
    #     p1,=ax.semilogy(psin1, argon_3['Ar{}'.format(i)]/nar3,c=color[i], label=i)
    #     leg.append(p1)
    # ax.legend(handles=leg, bbox_to_anchor=[1.05, 1], ncol=2)
    # ax.set_ylim(1e-3, 1)
    # ax.set_title(title)
    # ax.set_xlabel('psiN')
    # ax.set_ylabel('F.a.')
    # plt.tight_layout()
    # plt.savefig(save_path+title + '.png')
    plt.show(block=True)
    print()

if __name__=="__main__":
    main("/home/michael.gemmell")