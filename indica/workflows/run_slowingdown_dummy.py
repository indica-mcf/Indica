import scipy.interpolate as interp
import slowingdown

amu2kg = 1.672e-27
eV2J = 1.602e-19

rhov = np.linspace(0.05, 0.95, 10)
Te = np.array(
    [
        3935.23408201,
        3788.92702285,
        3521.80277339,
        3184.75402347,
        2818.53606058,
        2440.74809717,
        2041.54215192,
        1590.61672625,
        1056.6714395,
        409.09603571,
    ]
)
ne = np.array(
    [
        7.13449907e19,
        6.99981218e19,
        6.76047271e19,
        6.46470768e19,
        6.14714559e19,
        5.80791621e19,
        5.38962101e19,
        4.75178926e19,
        3.68334779e19,
        1.85729312e19,
    ]
)
anum_plasma = 2
znum_plasma = 1
anum_beam = 2
znum_beam = 1
full_energy = 55e3
energy_frac = [full_energy, full_energy / 2, full_energy / 3]
power = 1e6
power_frac = [0.6, 0.3, 0.1]

# Rv, zv, rho2d, vol from equilibrium

location_hnbi = array([0.33704, 0.93884, 0.0])  # hnbi entrance
direction_hnbi = np.array([-0.704, -0.709, 0.0])

location_rfx = array([-0.341, -0.940, 0.0])  # rfx entrance
direction_rfx = np.array([0.704, 0.709, 0.0])

source = np.zeros((len(rhov), 3))

for i in range(3):
    source[:, i] = slowingdown.simulate_finite_source(
        rhov,
        ne,
        Te,
        anum_plasma,
        Rv,
        zv,
        rho2d,
        vol,
        location_hnbi,
        direction_hnbi,
        energy_frac[i],
        anum_beam,
        power,
        width=0.25,
        n=10,
    )

out = slowingdown.simulate_slowingdown(
    ne,
    Te,
    anum_plasma * amu2kg,
    znum_plasma * eV2J,
    energy_frac,
    source,
    anum_beam * amu2kg,
    znum_beam * eV2J,
    Nmc=10,
)

plt.plot(rhov, out["nfast"])
plt.plot(rhov, out["pressure"])
