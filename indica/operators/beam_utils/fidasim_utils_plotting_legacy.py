"""Archived plotting/debug snippets previously in fidasim_utils.py.

These blocks are preserved for reference only and are not executed.
"""

PLOTTING_SNIPPETS = r"""
# Geometry plot for inspection
ax = None
if plot_geo:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    Rmaj = plt.Circle((0,0), 40, color='k', fill=False)
    Rsep = plt.Circle((0,0), 40+26, color='darkgrey', fill=False)
    ax.add_patch(Rmaj)
    ax.add_patch(Rsep)
    art3d.pathpatch_2d_to_3d(Rmaj, z=0, zdir="z")
    art3d.pathpatch_2d_to_3d(Rsep, z=0, zdir="z")

if plot_geo:
    plt.show()

debugging_shape = False
if debugging_shape:
    plt.figure()
    plt.imshow(mask)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.contourf(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        plasma['ti'],
    )
    plt.contour(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        plasma['ti'],
        [1.0*1e-3],
        colors='k',
    )
    plt.ylim([rmin, rmax])
    plt.xlim([zmin, zmax])
    plt.subplot(122)
    plt.contourf(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        rhogrid,
    )
    plt.contour(
        grid["z2d"][0, :],
        grid["r2d"][:, 0],
        rhogrid,
        [1.0],
        colors='k',
    )
    plt.ylim([rmin, rmax])
    plt.xlim([zmin, zmax])

    plt.figure()
    plt.subplot(131)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['br'],
    )
    plt.colorbar()
    plt.subplot(132)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['bz'],
    )
    plt.colorbar()
    plt.subplot(133)
    plt.contourf(
        grid['r2d'][:, 0],
        grid['z2d'][0, :],
        equil['bt'],
    )
    plt.colorbar()

    plt.show(block=True)

if (not block) and debug:
    plt.show(block=True)
else:
    plt.close('all')
"""
