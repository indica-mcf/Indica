import random

import numpy as np
from deap import creator


def normal_from_hit_edge(angle_deg, transform):
    """
    Determine which edge of the rectangle is hit when casting a ray
    from the center at polar angle, and return the inward normal
    direction angle (deg).
 
    Rectangle is defined by transform._machine_dims:
        x ∈ [x0, x1], z ∈ [z0, z1]
    """
 
    x0, x1 = transform._machine_dims[0]
    z0, z1 = transform._machine_dims[1]
    cx, cz = 0.5*(x0+x1), 0.5*(z0+z1)
    ax, az = 0.5*(x1-x0), 0.5*(z1-z0)
 
    theta = np.deg2rad(angle_deg)
    dx, dz = np.cos(theta), np.sin(theta)
 
    # how far we need to travel to hit vertical/horizontal edges
    tx = ax / abs(dx) if dx != 0 else np.inf
    tz = az / abs(dz) if dz != 0 else np.inf
 
    if tx < tz:
        # hits a vertical edge
        if dx > 0:   # right edge
            return 180.0   # inward normal = -x
        else:        # left edge
            return 0.0     # inward normal = +x
    else:
        # hits a horizontal edge
        if dz > 0:   # top edge
            return 270.0   # inward normal = -z
        else:        # bottom edge
            return 90.0    # inward normal = +z

def _normalize_rects(rects):

    out = []

    for xmin, xmax, zmin, zmax in rects:

        if xmin > xmax: xmin, xmax = xmax, xmin

        if zmin > zmax: zmin, zmax = zmax, zmin

        out.append((xmin, xmax, zmin, zmax))

    return out

def _ray_hits_rect_2d(origin_xz, dir_xz, rect):

    ox, oz = origin_xz

    dx, dz = dir_xz

    xmin, xmax, zmin, zmax = rect

    # starts inside -> hit

    if xmin <= ox <= xmax and zmin <= oz <= zmax:

        return True

    inv_dx = np.inf if dx == 0.0 else 1.0/dx

    inv_dz = np.inf if dz == 0.0 else 1.0/dz

    t1x = (xmin - ox)*inv_dx; t2x = (xmax - ox)*inv_dx

    t1z = (zmin - oz)*inv_dz; t2z = (zmax - oz)*inv_dz

    tmin_x, tmax_x = (min(t1x,t2x), max(t1x,t2x))

    tmin_z, tmax_z = (min(t1z,t2z), max(t1z,t2z))

    t_enter = max(tmin_x, tmin_z)

    t_exit  = min(tmax_x, tmax_z)

    return t_exit >= max(t_enter, 0.0)

def generate_valid_pair_pool(transform, rects, *,

                             angle_step_deg=1.0,

                             offsets_per_angle= 13,

                             offset_kind="grid",   # "grid" or "random"

                             max_pairs=None,

                             rng=None):

    """

    Returns an array of shape (K, 2): columns = [angle_deg, offset].

    """

    rects = _normalize_rects(rects)

    if rng is None:

        rng = np.random.default_rng()
 
    angles = np.arange(0.0, 360.0, angle_step_deg, dtype=float)
 
    # optionally: skip left-edge origins (fast, conservative)

    # Uncomment if you want this pre-filter.

    cx, cz = 0.5*(transform._machine_dims[0][0] + transform._machine_dims[0][1]), 0.5*(transform._machine_dims[1][0] + transform._machine_dims[1][1])

    ax = 0.5*abs(transform._machine_dims[0][1] - transform._machine_dims[0][0])

    az = 0.5*abs(transform._machine_dims[1][1] - transform._machine_dims[1][0])

    alpha = np.degrees(np.arctan2(az, ax))

    mask = ~(((angles >= (180.0 - alpha)) & (angles <= (180.0 + alpha))))

    angles = angles[mask]
 
    # offsets to try per angle

    if offset_kind == "grid":

        offsets = np.linspace(-1.0, 1.0, offsets_per_angle)

    else:

        offsets = None  # sample random for each angle
 
    out = []

    for ang in angles:

        # origins on perimeter for this angle

        ox, oz = origin_from_polar_angle(ang, transform)
 
        # try offsets

        if offsets is None:

            offs = rng.uniform(-1.0, 1.0, size=offsets_per_angle)

        else:

            offs = offsets
 
        inward = (ang + 180.0) % 360.0

        for off in offs:

            # direction from angle+offset

            dx, dz = direction_from_polar_and_dir_offset(ang, off, transform)

            # if any rect hit, reject

            hit = any(_ray_hits_rect_2d((ox, oz), (dx, dz), R) for R in rects)

            if not hit:

                out.append((float(ang % 360.0), float(np.clip(off, -1.0, 1.0))))

                if max_pairs is not None and len(out) >= max_pairs:

                    return np.array(out, dtype=float)
 
    return np.array(out, dtype=float)

def save_pair_pool_csv(pairs, path):

    # columns: angle_deg, offset

    header = "angle_deg,offset"

    np.savetxt(path, pairs, delimiter=",", header=header, comments="", fmt="%.8f")

def random_angle():
    return random.uniform(0.0, 360.0)

def _ray_intersects_rect_2d(origin, direction, rect):
    """
    Ray (origin + t*dir, t>=0) vs axis-aligned rectangle in x–z.
    origin: (x,z), direction: (dx,dz), rect: (xmin,xmax,zmin,zmax)
    """
    ox, oz = float(origin[0]), float(origin[1])
    dx, dz = float(direction[0]), float(direction[1])
    xmin, xmax, zmin, zmax = rect
 
    # If starting inside, count as intersecting
    if (xmin <= ox <= xmax) and (zmin <= oz <= zmax):
        return True
 
    inv_dx = np.inf if dx == 0.0 else 1.0 / dx
    inv_dz = np.inf if dz == 0.0 else 1.0 / dz
 
    t1x = (xmin - ox) * inv_dx
    t2x = (xmax - ox) * inv_dx
    tmin_x, tmax_x = (min(t1x, t2x), max(t1x, t2x))
 
    t1z = (zmin - oz) * inv_dz
    t2z = (zmax - oz) * inv_dz
    tmin_z, tmax_z = (min(t1z, t2z), max(t1z, t2z))
 
    t_enter = max(tmin_x, tmin_z)
    t_exit  = min(tmax_x, tmax_z)
    return (t_exit >= max(t_enter, 0.0))

def _any_los_hits_rects(origins_xz, dirs_xz, rects):
    for (ox, oz), (dx, dz) in zip(origins_xz, dirs_xz):
        for rect in rects:
            if _ray_intersects_rect_2d((ox, oz), (dx, dz), rect):
                return True
    return False

def load_pair_pool_csv(path):
    print(f"Loading valid pairs from {path}")
    return np.loadtxt(path, delimiter=",", skiprows=1)

def make_individual_from_pool(pair_pool, n_los, rng=None):
    """
    Returns an Individual with genome [angles..., offsets...].
    Samples n_los rows from the pool (without replacement by default).
    """
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(pair_pool), size=n_los, replace=False)
    sel = pair_pool[idx]
    angles  = sel[:, 0]
    offsets = sel[:, 1]
    genome = np.concatenate([angles, offsets]).tolist()
    return creator.Individual(genome)

def obstacle_penalty_factor(individual, transform, rects):
    """
    Genome: first half = angles (deg), second half = dir_offsets [-1,1].
    rects: list of (xmin,xmax,zmin,zmax) rectangles in x–z.
    Returns 1.5 if any LOS intersects any rect, else 1.0.
    """
    g = np.asarray(individual, dtype=float)
    n = g.size // 2
    angles = (g[:n] % 360.0 + 360.0) % 360.0
    offsets = np.clip(g[n:], -1.0, 1.0)
 
    origins = np.empty((n, 2), dtype=float)
    dirs    = np.empty((n, 2), dtype=float)
    for i, (ang, off) in enumerate(zip(angles, offsets)):
        ox, oz = origin_from_polar_angle(ang, transform)           # (x,z)
        dx, dz = direction_from_polar_and_dir_offset(ang, off, transform)     # (dx,dz)
        origins[i] = (ox, oz)
        dirs[i]    = (dx, dz)
 
    return 2.5 if _any_los_hits_rects(origins, dirs, rects) else 1.0

def rotate_all(transform, t_min_deg):
    """
    Rotate arrays of origins (N,3) and directions (N,3) in the XY-plane
    by -t_min_deg degrees. Z is left unchanged.

    Parameters
    ----------
    origins : array-like, shape (N,3)
        Array of origins, each (x,y,0).
    directions : array-like, shape (N,3)
        Array of directions, each (dx,dy,0).
    t_min_deg : float
        Minimum angle in degrees; rotation is by -t_min_deg CCW.

    Returns
    -------
    origins_rot : np.ndarray, shape (N,3)
    directions_rot : np.ndarray, shape (N,3)
    """
    origins = transform.origin
    directions = transform.direction

    phi = np.deg2rad(-t_min_deg)  # CCW rotation by -t_min
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])  # 2x2 rotation matrix

    # apply rotation to x,y parts
    origins_xy = origins[:, :2] @ R.T
    dirs_xy = directions[:, :2] @ R.T

    # reassemble with z untouched
    origins_rot = np.column_stack([origins_xy, origins[:, 2]])
    directions_rot = np.column_stack([dirs_xy, directions[:, 2]])

    transform.set_origin(origins_rot)
    transform.set_direction(directions_rot)

def random_angle_test(transform):


    los_angles = np.array(
        [random_angle_avoiding_left(transform) for _ in range(8)]
    )
    los_angles=[59.63209549196224, 95.6216568731703, 266.8815031970812, 268.4670419152264, 308.5250168863993, 341.2164113018966, 342.5982764331459, 357.7043321523542]
    offsets=[-0.9581814903561077, 0.3645514201473765, 0.0, -0.1651885848622241, 0.3344766602479097, -0.41803296599972817, 0.18487535897572593, -0.0416023700388628]
    dirs=[]
    for angle, offset in zip(los_angles,offsets):
        dirs.append(direction_from_polar_and_dir_offset(angle,offset))

    #min_los_angle = np.min(los_angles)
    origin = transform.origin
    direction = transform.direction
    origin = np.delete(origin, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_origin(origin)
    direction = np.delete(direction, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_direction(direction)

    a=0
    for angle in los_angles:
        new_origin_x, new_origin_z = origin_from_polar_angle(angle, transform)
        transform.add_origin((new_origin_x, 0, new_origin_z))

        #new_dir_x, new_dir_z = random_feasible_direction_from_polar_angle(
        #    angle
        #)
        new_dir_x,new_dir_z=dirs[a]
        transform.add_direction((new_dir_x, 0, new_dir_z))
        a+=1

    #rotate_all(transform, min_los_angle)

    update_los(transform)

def _rect_center_and_extents(transform):
    x0 = transform._machine_dims[0][0]
    x1 = transform._machine_dims[0][1]
    z0 = transform._machine_dims[1][0]
    z1 = transform._machine_dims[1][1]
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    ax = 0.5 * (x1 - x0)  # half-width x
    az = 0.5 * (z1 - z0)  # half-height z
    return cx, cz, ax, az

def _ray_to_rect_boundary(angle_deg, transform):
    cx, cz, ax, az = _rect_center_and_extents(transform)
    th = np.deg2rad(angle_deg)
    ux, uz = np.cos(th), np.sin(th)  # unit direction in x–z
    eps = 1e-12
    tx = ax / (abs(ux) + eps)
    tz = az / (abs(uz) + eps)
    t = min(tx, tz)
    return cx + t * ux, cz + t * uz

def random_angle_avoiding_left(transform, ):

    # Half-extents
    x0, x1 = transform._machine_dims[0]
    z0, z1 = transform._machine_dims[1]
    ax = 0.5 * abs(x1 - x0)  # half-width
    az = 0.5 * abs(z1 - z0)  # half-height
 
    # Angular half-width of the left-edge exclusion
    alpha = np.degrees(np.arctan2(az, ax))  # deg
    start = 180.0 - alpha
    end   = 180.0 + alpha
 
    # Allowed set: [0, start) U (end, 360)
    width1 = max(0.0, start - 0.0)
    width2 = max(0.0, 360.0 - end)
    total = width1 + width2
 
    if random.random() < (width1 / total):
        angle = random.uniform(0.0, start)
    else:
        angle = random.uniform(end, 360.0)
    return angle

def random_feasible_direction_from_polar_angle(angle):

    inward_direction = (angle + 180.0) % 360.0
    direction_angle = inward_direction + random.uniform(-65.0, 65.0)
    th = np.deg2rad(direction_angle)
    return np.cos(th), np.sin(th)  # (dx, dz)

def direction_from_polar_and_dir_offset(angle, offset, transform):
    base = normal_from_hit_edge(angle, transform)  # axis-aligned normal
    direction_angle = base + 90.0 * offset
    return np.cos(np.deg2rad(direction_angle)), np.sin(np.deg2rad(direction_angle))

def origin_from_polar_angle(angle, transform):

    return _ray_to_rect_boundary(angle, transform)  # (x, z)

def update_los(transform):

    transform.x1 = list(np.arange(0, len(transform.origin)))

    transform.distribute_beamlets(debug=False)
    transform.set_dl(0.01)
    transform.convert_to_rho_theta()

def apply_individual_to_transform(individual, transform):
    """
    Genome layout: first half = direction offsets in [-1,1],
                   second half = angles in degrees [0,360).
    Recomputes ALL origins and directions and writes them to `transform`.
    """
    g = np.asarray(individual, dtype=float)
    n = g.size // 2
    dir_offsets = np.clip(g[:n], -1.0, 1.0)
    angles = (g[n:] % 360.0 + 360.0) % 360.0
 
    # Build origins (x,0,z) from angles
    origins = np.empty((n, 3), dtype=float)
    for i, ang in enumerate(angles):
        x, z = origin_from_polar_angle(ang, transform)
        origins[i] = (x, 0.0, z)
 
    # Build directions (dx,0,dz) from (angle, offset)
    directions = np.empty((n, 3), dtype=float)
    for i, (ang, off) in enumerate(zip(angles, dir_offsets)):
        dx, dz = direction_from_polar_and_dir_offset(ang, off, transform)
        directions[i] = (dx, 0.0, dz)
 
    # Replace all LOS and update
    transform.set_origin(origins)
    transform.set_direction(directions)
    update_los(transform)
 
    return transform, directions, origins


 


    """

    Interactive plot with a slider to explore timeslices.
 
    Parameters

    ----------

    phantom_emission : xarray.DataArray

        True emission, with dims including 't' and 'rhop'.

    downsampled_inverted : xarray.DataArray

        Reconstructed emission, with dims including 't' and 'rhop'.

    """

    # Extract time coordinates as a NumPy array

    t_vals = np.asarray(phantom_emission.t)

    nT = len(t_vals)
 
    # Initial index/time

    i0 = 0

    t0 = t_vals[i0]
 
    # Create figure and axis

    fig, ax = plt.subplots()

    plt.subplots_adjust(bottom=0.18)  # leave space for slider
 
    # Initial plot

    (line_phantom,) = ax.plot(

        phantom_emission.rhop,

        phantom_emission.sel(t=t0, method="nearest"),

        label="Phantom",

    )

    (line_recon,) = ax.plot(

        downsampled_inverted.rhop,

        downsampled_inverted.sel(t=t0, method="nearest"),

        linestyle="dashed",

        label="Reconstructed",

    )

    ax.set_xlabel("rhop")

    ax.set_ylabel("emission")

    ax.legend()
 
    # Fix initial y-limits (optional)

    ymin = float(np.nanmin([phantom_emission.min(), downsampled_inverted.min()]))
    ymax = float(np.nanmax([phantom_emission.max(), downsampled_inverted.max()]))
    ax.set_ylim(ymin, ymax * 1.05)  # +5% headroom

    ax.set_title(f"t = {t0}")
 
    # Add slider

    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.04])

    s_t = Slider(

        ax=ax_slider, label="t index",

        valmin=0, valmax=nT - 1,

        valinit=i0, valstep=1

    )
 
    # Update function

    def update(idx):

        idx = int(idx)

        tt = t_vals[idx]

        y_p = phantom_emission.sel(t=tt, method="nearest")

        y_r = downsampled_inverted.sel(t=tt, method="nearest")

        line_phantom.set_ydata(y_p)

        line_recon.set_ydata(y_r)

        ax.set_title(f"t = {tt}")

        fig.canvas.draw_idle()
 
    s_t.on_changed(update)
 
    plt.show()

def assert_valid_maximum_impact(transform):
    
    imp2=transform.impact_rho.sel(t=0,method="nearest")
    assert(np.max(imp2)<1.5)

def assert_valid_impact_params(transform):
    
    #imp=np.sort(transform.impact_parameter["dist"])
    imp2=transform.impact_rho.sel(t=0,method="nearest")
    imp2_s=np.sort(transform.impact_rho.sel(t=0,method="nearest"))
    assert(np.all(0.01<np.diff(imp2_s)))
