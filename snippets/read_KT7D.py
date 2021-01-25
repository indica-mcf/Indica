import getdat as gd
import numpy as np


def get_kt7c(pulse, skip=None, trim40=False, backgrd=True, foregrd=False):

    if skip is None:
        skip = 5

    # get data
    nwds = 0
    data_node = "df/t7-spec:003"
    data, tvec, nwds, title, units, ier = gd.getdat(data_node, pulse, nwds=nwds)
    ndata = nwds

    # get time vector
    nwds_tim = 0
    tvec_tim, nwds_tim, ier_tim = gd.gettim(data_node, pulse, nwds=nwds_tim)

    # get number of frames
    nframe_node = "DF/T7-NFRAME<MPX:003"
    nframe, nwds, title, ier = gd.getraw(nframe_node, pulse, nwds=0)
    nframe = nframe[0]

    # get info on which pixels are in use
    pixon_node = "DF/T7-PIXON<MPX:003"
    pixon, nwds, title, ier = gd.getraw(pixon_node, pulse, nwds=0)
    pixoff_node = "DF/T7-PIXOFF<MPX:003"
    pixoff, nwds, title, ier = gd.getraw(pixoff_node, pulse, nwds=0)
    npixel = np.long((pixon - pixoff)[0]) + 1  # kt7a-c special case

    # check data dimensions
    if ndata > npixel * nframe:
        data = data[0 : npixel * nframe]
    elif ndata < npixel * nframe:
        nframe = ndata // npixel
        data = data[0 : npixel * nframe]

    # calculate exposure time
    treadout = tvec_tim[0:nframe]
    exp_time = treadout * 0.0
    exp_time[1:] = treadout[1:] - treadout[0:-1]
    exp_time[0] = exp_time[1]  # bald-faced lie (unknown exp for first frame)
    exp_time = exp_time.round(decimals=4)
    time = treadout - (exp_time / 2.0)

    # reshape data array
    shape = (nframe, npixel)
    data = np.reshape(data, shape)

    # keep only used pixels
    npixel = 1024
    pixel = np.arange(npixel)
    data = data[:, 0:npixel]

    # Calculate background or foreground:
    # - background: exclude first 5 frames & the 2 'move' frames just before 40s
    # - foreground: average last 20 frames
    # In both cases normalise to exp_time (can vary during the discharge)
    bkd = np.zeros(npixel)
    if backgrd and not foregrd:
        ind_bkd = np.argwhere(time < 40.0)[5:-2].flatten()
        bkd = []
        for i in ind_bkd:
            bkd.append(data[i, :] / exp_time[i])
        bkd = np.array(bkd).mean(axis=0).flatten()

    fgd = np.zeros(npixel)
    if foregrd:
        ind_fgd = range(nframe - 20, nframe)
        fgd = []
        for i in ind_fgd:
            fgd.append(data[i, :] / exp_time[i])
        fgd = np.array(fgd).mean(axis=0).flatten()

    # Normalise to exposure time (can vary during the discharge) and
    # subtract background/foreground
    for i in range(len(time)):
        data[i, :] = data[i, :] / exp_time[i] - bkd - fgd

    # Retain data only for time > 40 s
    if trim40:
        indx40 = np.argwhere(time > 40.0).flatten()
        data = data[indx40, :]
        time = time[indx40]
        treadout = treadout[indx40]
        exp_time = exp_time[indx40]

    # Moc-up wavelength calibration: pixel offset accounts for changes
    # in detector position
    if pulse >= 80685:
        pix_offs = +149
    elif pulse >= 80724:
        pix_offs = 0
    elif pulse >= 84931:
        pix_offs = -24
    elif pulse >= 89473:
        pix_offs = -2
    else:
        pix_offs = 0

    c = [0.0] * 3
    c[0] = 4.0351381
    c[1] = 0.0033944632
    c[2] = -3.4947697e-07

    # wavelength (nm)
    pix = pixel + pix_offs
    wave = c[0] + c[1] * pix + c[2] * pix ** 2

    out = {
        "data": data,
        "time": time,
        "treadout": treadout,
        "exptime": exp_time,
        "pixel": pixel,
        "wave": wave,
    }

    return out
