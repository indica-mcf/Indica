import matplotlib.pyplot as plt
import numpy as np
import os
from MDSplus import *
import pickle
from scipy.stats import pearsonr
from scipy import interpolate

optimized_plot      = True
generate_data       = True
generate_corr       = True
generate_plot_data  = True
plot_evolution      = True
plot_correlation    = True

save_plot = True

ticksize = 15
labelsize = 20

# pulseNos = [9229]
pulseNos = [9184,9408,9409,9411,9539,9537,9560,9229,9538]
xlim = [0.01,0.15]

#PULSE TYPES
data_pulseTypes = dict(
    ohmic = dict(
        pulseNos = [9408,9409,9411,9184],
        color = 'r',
        symbol = 'v',
        ),
    NBI = dict(
        pulseNos = [9537,9538,9539,9560,9229],    
        color = 'b',
        symbol = 'o',
    ),
    )

#SAVE DIRECTORY
if optimized_plot:
    save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots_optimized/plots'
else:
    save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots/plots'

#FUNCTION TO GET SAVE DIRECTORY
def get_save_directory(folder = ''):
    os.chdir(save_directory_base)
    try:
        os.mkdir(folder)
    except:
        pass
    os.chdir(folder)
    return os.getcwd()

#NODE INFORMATION
infoNodes = dict(
    R           = dict(
        node    = ['EFIT.BEST.GLOBAL:RMAG','EFIT.BEST.GLOBAL:RGEO'],
        color   = ['red','purple'],
        norm    = 1.e-2,
        label   = 'R [cm]',
        labels  = ['Rmag','Rgeo'], 
        ylim    = None,
        title   = ['Rmag (EFIT)','Rgeo (EFIT)']
        ),
    Z           = dict(
        node    = ['EFIT.BEST.GLOBAL:ZMAG','EFIT.BEST.GLOBAL:ZGEO'],
        color   = ['purple','green','brown'],
        norm    = 1.e-2,
        label   = 'Z [cm]',
        labels  = ['Zmag','Zgeo','Zmag-Zshift'], 
        ylim    = None,
        title   = ['Zmag (EFIT)','Zgeo (EFIT)']
        ),
    AREA        = dict(
        node    = ['EFIT.BEST.GLOBAL:AREA'],
        color   = ['purple'],
        norm    = 1.e0,
        label   = 'Ap [m2]',
        labels  = [], 
        ylim    = None,
        title   = ['Aplasma (EFIT)']
        ),
    VOLUME      = dict(
        node    = ['EFIT.BEST.GLOBAL:VOLM'],
        color   = ['green'],
        norm    = 1.e0,
        label   = 'Vp [m3]',
        labels  = [], 
        ylim    = None,
        title   = ['Vplasma (EFIT)']
        ),
    AMINOR      = dict(
        node    = ['EFIT.BEST.GLOBAL:CR0'],
        color   = ['red'],
        norm    = 1.e-2,
        label   = 'AMINOR\n[cm]',
        labels  = [], 
        ylim    = None,
        title   = ['Aminor (EFIT)']
        ),        
    ELON        = dict(
        node    = ['EFIT.BEST.GLOBAL:ELON'],
        color   = ['purple','green'],
        norm    = 1.e0,
        label   = '$\\kappa$ [no unit]',
        labels  = [], 
        ylim    = None,
        title   = ['$\\kappa$ (EFIT)']
        ),
    PSI         = dict(
        node    = ['EFIT.BEST.GLOBAL:FBND','EFIT.BEST.GLOBAL:FAXS'],
        color   = ['blue','red'],
        norm    = 1.e-2,
        label   = 'psi\n[Wb/2pi]',
        labels  = ['(x10$^{-2}$) @ boundary','(x10$^{-2}$) @ magnetic axis'], 
        ylim    = None,
        title   = ['psi(boun) (EFIT)','psi(0) (EFIT)']
        ),
    BETA        = dict(
        node    = ['EFIT.BEST.VIRIAL:BTPM','EFIT.BEST.VIRIAL:BTPD'],
        color   = ['blue','GREEN'],
        norm    = 1.e0,
        label   = 'betaP',
        labels  = ['betaP(3) MHD','diamagnetic'], 
        ylim    = None,
        title   = ['beta (EFIT)']
        ),
    R_PFIT      = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:RGEO','PFIT.POST_BEST.RESULTS.GLOBAL:RIP'],
        color   = ['red','purple'],
        norm    = 1.e-2,
        label   = 'R [cm]',
        labels  = ['Rgeo','RIp'], 
        ylim    = None,
        title   = ['Rgeo (PFIT)','RIp (PFIT)']
        ),
    Z_PFIT      = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:ZGEO','PFIT.POST_BEST.RESULTS.GLOBAL:ZIP'],
        color   = ['red','green'],
        norm    = 1.e-2,
        label   = 'Z [cm]',
        labels  = ['Zgeo','ZIp'], 
        ylim    = None,
        title   = ['Zgeo (PFIT)','ZIp (PFIT)']
        ),
    AMINOR_PFIT = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:AMINOR'],
        color   = ['red'],
        norm    = 1.e-2,
        label   = 'AMINOR\n[cm]',
        labels  = [], 
        ylim    = None,
        title   = ['Aminor (PFIT)']
        ),        
    VOLUME_PFIT = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:VPLASMA'],
        color   = ['green'],
        norm    = 1.e0,
        label   = 'Vp [m3]',
        labels  = [], 
        ylim    = None,
        title   = ['Vplasma (PFIT)']
        ),
    ELON_PFIT   = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:KAPPA'],
        color   = ['purple'],
        norm    = 1.e0,
        label   = '$\\kappa$ [no unit]',
        labels  = [], 
        ylim    = None,
        title   = ['$\\kappa$ (PFIT)']
        ),
    BETA_PFIT   = dict(
        node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:BETAP2'],
        color   = ['blue','GREEN','purple'],
        norm    = 1.e0,
        label   = 'betaP',
        labels  = [], 
        ylim    = None,
        title   = ['betaP2 (PFIT)']
        ),             
    SXR_SP      = dict(
        node    = ['SXR.DIODE_DETR.BEST.FILTER_001:SIGNAL'],
        color   = ['brown'],
        norm    = 1.e+3,
        label   = 'SXR',
        labels  = ['DIODE_DETR:FILTER001 (Single point) [kW/m2]'],       
        ylim    = None,
        title   = ['SXR SP - Filter 1'],
        ),
    IP          = dict(
        node    = ['EFIT.BEST.CONSTRAINTS.IP:CVALUE','PFIT.POST_BEST.RESULTS.GLOBAL:IP'],
        color   = ['blue','red'],
        norm    = 1.e+3,
        label   = 'Ip [MA]',
        labels  = ['PFIT#POST_BEST','EFIT#BEST'], 
        ylim    = None,
        title   = ['Iplasma (PFIT)']
        ),
    LINE_DIODE  = dict(
        node    = ['SPECTROM.LINES.NOFILTER_MP1:INTENSITY'],
        color   = ['green'],
        norm    = 1.e0,
        label   = 'Line Diode [a.u.]',
        labels  = [], 
        ylim    = None,
        title   = ['Line diode (NOFILTER_MP1)']
        ),    
    NE         = dict(
        node    = ['INTERFEROM.SMMH1_SCALE.BEST.LINE_AV:NE_EFIT','INTERFEROM.NIRH1_BIN.BEST.LINE_AV:NE'],
        color   = ['brown','red'],
        norm    = 1.e+19,
        label   = '<ne>\n[10$^{19}$ m$^{-3}$]',
        labels  = ['SMMH','NIRH'],    
        ylim    = None,
        title   = ['<ne> (SMMH)','<ne> (NIRH)']
        ), 
    TE          = dict(
        node    = ['SXR.XRCS.BEST.TE_'+x+':TE' for x in ['KW','N3W']],
        color   = ['brown','purple'],
        norm    = 1.e+3,
        label   = 'Te [keV]',
        labels  = ['Te ('+x+')' for x in ['KW','N3W']],    
        ylim    = None,
        title   = ['Te ('+x+')' for x in ['KW','N3W']],    
        ), 
    VLOOP       = dict(
        node    = ['MAG.BEST.FLOOP.L016:V'],
        color   = ['red'],
        norm    = 1.e+0,
        label   = 'Vloop [V]',
        labels  = [],    
        ylim    = None,
        title   = ['Vloop 016 (MAG#BEST)'],    
        ), 
    W           = dict(
        node    = ['EFIT.BEST.VIRIAL:WP'],
        color   = ['blue','red','green'],
        norm    = 1.e+3,
        label   = 'W$_{MHD}$ [kJ]',
        labels  = ['EFIT#BEST'], 
        ylim    = None,
        title   = ['W$_{MHD}$ (EFIT)']
        ),  
    LI          = dict(
        node    = ['EFIT.BEST.VIRIAL:LI3M'],
        color   = ['brown','purple'],
        norm    = 1.e+0,
        label   = 'li',
        labels  = ['EFIT#BEST'], 
        ylim    = None,
        title   = ['l$_{i}$ (EFIT)']
        ),  
    Q           = dict(
        node    = ['EFIT.BEST.PROFILES.PSI_NORM:Q','EFIT.BEST.GLOBAL:Q95'],
        color   = ['brown','purple'],
        norm    = 1.e+0,
        label   = 'q',
        labels  = ['q$_{0}$ (EFIT)','q$_{95}$ (EFIT)'], 
        ylim    = None,
        title   = ['q$_{0}$ (EFIT)','q$_{95}$ (EFIT)']
        ),  
    SXR_CAM     = dict(
        node    = ['SXR.DIODE_ARRAYS.BEST.MIDDLE_HEAD.FILTER_4:CH0'+str(x) for x in range(61,81)],
        color   = ['#440154','#481567','#482677','#453781','#404788','#39568C','#33638D','#2D708E','#287D8E','#238A8D','#1F968B','#20A387','#29AF7F','#3CBB75','#55C667','#73D055','#95D840','#B8DE29','#DCE319','#FDE725'],
        norm    = 1.e+3,
        label   = 'SXR\n[kW/m2]',
        labels  = [],     
        ylim    = None,
        title   = ['CH0'+str(x) for x in range(61,81)],
        ),
    CHI2        = dict(
        label   = '$\\chi^2$ [no unit]',
        labels  = [],     
        ylim    = [0,2],
        color   = ['blue','red'],
        title   = ['$\\chi^2$'],
        ),
    CHI_EFIT    = dict(
        node    = ['EFIT.BEST.GLOBAL:'+x for x in ['CHIT','CHIM']],
        color   = ['brown','purple'],
        norm    = 1.e+0,
        label   = 'chi2',
        labels  = ['chi2$_{tot}$ (EFIT)','chi2$_{mag}$ (EFIT)'], 
        ylim    = None,
        title   = ['chi2$_{tot}$ (EFIT)','chi2$_{mag}$ (EFIT)']
        ),  
)

#GETTING THE DATA FROM THE NODES
def get_data(nodes,conn,xlim):
    #RETURN DATA DECLARATION
    time = ()
    data = ()
    #SWEEP OF NODES
    for node in nodes:
        temp_data = conn.get(node).data()
        temp_time = conn.get('dim_of('+node+')').data()
        if ('RIP' in node) | ('ZIP' in node):
            node2 = 'PFIT.POST_BEST.RESULTS.GLOBAL:IP'
            temp_data = temp_data / interpolate.interp1d(conn.get('dim_of('+node2+')').data(),conn.get(node2).data())(temp_time)
        if 'PSI_NORM:Q' in node:
            temp_data = temp_data[0,:]
        #CUTTING THE DATA
        sel_map = (temp_time>=xlim[0])&(temp_time<=xlim[1])
        #SELECTED DATA
        time += (temp_time[sel_map],)
        data += (temp_data[sel_map],)    
    #RETURNING TIME AND DATA
    return time,data

#GENERATING DATA
if generate_data:
    #RETURN DATA
    pulse_data = {}
    #OPENING MDSPLUS CONNECTION
    conn = Connection('192.168.1.7')
    #PULSE TYPES DECLARATION
    pulse_data['pulseType'] = []
    #FIELDS DECLARATION
    pulse_data['fields'] = {}
    #SWEEP OF PULSES
    for ipulse,pulseNo in enumerate(pulseNos):    
        #PULSE DATA
        pulse_data[str(pulseNo)] = {}
        #OPENING ST40 TREE
        conn.openTree('ST40',pulseNo)
        #SWEEP OF ITEMS
        for key,value in infoNodes.items():
            #DECLARATION
            pulse_data[str(pulseNo)][key] = {}
            #DATA
            if key!='CHI2':
                pulse_data[str(pulseNo)][key]['TIME'],pulse_data[str(pulseNo)][key]['DATA'] = get_data(value['node'],conn,xlim)
                #CHANGING THE KEY OF Z
                if (key=='Z')&(optimized_plot):
                    addTitle = 'Zmag(EFIT)-Zshift'
                    if addTitle not in value['title']:
                        value['title'] += [addTitle]
            else:
                if optimized_plot:
                    filename = save_directory_base+'/../'+str(pulseNo)+'/optimize_plots/optimized_'+str(pulseNo)+'.p'
                else:
                    filename = save_directory_base+'/../'+str(pulseNo)+'/'+str(pulseNo)+'.p'
                inv_data_all = pickle.load(open(filename,'rb'))
                inv_data = inv_data_all['filter_4']
                pulse_data[str(pulseNo)][key]['TIME'] = inv_data['t']
                pulse_data[str(pulseNo)][key]['DATA'] = inv_data['back_integral']['chi2']
                if optimized_plot:
                    pulse_data[str(pulseNo)][key]['DATA0']   = inv_data_all['all_results']['sweep_value_'+str(np.where(inv_data_all['results_optimize']['z_shifts']==0)[0][0]+1)]['filter_4']['back_integral']['chi2']
                    pulse_data[str(pulseNo)][key]['ZSHIFT'] = inv_data['z_shift']
            #PLOT DATA
            pulse_data[str(pulseNo)][key]['PLOT_DATA'] = value
            #FIELDS
            if ipulse==0:
                pulse_data['fields'][key] = value['title']
        #ZMAG-ZSHIFT DATA
        if optimized_plot:
            key = 'Z'
            t = pulse_data[str(pulseNo)]['CHI2']['TIME']
            Zmag = interpolate.interp1d(pulse_data[str(pulseNo)][key]['TIME'][0],pulse_data[str(pulseNo)][key]['DATA'][0])(t)
            pulse_data[str(pulseNo)][key]['TIME'] += (t,)
            pulse_data[str(pulseNo)][key]['DATA'] += (Zmag-pulse_data[str(pulseNo)]['CHI2']['ZSHIFT'],)
        #PULSETYPE
        for pulseType,pulseData in data_pulseTypes.items():
            if pulseNo in pulseData['pulseNos']:
                pulse_data['pulseType'] += [pulseType]
    #OTHER INFORMATION
    pulse_data['xlim'] = xlim
    pulse_data['pulseNos'] = pulseNos
    pulse_data['quantities'] = list(infoNodes.keys())
    #SAVING THE DATA
    filename = save_directory_base + '/pulse_data.p'
    with open(filename,'wb') as handle:
        pickle.dump(pulse_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(filename+' is saved!')
else:
    #LOADING THE DATA
    filename = save_directory_base + '/pulse_data.p'
    with open(filename,'rb') as handle:
        pulse_data = pickle.load(handle)
    print(filename+' is loaded!')

#PERFORMING CORRELATION
if generate_corr:
    #DECLARATION
    corr_data = {}
    #EXCLUDE IN CORRELATION
    exclude = ['CHI2']
    #SWEEEP OF QUANTITIES
    for quantity in pulse_data['quantities']:
        if quantity not in exclude:
            #DECLARATION
            corr_data[quantity] = {}
            #SWEEP OF FIELDS INSIDE THE QUANTITY
            for ifield,field in enumerate(pulse_data['fields'][quantity]):
                #DATA DECLARATION
                corr_data[quantity][field] = np.nan * np.ones(len(pulse_data['pulseNos']))
                #SWEEP OF PULSES
                for ipulse,pulseNo in enumerate(pulse_data['pulseNos']):
                    #SELECTED DATA
                    sel_data = pulse_data[str(pulseNo)]      
                    x = interpolate.interp1d(sel_data[quantity]['TIME'][ifield],sel_data[quantity]['DATA'][ifield],bounds_error=False)(sel_data['CHI2']['TIME'])                
                    y = sel_data['CHI2']['DATA']
                    map_sel = np.isfinite(x) & np.isfinite(y)
                    sel_x = x[map_sel]
                    sel_y = y[map_sel]
                    if np.sum(map_sel)>2:
                        corr_data[quantity][field][ipulse] = pearsonr(sel_x,sel_y)[0]
    #SAVING THE DATA
    filename = save_directory_base + '/corr_data.p'
    with open(filename,'wb') as handle:
        pickle.dump(corr_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(filename+' is saved!')
else:
    #LOADING THE DATA
    filename = save_directory_base + '/corr_data.p'
    with open(filename,'rb') as handle:
        corr_data = pickle.load(handle)
    print(filename+' is loaded!')     

#GENERATE PLOT DATA
if generate_plot_data:
    #PLOT DATA
    data_plot_evolution = dict(        
        #EQUILIBRIUM FROM EFIT
        Equilibrium_EFIT = [
            'R', 'Z',
            'AREA', 'VOLUME',
            'AMINOR', 'ELON',
            'PSI', 'BETA',
            'W','CHI_EFIT'
            ],
        #EQUILIBRIUM FROM PFIT
        Equilibrium_PFIT = [
            'R_PFIT', 'Z_PFIT',
            'AMINOR_PFIT', 'VOLUME_PFIT',
            'ELON_PFIT', 'BETA_PFIT'
            ],
        #MHD PLOTS
        MHD = [
            'NE', 'SXR_SP',
            'IP', 'LINE_DIODE'
            ],
        #PLASMA PARAMETERS
        Plasma_parameters = [
            'NE', 'TE',
            'VLOOP', 'IP',
            'Q', 'LI',
            'VOLUME','W'
            ],
        )
    #SAVING THE DATA
    filename = save_directory_base + '/plot_data.p'
    with open(filename,'wb') as handle:
        pickle.dump(data_plot_evolution,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(filename+' is saved!')
else:
    #LOADING THE DATA
    filename = save_directory_base + '/plot_data.p'
    with open(filename,'rb') as handle:
        data_plot_evolution = pickle.load(handle)
    print(filename+' is loaded!')    


#EVOLUTION PLOTS
if plot_evolution:    
    #CLOSING ALL THE FIGURES
    plt.close('all')    
    #SWEEP OF PLOTS
    for title,plot_list in data_plot_evolution.items():        
        #SAVE DIRECTORY
        save_directory = get_save_directory(title.lower()+'_plots')
        #SWEEP OF PULSES
        for ipulse,pulseNo in enumerate(pulse_data['pulseNos']):
            #SUBPLOT DECLARATION
            no_rows = int(len(plot_list)/2) + 1
            no_cols = 2            
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(18,10),sharex=True)
            #COUNT VARIABLE
            k = 0            
            #SWEEP OF ROWS
            for i in range(0,no_rows):
                for j in range(0,no_cols):
                    #QUANTITY
                    if i<=no_rows-2:
                        quantity = plot_list[k]
                    else:
                        quantity = 'SXR_CAM' if j==0 else 'CHI2'
                    #DATA
                    sel_data = pulse_data[str(pulseNo)][quantity]
                    #SWEEP OF FIELDS
                    for ifield,field in enumerate(pulse_data['fields'][quantity]):
                        #SUBPLOT
                        if quantity!='CHI2':
                            if sel_data['PLOT_DATA']['labels']!=[]:
                                ax[i,j].plot(sel_data['TIME'][ifield]*1.e+3,sel_data['DATA'][ifield]/sel_data['PLOT_DATA']['norm'],color=sel_data['PLOT_DATA']['color'][ifield],label=sel_data['PLOT_DATA']['labels'][ifield])
                                ax[i,j].legend(ncol=3,fontsize=ticksize)
                            else:
                                ax[i,j].plot(sel_data['TIME'][ifield]*1.e+3,sel_data['DATA'][ifield]/sel_data['PLOT_DATA']['norm'],color=sel_data['PLOT_DATA']['color'][ifield])
                        else:
                            if optimized_plot:
                                ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA'],color=sel_data['PLOT_DATA']['color'][0],label='optimized z_shift')
                                ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA0'],color=sel_data['PLOT_DATA']['color'][1],label='z_shift=0cm')
                                ax[i,j].legend(ncol=1,fontsize=ticksize)
                            else:
                                ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA'],color=sel_data['PLOT_DATA']['color'][0])
                    #YLABEL
                    ax[i,j].set_ylabel(sel_data['PLOT_DATA']['label'],size=labelsize)
                    #LIMITS
                    if sel_data['PLOT_DATA']['ylim'] is not None:
                        if sel_data['PLOT_DATA']['ylim'][1] is None:
                            ax[i,j].set_ylim(sel_data['PLOT_DATA']['ylim'][0],ax[i,j].set_ylim()[1])
                        else:
                            ax[i,j].set_ylim(sel_data['PLOT_DATA']['ylim'][0],sel_data['PLOT_DATA']['ylim'][1])
                    #TICK SIZE
                    ax[i,j].tick_params(axis='both', labelsize=ticksize)
                    #AXIS TO THE RIGHT
                    if j==1:
                        ax[i,j].yaxis.tick_right()
                        ax[i,j].yaxis.set_label_position("right")
                    #COUNT UPDATE
                    k += 1
            #X LIMIT
            plt.xlim(xlim[0]*1.e+3,xlim[1]*1.e+3)
            #X LABEL
            for i in range(0,no_cols):
                ax[no_rows-1,i].set_xlabel('Time [ms]',size=labelsize)
            #PLOT TITLE
            fig.suptitle('#'+str(pulseNo)+' - '+title.replace('_',' '),fontsize=labelsize*1.5)
            plt.subplots_adjust(top=0.92,wspace=0.05, hspace=0.1)            
            #SAVING THE PLOT
            if save_plot:
                fileName = save_directory + '/' + str(pulseNo) +  '_' + title + '.png'
                #SAVING THE PLOT
                plt.savefig(fileName)
                print(fileName+' is saved')
                plt.close()            
 
#PLOT CORRELATION
if plot_correlation:
    #CLOSING ALL THE FIGURES
    plt.close('all')    
    #SWEEP OF PLOTS
    for title,plot_list in data_plot_evolution.items():        
        #SUBPLOT DECLARATION
        no_rows = int(len(plot_list)/2)
        no_cols = 2            
        #FIGURE DECLARATION
        fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(18,10),sharex=True,sharey=True)
        #COUNT VARIABLE
        k = 0            
        #X DATA
        x_data = np.arange(0,len(pulse_data['pulseNos']))
        x_text = [str(x) for x in pulse_data['pulseNos']]
        #SWEEP OF ROWS
        for i in range(0,no_rows):
            for j in range(0,no_cols):
                #SELECTING THE CORRELATION DATA
                sel_data = corr_data[plot_list[k]]
                #COLORS
                colors = infoNodes[plot_list[k]]['color']
                #YLIMIT
                ax[i,j].set_ylim(-1,1)
                #SWEEP OF DICTIONARY
                for ikey,key in enumerate(sel_data.keys()):
                    #DATA
                    y_data = sel_data[key]
                    #COLOR
                    color = colors[ikey]
                    #SWEEP OF PULSETYPES
                    for pulseType,value in data_pulseTypes.items():
                        #SCATTER SYMBOL
                        symbol = value['symbol']
                        #SELECTION MAP
                        sel_map = np.where(np.array(pulse_data['pulseType'])==pulseType)[0]
                        #PLOT
                        # print(k,i,j,(x_data[sel_map],sel_data[key][sel_map],))
                        ax[i,j].scatter(x_data[sel_map],sel_data[key][sel_map],color=colors[ikey],marker=symbol)
                    #TITLE
                    if ikey==0:
                        ax[i,j].text(ax[i,j].set_xlim()[0],ax[i,j].set_ylim()[1],key,fontsize=ticksize,ha='left',va='top',color=colors[ikey])
                    elif ikey==1:
                        ax[i,j].text(ax[i,j].set_xlim()[1],ax[i,j].set_ylim()[1],key,fontsize=ticksize,ha='right',va='top',color=colors[ikey])
                #COUNTING INCREMENT
                k += 1
                #TICK SIZE
                ax[i,j].tick_params(axis='both', labelsize=ticksize)
                #XLABEL
                if i==no_rows-1:
                    ax[i,j].set_xticks(x_data)
                    ax[i,j].set_xticklabels(x_text,fontsize=ticksize)        
        #FIGURE ATTRIBUTES
        fig.suptitle('correlations - '+title.replace('_',' '),fontsize=labelsize*1.5)
        plt.subplots_adjust(top=0.92,wspace=0.05, hspace=0.2)   
        #LABELS
        fig.text(0.5, 0.04, 'PulseNo', ha='center', fontsize = labelsize)
        fig.text(0.04, 0.5, 'Correlation Coefficient', va='center', rotation='vertical', fontsize=labelsize)
        #SAVING THE PLOT
        if save_plot:
            fileName = save_directory_base + '/' + title + '_correlation.png'
            #SAVING THE PLOT
            plt.savefig(fileName)
            print(fileName+' is saved')
            plt.close()   