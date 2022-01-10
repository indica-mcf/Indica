import matplotlib.pyplot as plt
import numpy as np
import os
from MDSplus import *
import pickle
from scipy.stats import pearsonr
from scipy import interpolate

plots = True
save_plot = True

ticksize = 15
labelsize = 20

# pulseNos = [9184]
pulseNos = [9184,9408,9409,9411,9539,9537,9560,9229,9538]


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
nodeInfoAll = dict(
    Equilibrium_EFIT = dict(
        R           = dict(
            node    = ['EFIT.BEST.GLOBAL:RMAG','EFIT.BEST.GLOBAL:RGEO','EFIT.BEST.GLOBAL:RC'],
            color   = ['red','purple','green'],
            norm    = 1.e-2,
            label   = 'R [cm]',
            labels  = ['Rmag','Rgeo','RIp'], 
            ylim    = None,
            title   = ['Rmag (EFIT)']
            ),
        Z           = dict(
            node    = ['EFIT.BEST.GLOBAL:ZMAG','EFIT.BEST.GLOBAL:ZGEO','EFIT.BEST.GLOBAL:ZC'],
            color   = ['red','purple','green'],
            norm    = 1.e-2,
            label   = 'Z [cm]',
            labels  = ['Zmag','Zgeo','ZIp'], 
            ylim    = None,
            title   = ['Zmag (EFIT)']
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
            title   = ['psi (EFIT)']
            ),
          BETA       = dict(
            node    = ['EFIT.BEST.VIRIAL:BTPM','EFIT.BEST.VIRIAL:BTPD'],
            color   = ['blue','GREEN'],
            norm    = 1.e0,
            label   = 'betaP',
            labels  = ['betaP(3) MHD','diamagnetic'], 
            ylim    = None,
            title   = ['beta (EFIT)']
            ),
        ),
    
    Equilibrium_PFIT = dict(
        R           = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:RGEO','PFIT.POST_BEST.RESULTS.GLOBAL:RIP'],
            color   = ['red','purple'],
            norm    = 1.e-2,
            label   = 'R [cm]',
            labels  = ['Rgeo','RIp'], 
            ylim    = None,
            title   = ['Rgeo (PFIT)']
            ),
        Z           = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:ZGEO','PFIT.POST_BEST.RESULTS.GLOBAL:ZIP'],
            color   = ['red','green'],
            norm    = 1.e-2,
            label   = 'Z [cm]',
            labels  = ['Zgeo','ZIp'], 
            ylim    = None,
            title   = ['Zgeo (PFIT)']
            ),
        AMINOR      = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:AMINOR'],
            color   = ['red'],
            norm    = 1.e-2,
            label   = 'AMINOR\n[cm]',
            labels  = [], 
            ylim    = None,
            title   = ['Aminor (PFIT)']
            ),        
        VOLUME      = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:VPLASMA'],
            color   = ['green'],
            norm    = 1.e0,
            label   = 'Vp [m3]',
            labels  = [], 
            ylim    = None,
            title   = ['Vplasma (PFIT)']
            ),
        ELON        = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:KAPPA'],
            color   = ['purple'],
            norm    = 1.e0,
            label   = '$\\kappa$ [no unit]',
            labels  = [], 
            ylim    = None,
            title   = ['$\\kappa$ (PFIT)']
            ),
          BETA       = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:BETAP2'],
            color   = ['blue','GREEN','purple'],
            norm    = 1.e0,
            label   = 'betaP',
            labels  = ['betaP2'], 
            ylim    = None,
            title   = ['betaP2 (PFIT)']
            ),
        ),    
    
    MHD = dict(
        SMMH        = dict(
            node    = ['INTERFEROM.SMMH1_SCALE.BEST.LINE_INT:NE_EFIT'],
            color   = ['brown'],
            norm    = 1.e+19,
            label   = 'nel\n[10$^{19}$ m$^{-2}$]',
            labels  = ['SMMH'],    
            ylim    = None,
            title   = ['nel (SMMH)']
            ),        
        SXR_SP         = dict(
            node    = ['SXR.DIODE_DETR.BEST.FILTER_001:SIGNAL'],
            color   = ['brown'],
            norm    = 1.e+3,
            label   = 'DIODE_DETR',
            labels  = ['DIODE_DETR:FILTER001 (Single point) [kW/m2]'],       
            ylim    = None,
            title   = ['SXR SP - Filter 1'],
            ),
        IP          = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:IP','EFIT.BEST.CONSTRAINTS.IP:CVALUE'],
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
            label   = 'Diode [a.u.]',
            labels  = [], 
            ylim    = None,
            title   = ['Line diode (NOFILTER_MP1)']
            ),        
        ),
    
    Plasma_parameters   = dict(
        NEL         = dict(
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
        IP          = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:IP','EFIT.BEST.CONSTRAINTS.IP:CVALUE'],
            color   = ['blue','red'],
            norm    = 1.e+3,
            label   = 'Ip [MA]',
            labels  = ['PFIT#POST_BEST','EFIT#BEST'], 
            ylim    = None,
            title   = ['Iplasma (PFIT)']
            ),  
        W           = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:WMHD','EFIT.BEST.VIRIAL:WP','DIALOOP.BEST.GLOBAL:WDIA'],
            color   = ['blue','red','green'],
            norm    = 1.e+3,
            label   = 'W$_{MHD}$ [kJ]',
            labels  = ['PFIT#POST_BEST','EFIT#BEST','DIALOOP#BEST'], 
            ylim    = None,
            title   = ['W$_{MHD}$ (PFIT)']
            ),  
         LI         = dict(
            node    = ['PFIT.POST_BEST.RESULTS.GLOBAL:LI2','EFIT.BEST.VIRIAL:LI3M'],
            color   = ['brown','purple'],
            norm    = 1.e+0,
            label   = 'li',
            labels  = ['PFIT#POST_BEST','EFIT#BEST'], 
            ylim    = None,
            title   = ['l$_{i}$ (PFIT)']
            ),  
        )
    )

        
d_sxr_cam = dict(
    SXR_CAM = dict(
    node    = ['SXR.DIODE_ARRAYS.BEST.MIDDLE_HEAD.FILTER_4:CH0'+str(x) for x in range(61,81)],
    color   = ['#440154','#481567','#482677','#453781','#404788','#39568C','#33638D','#2D708E','#287D8E','#238A8D','#1F968B','#20A387','#29AF7F','#3CBB75','#55C667','#73D055','#95D840','#B8DE29','#DCE319','#FDE725'],
    norm    = 1.e+3,
    label   = 'SXR\n[kW/m2]',
    labels  = [],     
    ylim    = [0,None]
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
        #CUTTING THE DATA
        sel_map = (temp_time>=xlim[0])&(temp_time<=xlim[1])
        #SELECTED DATA
        time += (temp_time[sel_map],)
        data += (temp_data[sel_map],)    
    #RETURNING TIME AND DATA
    return time,data

#PLOTS
if plots:
    #CLOSING ALL THE FIGURES
    plt.close('all')
    #OPENING MDSPLUS CONNECTION
    conn = Connection('192.168.1.7')
    #SWEEP OF PLOTS
    for title in nodeInfoAll.keys():
        #SAVE DIRECTORY
        save_directory = get_save_directory(title.lower()+'_plots')
        #NODE INFO
        nodeInfo = nodeInfoAll[title]
        nodeInfo.update(d_sxr_cam)
        #CORRELATION COEFFICIENT DECLARATION
        corr_coeff = np.nan * np.ones((len(pulseNos),len(nodeInfo.keys())-1))
        corr_title = []
        pulseTypes = []
        #SWEEP OF PULSES
        for ipulse,pulseNo in enumerate(pulseNos):
            #OPENING ST40 TREE
            conn.openTree('ST40',pulseNo)
            #SUBPLOT DECLARATION
            no_items = len(nodeInfo.keys())
            no_rows = int((no_items+2)/2)
            no_cols = 2
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(18,10),sharex=True)
            #SWEEP OF PULSE TYPES
            for pulseType,pulseData in data_pulseTypes.items():
                if pulseNo in pulseData['pulseNos']:
                    pulseTypes += [pulseType]
            #LOADING THE PICKLE FILE
            filename = save_directory_base+'/../'+str(pulseNo)+'/'+str(pulseNo)+'.p'
            inv_data = pickle.load(open(filename,'rb'))['filter_4']
            #X LIMIT
            xlim = [0.01,0.15]
            #QUANTITIES
            quantities = list(nodeInfo.keys())
            k = 0
            #SWEEP OF ROWS
            for i in range(0,no_rows):
                for j in range(0,no_cols):
                    if k<len(quantities):
                        #DATA AND TIME
                        value = nodeInfo[quantities[k]]
                        time,data = get_data(value['node'],conn,xlim)
                        #CORRELATION
                        if k<np.size(corr_coeff,1):
                            try:
                                x = interpolate.interp1d(time[0],data[0],bounds_error=False)(inv_data['t'])
                                y = inv_data['back_integral']['chi2']
                            except Exception as e:
                                print(e)
                            map_sel = np.isfinite(x) & np.isfinite(y)
                            sel_x = x[map_sel]
                            sel_y = y[map_sel]
                            if np.sum(map_sel)>2:
                                corr_coeff[ipulse,k] = pearsonr(sel_x,sel_y)[0]
                            if ipulse==0:
                                corr_title += [value['title'][0]]
                        #PLOTTING
                        for ii in range(0,len(time)):
                            if value['labels']!=[]:
                                ax[i,j].plot(time[ii]*1.e+3,data[ii]/value['norm'],color=value['color'][ii],label=value['labels'][ii])
                            else:
                                ax[i,j].plot(time[ii]*1.e+3,data[ii]/value['norm'],color=value['color'][ii])
                        #YLABEL
                        ax[i,j].set_ylabel(value['label'],size=labelsize)
                        #LIMITS
                        if value['ylim'] is not None:
                            if value['ylim'][1] is None:
                                ax[i,j].set_ylim(value['ylim'][0],ax[i,j].set_ylim()[1])
                            else:
                                ax[i,j].set_ylim(value['ylim'][0],value['ylim'][1])
                        #LEGEND
                        if value['labels']!=[]:
                            ax[i,j].legend(ncol=3,fontsize=ticksize)
                        #TICK SIZE
                        ax[i,j].tick_params(axis='both', labelsize=ticksize)
                        #AXIS TO THE RIGHT
                        if j==1:
                            ax[i,j].yaxis.tick_right()
                            ax[i,j].yaxis.set_label_position("right")
                        #COUNT UPDATE
                        k += 1
            #CHI2 PLOT
            ax[no_rows-1,1].scatter(inv_data['t']*1.e+3,inv_data['back_integral']['chi2'])
            ax[no_rows-1,1].set_ylim(0,2)
            ax[no_rows-1,1].set_ylabel('chi2\n[no unit]',size=labelsize)
            ax[no_rows-1,1].tick_params(axis='both', labelsize=ticksize)
            ax[no_rows-1,1].yaxis.tick_right()
            ax[no_rows-1,1].yaxis.set_label_position("right")
            #X LIMIT
            plt.xlim(xlim[0]*1.e+3,xlim[1]*1.e+3)
            #X LABEL
            xlab = 'Time [ms]'
            ax[no_rows-1,0].set_xlabel(xlab,size=labelsize)
            ax[no_rows-1,1].set_xlabel(xlab,size=labelsize)
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
        #SAVING THE CORRELATION COEFFICIENT DATA
        corr_data = dict(
            pulseNos = pulseNos,
            pulseTypes = pulseTypes,
            data_pulseTypes = data_pulseTypes,
            corr_coeff = corr_coeff,    
            corr_fields = corr_title,
            )    
        #SAVE DIRECTORY
        filename = save_directory + '/correlation_data.p'
        with open(filename,'wb') as handle:
            pickle.dump(corr_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print(filename+' is saved')
            
        #CLOSING ALL THE PLOTS
        plt.close('all')
        #CORRELATION PLOT
        no_rows = int(len(corr_data['corr_fields'])/2)
        no_cols = 2
        fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(16,10),sharex=True,sharey=True)
        k = 0
        #SWEEEP OF PLOTS
        x_data = np.arange(0,len(corr_data['pulseNos']))
        x_text = [str(x) for x in corr_data['pulseNos']]
        for i in range(0,no_rows):
            for j in range(0,no_cols):
                #SWEEP OF PULSE TYPES
                for pulseType in data_pulseTypes.keys():
                    #SELECTED PULSES
                    sel_map = np.where(np.array(corr_data['pulseTypes'])==pulseType)[0]
                    #SCATTER PLOT
                    ax[i,j].scatter(x_data[sel_map],corr_data['corr_coeff'][sel_map,k],color=data_pulseTypes[pulseType]['color'],marker=data_pulseTypes[pulseType]['symbol'],label=pulseType)
                #LEGEND
                if (i==0) & (j==0):
                    ax[i,j].legend(ncol=3,fontsize=ticksize)
                #TITLE
                ax[i,j].set_title('chi2 vs '+corr_data['corr_fields'][k],fontsize=ticksize)
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
            fileName = save_directory + '/' + title + '_correlation.png'
            #SAVING THE PLOT
            plt.savefig(fileName)
            print(fileName+' is saved')
            plt.close()   