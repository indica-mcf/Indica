import matplotlib.pyplot as plt
import numpy as np
import os
from MDSplus import *
import pickle
from scipy.stats import pearsonr
from scipy import interpolate

optimized_plot      = False
generate_data       = True
generate_corr       = True
generate_plot_data  = True
plot_evolution      = True
plot_correlation    = True

save_plot = True

ticksize = 15
labelsize = 20

pulseNos = [9409]
# pulseNos = [9184,9408,9409,9411,9539,9537,9560,9229,9538]
pulseNos_RUN01 = [9408,9409]
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
    save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots_tomo_1D/plots_EFIT'

#FUNCTION TO GET SAVE DIRECTORY
def get_save_directory(folder = ''):
    os.chdir(save_directory_base)
    try:
        os.mkdir(folder)
    except:
        pass
    os.chdir(folder)
    return os.getcwd()

# #NODE INFORMATION
# infoNodes = dict(
#     CHI2        = dict(
#         label   = '$\\chi^2$ [no unit]',
#         labels  = [],     
#         ylim    = [0,2],
#         color   = ['blue','red'],
#         title   = ['$\\chi^2$'],
#         ),
#     CHI_EFIT    = dict(
#         node    = ['EFIT.BEST.GLOBAL:'+x for x in ['CHIT','CHIM']],
#         color   = ['brown','purple'],
#         norm    = 1.e+0,
#         label   = 'chi2',
#         labels  = ['chi2$_{tot}$ (EFIT)','chi2$_{mag}$ (EFIT)'], 
#         ylim    = None,
#         title   = ['chi2$_{tot}$ (EFIT)','chi2$_{mag}$ (EFIT)']
#         ),  
# )

#EFIT SPECIAL FIELDS
fields_EFIT = dict(
    #FLUX LOOPS
    FLUX    = dict(
            nodes = ['PSI_FLOOP_001', 'PSI_FLOOP_002', 'PSI_FLOOP_003', 'PSI_FLOOP_004', 'PSI_FLOOP_005', 'PSI_FLOOP_006', 'PSI_FLOOP_007', 'PSI_FLOOP_008', 'PSI_FLOOP_009', 'PSI_FLOOP_010', 'PSI_FLOOP_011', 'PSI_FLOOP_012', 'PSI_FLOOP_013', 'PSI_FLOOP_014', 'PSI_FLOOP_015', 'PSI_FLOOP_016', 'PSI_FLOOP_017', 'PSI_FLOOP_018', 'PSI_FLOOP_019', 'PSI_FLOOP_020', 'PSI_FLOOP_021', 'PSI_FLOOP_022', 'PSI_FLOOP_023', 'PSI_FLOOP_024', 'PSI_FLOOP_025', 'PSI_FLOOP_026', 'PSI_FLOOP_027', 'PSI_FLOOP_028', 'PSI_FLOOP_029', 'PSI_FLOOP_030','PSI_FLOOP_101', 'PSI_FLOOP_102', 'PSI_FLOOP_103', 'PSI_FLOOP_104', 'PSI_FLOOP_105', 'PSI_FLOOP_106', 'PSI_FLOOP_201', 'PSI_FLOOP_212'],
            title = 'Flux loops',
            ),
    #BP PROBES
    BP       = dict(
            nodes = ['B_BPPROBE_101', 'B_BPPROBE_102', 'B_BPPROBE_103', 'B_BPPROBE_104', 'B_BPPROBE_105', 'B_BPPROBE_106', 'B_BPPROBE_107', 'B_BPPROBE_108', 'B_BPPROBE_109', 'B_BPPROBE_110', 'B_BPPROBE_111', 'B_BPPROBE_112', 'B_BPPROBE_113', 'B_BPPROBE_114', 'B_BPPROBE_115', 'B_BPPROBE_116', 'B_BPPROBE_117', 'B_BPPROBE_118', 'B_BPPROBE_119', 'B_BPPROBE_120', 'B_BPPROBE_121', 'B_BPPROBE_122', 'B_BPPROBE_123', 'B_BPPROBE_124', 'B_BPPROBE_125', 'B_BPPROBE_126', 'B_BPPROBE_127', 'B_BPPROBE_128', 'B_BPPROBE_129', 'B_BPPROBE_130', 'B_BPPROBE_131', 'B_BPPROBE_132', 'B_BPPROBE_133', 'B_BPPROBE_134'],
            title = 'BP Probe',
            ),
    #ROG COILS
    ROGC     = dict(
            nodes = ['I_ROG_MCT', 'I_ROG_MCB', 'I_ROG_BVLT', 'I_ROG_BVLB', 'I_ROG_INIVC000', 'I_ROG_DIVT', 'I_ROG_DIVB', 'I_ROG_GASBFLT', 'I_ROG_GASBFLB', 'I_ROG_HFSPSRT', 'I_ROG_HFSPSRB', 'I_ROG_DIVPSRT', 'I_ROG_DIVPSRB'],
            root_node = 'Rogowski Coil',
            ),
    # #PF COILS
    # PF      = dict(
    #         nodes = ['PFC1', 'PFC2', 'PFC3', 'PFC4', 'PFC5', 'PFC6', 'PFC7', 'PFC8', 'PFC9', 'PFC10', 'PFC11', 'PFC12', 'SOL', 'DIVT', 'DIVB', 'MC', 'BVLT', 'BVLB', 'BVUT', 'BVUB', 'PSHT1', 'PSHT2', 'PSHB1', 'PSHB2'],
    #         root_node = '',
    #         ),
    
    # #IVC FIELDS
    # fields_EIG = ['EIG1', 'EIG2', 'EIG3', 'EIG4', 'EIG5', 'EIG6', 'EIG7', 'EIG8', 'EIG9', 'EIG10', 'EIG11', 'EIG12', 'EIG13', 'EIG14', 'EIG15']
    
    
    )

#SENSOR NAMES
names = dict(
    BP      = ['B_BPPROBE_101', 'B_BPPROBE_102', 'B_BPPROBE_103', 'B_BPPROBE_104', 'B_BPPROBE_105', 'B_BPPROBE_106', 'B_BPPROBE_107', 'B_BPPROBE_108', 'B_BPPROBE_109', 'B_BPPROBE_110', 'B_BPPROBE_111', 'B_BPPROBE_112', 'B_BPPROBE_113', 'B_BPPROBE_114', 'B_BPPROBE_115', 'B_BPPROBE_116', 'B_BPPROBE_117', 'B_BPPROBE_118', 'B_BPPROBE_119', 'B_BPPROBE_120', 'B_BPPROBE_121', 'B_BPPROBE_122', 'B_BPPROBE_123', 'B_BPPROBE_124', 'B_BPPROBE_125', 'B_BPPROBE_126', 'B_BPPROBE_127', 'B_BPPROBE_128', 'B_BPPROBE_129', 'B_BPPROBE_130', 'B_BPPROBE_131', 'B_BPPROBE_132', 'B_BPPROBE_133', 'B_BPPROBE_134'],
    ROGC    = ['I_ROG_MCT', 'I_ROG_MCB', 'I_ROG_DIVT', 'I_ROG_DIVB', 'I_ROG_ERINGT', 'I_ROG_ERINGB', 'I_ROG_BVLT', 'I_ROG_BVLB', 'I_ROG_OVC', 'I_ROG_DIVPSRT', 'I_ROG_DIVPSRB', 'I_ROG_HFSPSRT', 'I_ROG_HFSPSRB'],
    FLUX    = ['PSI_FLOOP_001', 'PSI_FLOOP_002', 'PSI_FLOOP_003', 'PSI_FLOOP_004', 'PSI_FLOOP_005', 'PSI_FLOOP_006', 'PSI_FLOOP_007', 'PSI_FLOOP_008', 'PSI_FLOOP_009', 'PSI_FLOOP_010', 'PSI_FLOOP_011', 'PSI_FLOOP_012', 'PSI_FLOOP_013', 'PSI_FLOOP_014', 'PSI_FLOOP_015', 'PSI_FLOOP_016', 'PSI_FLOOP_017', 'PSI_FLOOP_018', 'PSI_FLOOP_019', 'PSI_FLOOP_020', 'PSI_FLOOP_021', 'PSI_FLOOP_022', 'PSI_FLOOP_023', 'PSI_FLOOP_024', 'PSI_FLOOP_025', 'PSI_FLOOP_026', 'PSI_FLOOP_027', 'PSI_FLOOP_028', 'PSI_FLOOP_029', 'PSI_FLOOP_030', 'PSI_FLOOP_101', 'PSI_FLOOP_102', 'PSI_FLOOP_103', 'PSI_FLOOP_104', 'PSI_FLOOP_105', 'PSI_FLOOP_106', 'PSI_FLOOP_201', 'PSI_FLOOP_202', 'PSI_FLOOP_203', 'PSI_FLOOP_210', 'PSI_FLOOP_211', 'PSI_FLOOP_212'],
    )

#GETTING THE DATA FROM THE NODES
def get_data(whichRun,fields,conn,xlim):
    #RETURN DATA DECLARATION
    return_data = {}
    #GETTING THE EFIT TIME
    temp_time = conn.get(whichRun+':TIME').data()
    #IF TIME IS AVAILABLE
    if type(temp_time)==np.ndarray:
        #SEL MAP
        sel_map = (temp_time>=xlim[0])&(temp_time<=xlim[1])
        #SELECTING VALID TIMES
        if np.sum(sel_map)>1:
            #SWEEP OF BASE NODES
            for ifield,baseNode in enumerate(fields):
                #GETTING THE CHI2
                temp_chi2 = conn.get(whichRun+'.CONSTRAINTS.'+baseNode+':CHI').data()
                #SWEEP OF NAMES
                for i,name in enumerate(names[baseNode]):
                    return_data[name] = dict(
                        TIME    = temp_time[sel_map],
                        DATA    = temp_chi2[sel_map,i],
                        LABEL   = name
                        )
            #CHI2 OF EFIT
            temp_chi2 = conn.get(whichRun+'.GLOBAL:CHIT').data()
            return_data['CHI2_EFIT'] = dict(
                TIME    = temp_time[sel_map],
                DATA    = temp_chi2[sel_map],
                LABEL   = 'CHI2_EFIT'
                )
    #RETURNING THE DATA
    return return_data
    
#GENERATING DATA
if generate_data:
    #RETURN DATA
    pulse_data = {}
    #OPENING MDSPLUS CONNECTION
    conn = Connection('192.168.1.7')
    #PULSE TYPES DECLARATION
    pulse_data['pulseType'] = []
    #SWEEP OF PULSES
    for ipulse,pulseNo in enumerate(pulseNos):    
        #PULSE DATA
        pulse_data[str(pulseNo)] = {}
        #OPENING EFIT TREE
        conn.openTree('EFIT',pulseNo)
        #WHICHRUN
        whichRun = 'BEST' if (pulseNo not in pulseNos_RUN01) else 'RUN01'
        #BASE NODES TO READ
        baseNodes = list(fields_EFIT.keys())
        #TIME NODE
        timeNode = whichRun + ':TIME'
        #GETTING ALL THE DATA
        pulse_data[str(pulseNo)]['DATA'] = get_data(whichRun,baseNodes,conn,xlim)
        #PULSETYPE
        for pulseType,pulseData in data_pulseTypes.items():
            if pulseNo in pulseData['pulseNos']:
                pulse_data['pulseType'] += [pulseType]
        #GETTING THE SXR INVERSION CHI2 DATA
        filename = save_directory_base+'/../'+str(pulseNo)+'/'+str(pulseNo)+'.p'
        with open(filename,'rb') as handle:
            inv_data = pickle.load(handle)['filter_4']
            pulse_data[str(pulseNo)]['DATA']['CHI2_SXR'] = dict(
                TIME    = inv_data['t'],
                DATA    = inv_data['back_integral']['chi2'],
                LABEL   = 'CHI2_SXR',
                )
    #OTHER INFORMATION
    pulse_data['xlim'] = xlim
    pulse_data['pulseNos'] = pulseNos
    pulse_data['quantities'] = list(pulse_data[str(pulseNo)]['DATA'].keys())    
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
    exclude = ['CHI2_SXR','CHI2_EFIT']
    #CORR DATA
    corr_data = np.nan * np.ones((len(pulse_data['pulseNos']),len(pulse_data['quantities'])-len(exclude)))
    #SWEEEP OF QUANTITIES
    k = 0
    for quantity in pulse_data['quantities']:
        if quantity not in exclude:
            #SWEEP OF PULSES
            for ipulse,pulseNo in enumerate(pulse_data['pulseNos']):
                #SELECTED DATA
                sel_time    = pulse_data[str(pulseNo)]['DATA'][quantity]['TIME']
                sel_data    = pulse_data[str(pulseNo)]['DATA'][quantity]['DATA']
                chi_time    = pulse_data[str(pulseNo)]['DATA']['CHI2_SXR']['TIME']
                chi_data    = pulse_data[str(pulseNo)]['DATA']['CHI2_SXR']['DATA']
                #INTERPOLATION DATA
                x = interpolate.interp1d(sel_time,sel_data,bounds_error=False)(chi_time)      
                y = chi_data
                map_sel = np.isfinite(x) & np.isfinite(y)
                sel_x = x[map_sel]
                sel_y = y[map_sel]
                if np.sum(map_sel)>2:
                    corr_data[ipulse,k] = pearsonr(sel_x,sel_y)[0]
            #COUNT INCREMENT
            k += 1
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

# #EVOLUTION PLOTS
# if plot_evolution:    
#     #CLOSING ALL THE FIGURES
#     plt.close('all')    
#     #SWEEP OF PLOTS
#     for title,plot_list in data_plot_evolution.items():        
#         #SAVE DIRECTORY
#         save_directory = get_save_directory(title.lower()+'_plots')
#         #SWEEP OF PULSES
#         for ipulse,pulseNo in enumerate(pulse_data['pulseNos']):
#             #SUBPLOT DECLARATION
#             no_rows = int(len(plot_list)/2) + 1
#             no_cols = 2            
#             #FIGURE DECLARATION
#             fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(18,10),sharex=True)
#             #COUNT VARIABLE
#             k = 0            
#             #SWEEP OF ROWS
#             for i in range(0,no_rows):
#                 for j in range(0,no_cols):
#                     #QUANTITY
#                     if i<=no_rows-2:
#                         quantity = plot_list[k]
#                     else:
#                         quantity = 'SXR_CAM' if j==0 else 'CHI2'
#                     #DATA
#                     sel_data = pulse_data[str(pulseNo)][quantity]
#                     #SWEEP OF FIELDS
#                     for ifield,field in enumerate(pulse_data['fields'][quantity]):
#                         #SUBPLOT
#                         if quantity!='CHI2':
#                             if sel_data['PLOT_DATA']['labels']!=[]:
#                                 ax[i,j].plot(sel_data['TIME'][ifield]*1.e+3,sel_data['DATA'][ifield]/sel_data['PLOT_DATA']['norm'],color=sel_data['PLOT_DATA']['color'][ifield],label=sel_data['PLOT_DATA']['labels'][ifield])
#                                 ax[i,j].legend(ncol=3,fontsize=ticksize)
#                             else:
#                                 ax[i,j].plot(sel_data['TIME'][ifield]*1.e+3,sel_data['DATA'][ifield]/sel_data['PLOT_DATA']['norm'],color=sel_data['PLOT_DATA']['color'][ifield])
#                         else:
#                             if optimized_plot:
#                                 ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA'],color=sel_data['PLOT_DATA']['color'][0],label='optimized z_shift')
#                                 ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA0'],color=sel_data['PLOT_DATA']['color'][1],label='z_shift=0cm')
#                                 ax[i,j].legend(ncol=1,fontsize=ticksize)
#                             else:
#                                 ax[i,j].scatter(sel_data['TIME']*1.e+3,sel_data['DATA'],color=sel_data['PLOT_DATA']['color'][0])
#                     #YLABEL
#                     ax[i,j].set_ylabel(sel_data['PLOT_DATA']['label'],size=labelsize)
#                     #LIMITS
#                     if sel_data['PLOT_DATA']['ylim'] is not None:
#                         if sel_data['PLOT_DATA']['ylim'][1] is None:
#                             ax[i,j].set_ylim(sel_data['PLOT_DATA']['ylim'][0],ax[i,j].set_ylim()[1])
#                         else:
#                             ax[i,j].set_ylim(sel_data['PLOT_DATA']['ylim'][0],sel_data['PLOT_DATA']['ylim'][1])
#                     #TICK SIZE
#                     ax[i,j].tick_params(axis='both', labelsize=ticksize)
#                     #AXIS TO THE RIGHT
#                     if j==1:
#                         ax[i,j].yaxis.tick_right()
#                         ax[i,j].yaxis.set_label_position("right")
#                     #COUNT UPDATE
#                     k += 1
#             #X LIMIT
#             plt.xlim(xlim[0]*1.e+3,xlim[1]*1.e+3)
#             #X LABEL
#             for i in range(0,no_cols):
#                 ax[no_rows-1,i].set_xlabel('Time [ms]',size=labelsize)
#             #PLOT TITLE
#             fig.suptitle('#'+str(pulseNo)+' - '+title.replace('_',' '),fontsize=labelsize*1.5)
#             plt.subplots_adjust(top=0.92,wspace=0.05, hspace=0.1)            
#             #SAVING THE PLOT
#             if save_plot:
#                 fileName = save_directory + '/' + str(pulseNo) +  '_' + title + '.png'
#                 #SAVING THE PLOT
#                 plt.savefig(fileName)
#                 print(fileName+' is saved')
#                 plt.close()            
 
# #PLOT CORRELATION
# if plot_correlation:
#     #CLOSING ALL THE FIGURES
#     plt.close('all')    
#     #SWEEP OF PLOTS
#     for title,plot_list in data_plot_evolution.items():        
#         #SUBPLOT DECLARATION
#         no_rows = int(len(plot_list)/2)
#         no_cols = 2            
#         #FIGURE DECLARATION
#         fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(18,10),sharex=True,sharey=True)
#         #COUNT VARIABLE
#         k = 0            
#         #X DATA
#         x_data = np.arange(0,len(pulse_data['pulseNos']))
#         x_text = [str(x) for x in pulse_data['pulseNos']]
#         #SWEEP OF ROWS
#         for i in range(0,no_rows):
#             for j in range(0,no_cols):
#                 #SELECTING THE CORRELATION DATA
#                 sel_data = corr_data[plot_list[k]]
#                 #COLORS
#                 colors = infoNodes[plot_list[k]]['color']
#                 #YLIMIT
#                 ax[i,j].set_ylim(-1,1)
#                 #SWEEP OF DICTIONARY
#                 for ikey,key in enumerate(sel_data.keys()):
#                     #DATA
#                     y_data = sel_data[key]
#                     #COLOR
#                     color = colors[ikey]
#                     #SWEEP OF PULSETYPES
#                     for pulseType,value in data_pulseTypes.items():
#                         #SCATTER SYMBOL
#                         symbol = value['symbol']
#                         #SELECTION MAP
#                         sel_map = np.where(np.array(pulse_data['pulseType'])==pulseType)[0]
#                         #PLOT
#                         # print(k,i,j,(x_data[sel_map],sel_data[key][sel_map],))
#                         ax[i,j].scatter(x_data[sel_map],sel_data[key][sel_map],color=colors[ikey],marker=symbol)
#                     #TITLE
#                     if ikey==0:
#                         ax[i,j].text(ax[i,j].set_xlim()[0],ax[i,j].set_ylim()[1],key,fontsize=ticksize,ha='left',va='top',color=colors[ikey])
#                     elif ikey==1:
#                         ax[i,j].text(ax[i,j].set_xlim()[1],ax[i,j].set_ylim()[1],key,fontsize=ticksize,ha='right',va='top',color=colors[ikey])
#                 #COUNTING INCREMENT
#                 k += 1
#                 #TICK SIZE
#                 ax[i,j].tick_params(axis='both', labelsize=ticksize)
#                 #XLABEL
#                 if i==no_rows-1:
#                     ax[i,j].set_xticks(x_data)
#                     ax[i,j].set_xticklabels(x_text,fontsize=ticksize)        
#         #FIGURE ATTRIBUTES
#         fig.suptitle('correlations - '+title.replace('_',' '),fontsize=labelsize*1.5)
#         plt.subplots_adjust(top=0.92,wspace=0.05, hspace=0.2)   
#         #LABELS
#         fig.text(0.5, 0.04, 'PulseNo', ha='center', fontsize = labelsize)
#         fig.text(0.04, 0.5, 'Correlation Coefficient', va='center', rotation='vertical', fontsize=labelsize)
#         #SAVING THE PLOT
#         if save_plot:
#             fileName = save_directory_base + '/' + title + '_correlation.png'
#             #SAVING THE PLOT
#             plt.savefig(fileName)
#             print(fileName+' is saved')
#             plt.close()   