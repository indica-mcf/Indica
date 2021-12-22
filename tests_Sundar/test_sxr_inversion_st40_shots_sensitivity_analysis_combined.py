import test_sxr_inversion_st40_shots_sensiivity_analysis as ss
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

#SAVE DIRECTORY
save_directory_data = ss.get_save_directory('combined_data')
save_directory_plot = ss.get_save_directory('combined_plot')
save_directory_var  = ss.get_save_directory('combined_variation')
base_directory = ss.save_directory_base

#FIELDS
fields = ss.fields

#SWITCHES
plot_combined  = True
plot_variation = True

#CAMERA
camera = 'filter_4'

#COMBINED BACK-INTEGRAL PLOT
if plot_combined:
    #SWEEP OF FIELDS
    for field in fields:
        #NUMBER OF SWEEPS
        no_sweeps = len(ss.sweep_values[field])
        #ALL FILES DATA
        if ss.version_control:
            directory = ss.get_save_directory('sweep_'+field+'_'+ss.version)
        else:
            directory = ss.get_save_directory('sweep_'+field)
        all_files = os.listdir(directory)
        #COMBINED DATA DECLARAION
        combined_data = {}
        for file in all_files:
            if ('.p' in file) & ('.png' not in file):
                filename = file
                with open(directory+'/'+filename, 'rb') as handle:
                    # combined_data[filename.replace('.p','')] = pickle.load(handle)
                    temp_data = pickle.load(handle)
                    for key,value in temp_data.items():
                        combined_data[key] = value
        #SAVING THE COMBINED DATA
        if ss.version_control:
            filename = save_directory_data+'/sweep_'+field+'_'+ss.version+'.p'    
        else:
            filename = save_directory_data+'/sweep_'+field+'.p'
        with open(filename, 'wb') as handle:
                pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #TIMES
        times = combined_data[list(combined_data.keys())[0]][camera]['t'].data
        if field=='d_time':
            times = [times[0]]
        #NUMBER OF ROWS AND COLUMNS
        if no_sweeps>6:
            no_cols = 4
        elif no_sweeps<3:
            no_cols = 2
        else:
            no_cols = 3
        no_rows = int(no_sweeps/no_cols) if no_sweeps%no_cols==0 else int(no_sweeps/no_cols)+1
        no_delete = (no_rows*no_cols)-no_sweeps
        #ROWS AND COLUMNS SEQUENCE
        plt_sequence = []
        for i in range(0,no_rows):
            for j in range(no_cols):
                plt_sequence += [[i,j]]
        #SWEEP OF TIMES
        for i_time,time in enumerate(times):
                #FIGURE DECLARATION
                plt.close('all')
                fig,ax = plt.subplots(nrows=no_rows,ncols=no_cols,squeeze=True,figsize=(16,10),sharex=True,sharey=True)
                #SWEEP OF VALUES
                for i,value in enumerate(ss.sweep_values[field]):
                    #KEY
                    if '_shift' in field:
                        SuffixValue = int(value * 1.e+2)
                    elif '_time' in field:
                        SuffixValue = int(value * 1.e+3)
                    elif 'exclude' in field:
                        SuffixValue = value
                    else:
                        SuffixValue = int(value)
                    key = field+'_'+str(SuffixValue)
                    key = str(ss.pulseNo)+'_'+key
                    #CHI2 VALUE
                    chi2 = combined_data[key][camera]['back_integral']['chi2'][i_time]
                    #TITLE
                    title = field+' = '+str(SuffixValue)+', chi2 = '+str(np.round(chi2,2))
                    #SELECTED DATA
                    sel_data = combined_data[key][camera]['back_integral']
                    #I AND J
                    [i1,i2] = plt_sequence[i]
                    #AXIS DEFINITION
                    try:
                        sel_ax = ax[i1,i2]
                    except:
                        sel_ax = ax[i1+i2]
                    #PLOT
                    sel_ax.plot(sel_data['channel_no'][combined_data[key][camera]['channels_considered']],sel_data['data_theory'][i_time,combined_data[key][camera]['channels_considered']]/1.e+3,color='b')
                    sel_ax.scatter(sel_data['channel_no'][combined_data[key][camera]['channels_considered']],sel_data['data_experiment'][i_time,combined_data[key][camera]['channels_considered']]/1.e+3,color='k')
                    sel_ax.scatter(sel_data['channel_no'][np.logical_not(combined_data[key][camera]['channels_considered'])],sel_data['data_experiment'][i_time,np.logical_not(combined_data[key][camera]['channels_considered'])]/1.e+3,color='r',marker='v')
                    #TITLE
                    sel_ax.set_title(title)
                #DELETING THE LAST SUBPLOT
                for i in range(0,no_delete):
                    [i1,i2] = plt_sequence[no_sweeps+i]
                    fig.delaxes(ax[i1][i2])
                fig.text(0.5, 0.04, 'Channel number', ha='center',fontsize=25)
                fig.text(0.08, 0.5, 'Line emission [Kw/m-2]', va='center', rotation='vertical',fontsize=25)
                plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False,labelsize=50)
                plt.suptitle('#'+str(ss.pulseNo)+' @ t='+str(np.round((time-combined_data[key]['input_data']['d_time']/2)*1.e+3,2))+' - '+str(np.round((time-combined_data[key]['input_data']['d_time']/2)*1.e+3,2))+' ms - Sweep of '+field,fontsize=25)
                #SAVING THE PLOT
                if ss.version_control:
                    filenameFig = save_directory_plot+'/sweep_'+field+'_'+ss.version+'_'+str(ss.pulseNo)+'_t_'+str(i_time+1)+'.png'
                else:
                    filenameFig = save_directory_plot+'/sweep_'+field+'_'+str(ss.pulseNo)+'_t_'+str(i_time+1)+'.png'
                plt.savefig(filenameFig)
                plt.close('all')
                print(filenameFig+' saved successfully!')
    
    
#VARIATION PLOTS
if plot_variation:
    field = ['d_time']
    #SWEEP OF FIELDS
    for field in fields:
        #NUMBER OF SWEEPS
        no_sweeps = len(ss.sweep_values[field])
        #ALL FILES DATA
        if ss.version_control:
            directory = ss.get_save_directory('sweep_'+field+'_'+ss.version)
        else:
            directory = ss.get_save_directory('sweep_'+field)
        all_files = os.listdir(directory)
        #COMBINED DATA DECLARAION
        combined_data = {}
        for file in all_files:
            if ('.p' in file) & ('.png' not in file):
                filename = file
                with open(directory+'/'+filename, 'rb') as handle:
                    # combined_data[filename.replace('.p','')] = pickle.load(handle)
                    temp_data = pickle.load(handle)
                    for key,value in temp_data.items():
                        combined_data[key] = value
        #VALUES
        values = ss.sweep_values[field]
        #TIMES
        times = combined_data[list(combined_data.keys())[0]][camera]['t'].data
        if field=='d_time':
            times = [times[0]]
        #CHI2 VALUES
        chi2 = np.nan * np.ones((len(values),len(times)))
        #SWEEP OF VALUES
        for ival,value in enumerate(values):
            #KEY
            if '_shift' in field:
                SuffixValue = int(value * 1.e+2)
                data_unit = 'cm'
            elif '_time' in field:
                SuffixValue = int(value * 1.e+3)
                data_unit = 'ms'
            elif 'exclude' in field:
                SuffixValue = value
                data_unit = ''
            else:
                SuffixValue = int(value)
                data_unit = 'degree'
            key = field+'_'+str(SuffixValue)
            key = str(ss.pulseNo)+'_'+key
            #CHI2 VALUE
            if field=='d_time':
                chi2[ival,:] = combined_data[key][camera]['back_integral']['chi2'][0]
            else:
                chi2[ival,:] = combined_data[key][camera]['back_integral']['chi2']
        #FIGURE DECLARATION
        plt.close('all')
        plt.figure(figsize=(16,10))
        #SWEEP OF TIMES
        for i_time,time in enumerate(times):
            #LABEL
            label = '#'+str(ss.pulseNo)+' @ t='+str(np.round((time-combined_data[key]['input_data']['d_time']/2)*1.e+3,2))+' - '+str(np.round((time-combined_data[key]['input_data']['d_time']/2)*1.e+3,2))+' ms'            
            #PLOT
            plt.scatter(np.arange(0,len(values)),chi2[:,i_time])
            plt.plot(np.arange(0,len(values)),chi2[:,i_time],label=label)
            plt.legend(fontsize=15)
            if data_unit!='':
                plt.xlabel(field+' ['+data_unit+']',fontsize=25)
            else:
                plt.xlabel(field,fontsize=25)
            plt.ylabel('chi2 [no unit]',fontsize=25)
            plt.tick_params(axis='both',labelsize=20)
            plt.xticks(np.arange(0,len(values)),values)
        #SAVING THE PLOT
        if ss.version_control:
            filenameFig = save_directory_var+'/sweep_'+field+'_'+ss.version+'.png'
        else:
            filenameFig = save_directory_var+'/sweep_'+field+'.png'
        plt.savefig(filenameFig)
        plt.close('all')