import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import yasa

# Load data
data_path = "C:/Users/klacourse/Documents/NGosselin/data/edf/RBD/34_final_subjects/run_again/"
LOC_label = 'LOC'
ROC_label = 'ROC'
channel_2_rem = ['ECG DII', 'ECG D1']

REM_amplitude_det = 30

# Extract subject label from a list of .edf file
# List files with the .edf extension
edf_files = [file for file in os.listdir(data_path) if file.endswith('.edf')]
# Create a list of subject labels without the .edf extension
subject_label = [file[:-4] for file in edf_files]  # Removes the last 4 characters (.edf)

error_flag = False
for i_subject in subject_label:
    hypno = pd.read_csv(f"{data_path}{i_subject}_hypno.tsv", sep='\t')
    raw = mne.io.read_raw_edf(f"{data_path}{i_subject}.edf", preload=True, exclude=[channel_2_rem])

    # Select a subset of EEG channels
    # Look for the occurrence of LOC in the channels list "raw.ch_names" and pick only those
    LOC_chan = [load_chan for load_chan in raw.ch_names if LOC_label in load_chan]
    if not len(LOC_chan)==1:
        error_flag = True
        print('Error : LOC channel not found or found twice') 
        if len(LOC_chan)>1:
            LOC_chan = [LOC_chan[0]]
            print(f'LOC channel is forced to {LOC_chan}') 
        #raise ValueError('LOC channel not found or found twice')
    ROC_chan = [load_chan for load_chan in raw.ch_names if ROC_label in load_chan]
    if not len(ROC_chan)==1:
        error_flag = True
        if len(ROC_chan)>1:
            ROC_chan = [ROC_chan[0]]
            print(f'ROC channel is forced to {ROC_chan}') 
        #raise ValueError('ROC channel not found or found twice')
    channels_label = LOC_chan+ROC_chan
    raw.pick(channels_label)
    hypno = np.squeeze(hypno["name"].values)
    hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw._data[0,:], sf_data=raw.info['sfreq'])

    loc = raw._data[0,:] * 1e6
    roc = raw._data[1,:] * 1e6
    rem = yasa.rem_detect(loc, roc, raw.info['sfreq'], hypno=hypno_up, include=5, amplitude=(REM_amplitude_det, 325), duration=(0.3, 1.5), \
        freq_rem=(0.5, 5), relative_prominence=0.8, remove_outliers=True, verbose='info')
    # rem = yasa.rem_detect(loc, roc, raw.info['sfreq'], hypno=hypno_up, include=5, amplitude=(REM_amplitude_det, 325), duration=(0.3, 1.5), \
    #     freq_rem=(0.5, 5), relative_prominence=0.8, remove_outliers=False, verbose='info')

    # Save the REMs dataFrame in a tsv file
    rems_detection_df = rem.summary().round(3)
    rems_detection_df.to_csv(f"{data_path}{i_subject}_YASA_REMs_summary.tsv", sep='\t')
    # Modify the DataFrame to be compatible with Snooz
    #   Snooz dataframe : [group, name, start_sec, duration_sec, channels]
    #   Define 
    #       group as "YASA",
    #       name as "YASA_REM", 
    #       start_time as rems_detection_df['start'],
    #       duration as rems_detection_df['duration'],
    #       channel as channels_label
    snooz_rem = pd.DataFrame({
        'group': 'YASA',
        'name': 'YASA_REM',
        'start_sec': rems_detection_df['Start'],
        'duration_sec': rems_detection_df['Duration'],
        'channels': [channels_label] * len(rems_detection_df)
    })
    # Save the REMs dataFrame in a tsv file
    snooz_rem.to_csv(f"{data_path}{i_subject}_YASA_REMs_snooz.tsv", sep='\t', index=False)

if error_flag:
    raise ValueError