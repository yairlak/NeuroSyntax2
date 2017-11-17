import numpy as np
import os
import neo
from neo.io import BlackrockIO
import mne
import matplotlib.pyplot as plt
import scipy.io

class settings:
    def __init__(self):
        self.path2data = os.path.join('..', 'Data', 'TA719_NeuroSyntax2_sEEG_files_a')
        self.path2behavData = os.path.join('..', 'Data', 'TA719_NeuroSyntax2_Behaviorial_Data')
        self.file_stem = '20160618-123112-001'
        self.blocks = ['ExperimentalRun__2016_6_18_12_35_39.mat',
                       'ExperimentalRun__2016_6_18_12_41_11.mat',
                       'ExperimentalRun__2016_6_18_12_48_39.mat',
                       'ExperimentalRun__2016_6_18_12_56_38.mat',
                       'ExperimentalRun__2016_6_18_13_3_5.mat',
                       'ExperimentalRun__2016_6_18_13_11_41.mat']
class params:
    def __init__(self):
        self.sfreq = 2000 # Data sampling frequency [Hz]
        self.line_frequency = 60  # Line frequency [Hz]
        self.tmin = -2  # Start time before event [sec]
        self.tmax = 2 # End time after event [sec]

def load_data(settings):
    reader = BlackrockIO(filename=os.path.join(settings.path2data, settings.file_stem), nsx_to_load=3)
    return reader

def get_TTLs(TTL_channel, settings):
    # Read paradigm info for all blocks
    num_blocks = len(settings.blocks)
    for block, block_filename in enumerate(settings.blocks):
        # Load behav file
        curr_block = scipy.io.loadmat(os.path.join(settings.path2behavData, block_filename))
        # get number of trial
        num_trials = len(curr_block['wordonset'])
        if block == 0:
            num_words_sentence = np.empty([num_blocks, num_trials], dtype='int64')
            num_words_elliptic = np.empty([num_blocks, num_trials], dtype='int64')

        for trial in range(num_trials):
            num_words_sentence[block, trial] = curr_block['wordonset'][trial][0].size
            num_words_elliptic[block, trial] = curr_block['wordonset'][trial][1].size

    num_TTL = num_trials + np.sum(num_words_sentence, axis=1) + np.sum(num_words_elliptic, axis=1)
    TTLs = np.empty([0,1])
    for i in range(1, len(TTL_channel)):
        if TTL_channel[i-1]<30000 and TTL_channel[i]>=30000:
            TTLs = np.vstack((TTLs, i))
    # TTL for fixation/first/last words of full and elliptic sentences
    # plt.scatter(TTLs, np.ones(len(TTLs)))
    # plt.show
    #
    events = np.empty([len(TTLs), 3], dtype='int32')
    cnt_word_items = 1 # count number of items in either fixation(should be always one)/sentence/ellptic
    cnt_type = 0 # identify whether fixation/sentence/elliptic
    cnt_block = 0
    trial = 0 # identifies current trial number
    for i, TTL in enumerate(TTLs):
        # find current block
        #block = np.min(np.where(np.cumsum(num_TTL) >= i+1)) # counting blocks from zero
        curr_num_items = [1, num_words_sentence[cnt_block, trial], num_words_elliptic[cnt_block, trial]]
        if curr_num_items[cnt_type] == 0:
            cnt_word_items, cnt_type, trial, cnt_block = update_counters(cnt_word_items, cnt_type, trial, cnt_block)
        # Generate the current event id. Each block has potenial 100 events (0-99, 100-199, etc.).
        # Within each block, each type (fix/sentence/elliptic) has 30 possible item slots
        event_id = cnt_block * 100 + cnt_type * 30 + cnt_word_items # For example, event_id = 60 means last word of the sentence
        events[i] = [TTL, 0, event_id]
        # Update counts
        if cnt_word_items == curr_num_items[cnt_type]: #if last word in sequence
            event_id = cnt_block * 100 + cnt_type * 30 + 30 # Change id to last word (=30)
            events[i] = [TTL, 0, event_id] # Change id to last word (=30)
            cnt_word_items, cnt_type, trial, cnt_block = update_counters(cnt_word_items, cnt_type, trial, cnt_block)
        else:
            cnt_word_items += 1

    #TTLs_subset = events[np.argwhere(events[:,2] == 60), 0]
    #plt.scatter(TTLs, np.ones(len(TTLs)))
    #plt.scatter(TTLs_subset, np.ones(len(TTLs_subset)), edgecolors = 'r')
    #plt.show

    return events

def update_counters(cnt_word_items, cnt_type, trial, cnt_block):
    cnt_type += 1  # forward the type counter
    cnt_type = cnt_type % 3  # Make sure it is 3-cyclic
    cnt_word_items = 1  # Start counting the token items from 1
    if cnt_type == 0: # If elliptic finished, and back to fixation then change the trial number
        trial += 1
        if trial == 40:  # If last trial (assuming always 40 trials in each block) then increase block number
            trial = 0
            cnt_block += 1
    return cnt_word_items, cnt_type, trial, cnt_block

def generate_mne_raw_object(data_all_channels, params):
    num_channels = data_all_channels.header['signal_channels'].size
    ch_types = ['seeg' for s in range(num_channels)]
    ch_names = ['sEEG_%s' % s for s in range(num_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=params.sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(np.asarray(data_all_channels.nsx_data[2]).transpose(), info)
    return raw
# Load settings (path, file names, etc.)
settings = settings()
# Load parameters (sampling rate, etc.)
params = params()
# Load data (BlackRock, Neuralynx, etc.)
print('Loading data...')
data_all_channels = load_data(settings)
# Get TTLs from last channel and generate events mne-object
print('Collecting TTLs...')
events = get_TTLs(data_all_channels.nsx_data[2][:,128], settings)
# ---------------------------------------------
# Epoch the data according to a given event_id
event_ids = [30, 60, 130, 160, 230, 260]  # Choose Block, type and word item to lock to
# List of freq bands
#iter_freqs = [('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45), ('High-Gamma', 70, 150)]
iter_freqs = [('High-Gamma', 70, 150)]
fstep = 2 # [Hz] Step in spectrogram

for event_id in event_ids:
    for band, fmin, fmax in iter_freqs:
        # Convert data to mne raw
        print('Generating MNE raw object...')
        raw = generate_mne_raw_object(data_all_channels, params)
        raw.notch_filter(params.line_frequency, fir_design='firwin')

        # bandpass filter
        #raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
        #           l_trans_bandwidth=1,  # make sure filter params are the same
        #           h_trans_bandwidth=1,  # in each band and skip "auto" option.
        #           fir_design='firwin')
        # raw.apply_hilbert(n_jobs=1, envelope=False)
        # Epoch data
        epochs = mne.Epochs(raw, events, event_id, params.tmin, params.tmax, baseline=None, preload=True)
        # remove evoked response and get analytic signal (envelope)
        for channel in range(128):
            file_name = 'ERP_Patient_' + settings.file_stem + '_Channel_' + str(channel+1) + '_Event_id' + str(
                event_id) + '.png'
            epochs.plot_image(picks=channel, show = False)
            fig = plt.gcf()
            fig.savefig(os.path.join('..', 'Figures', file_name))
            plt.close(fig)

            file_name = 'Spec_Patient_' + settings.file_stem + '_Channel_' + str(channel + 1) + '_Event_id' + str(
                event_id) + '.png'
            epochs.plot_psd(picks=[channel], show = False)
            fig = plt.gcf()
            fig.savefig(os.path.join('..', 'Figures', file_name))
            plt.close(fig)

            freqs = np.arange(fmin, fmax, fstep)
            n_cycles = freqs / 2
            file_name = 'ERF_Patient_' + settings.file_stem + '_Channel_' + str(channel + 1) + '_Event_id' + str(
                event_id) + '.png'
            power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, average=False, n_cycles=n_cycles, return_itc=False, picks = [channel])
            power_ave = np.squeeze(np.average(power.data, axis=2))
            fig, ax = plt.subplots(figsize=(6, 6))
            map = ax.imshow(power_ave, extent=[np.min(power.times), np.max(power.times), 1, 40], interpolation='nearest', aspect='auto')
            plt.colorbar(map, label = 'Power')
            fig.savefig(os.path.join('..', 'Figures', file_name))
            plt.close(fig)