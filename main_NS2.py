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
        self.tmin = -5  # Start time before event [sec]
        self.tmax = 1 # End time after event [sec]

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
    trial = 0 # identifies current trial number
    for i, TTL in enumerate(TTLs):
        # find current block
        block = np.min(np.where(np.cumsum(num_TTL) >= i+1)) # counting blocks from zero
        curr_num_items = [1, num_words_sentence[block, trial], num_words_elliptic[block, trial]]
        # Generate the current event id. Each block has potenial 100 events (0-99, 100-199, etc.).
        # Within each block, each type (fix/sentence/elliptic) has 30 possible item slots
        event_id = block * 100 + cnt_type * 30 + cnt_word_items
        events[i] = [TTL, 0, event_id]
        # Update counts
        if cnt_word_items == curr_num_items[cnt_type]:
            # Change id to last word (=30)
            event_id = block * 100 + cnt_type * 30 + 30 # For example, event_id = 60 means last word of the sentence
            events[i] = [TTL, 0, event_id]
            cnt_type += 1 # forward the type counter
            cnt_type = cnt_type % 3 # Make sure it is 3-cyclic
            cnt_word_items = 1 # Start counting the token items from 1
            if cnt_type == 0:
                trial += 1
                trial = trial % 40 # Assuming always 40 trials in each block
        else:
            cnt_word_items += 1



    return events

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
data_all_channels = load_data(settings)
# Get TTLs from last channel and generate events mne-object
events = get_TTLs(data_all_channels.nsx_data[2][:,128], settings)

# ---------------------------------------------
# Epoch the data according to a given event_id
event_id = 160 # Choose Block, type and word item to lock to
picks = [1, 20, 30, 40, 50] # Choose channels
# List of freq bands
#iter_freqs = [('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45), ('High-Gamma', 70, 150)]
iter_freqs = [('High-Gamma', 70, 150)]

for band, fmin, fmax in iter_freqs:
    # Convert data to mne raw
    raw = generate_mne_raw_object(data_all_channels, params)
    raw.notch_filter(params.line_frequency)

    # bandpass filter and compute Hilbert
    raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1,  # in each band and skip "auto" option.
               fir_design='firwin')
    #raw.apply_hilbert(n_jobs=1, envelope=False)
    # Epoch data
    epochs = mne.Epochs(raw, events, event_id, params.tmin, params.tmax, baseline=None, preload=True)
    # remove evoked response and get analytic signal (envelope)
    epochs.plot_image(picks = picks)
