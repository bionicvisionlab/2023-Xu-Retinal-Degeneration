import pickle
import numpy as np
from brian2 import *
import sys

def bin_spikes(folder, cell_type, cell_index, num_of_batches):
    for i in range(num_of_batches):
        with open("{}/spikes_gl_{}_{}.pickle".format(folder, cell_type, i), 'rb') as f:
            spikes = pickle.load(f)['t'][cell_index]/ms
        # 25000 ms is the total time per batch, defined in cloud_simulation.py
        # 50 ms or 100 ms is each bin, which is the stimulus refresh rate in cloud_simulation.py
        binned = np.histogram(spikes, np.arange(0,25001,50))
        np.save("{}/{}_binned_{}_{}.npy".format(folder, i, cell_type, cell_index), binned[0])
