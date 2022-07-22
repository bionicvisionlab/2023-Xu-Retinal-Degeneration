from brian2 import *
import numpy as np
from retinasim.simulation_utils import save_delta_ve, SimulationParam
from retinasim.retina_network_fi import retina_simulation

# sp can be either the path to a txt file containing the simulation parameters or a SimulationParam object
def fi_curve_simulation(time_in_ms, sp, injected_current_on, injected_current_off, light_g_max, pre_computed_weights_folder, lstim=None, estim=None, delta_ve_folder="delta_ve_workspace", dt=0.1, debug=False, gpu=True, select_GPU=None, genn_directory=None):
    
    start = time.time()
    
    if isinstance(sp, str):
        sp = SimulationParam(sp)
    NUM_CR = np.load("{}/CR.npy".format(sp.xy_coord_folder)).shape[1]
        
    # set up constant background light (0.5 intensity)
    lstim = TimedArray(np.ones((time_in_ms, NUM_CR)) * 0.5, dt=1*ms)

    data = retina_simulation(time_in_ms, sp, injected_current_on, injected_current_off, lstim, light_g_max=light_g_max, pre_computed_weights_folder=pre_computed_weights_folder, delta_ve_folder=None, delta_ve_dt=dt, simulation_timestep=dt, debug=debug, gpu=gpu, select_GPU=select_GPU, genn_directory=genn_directory)

    end = time.time()
    print("simulation time: {}".format(end-start))
    
    return data