from brian2 import *
import numpy as np
from model.retinasim_phase12.simulation_utils import save_delta_ve, SimulationParam
from model.retinasim_phase12.retina_network import retina_simulation

# sp can be either the path to a txt file containing the simulation parameters or a SimulationParam object
def stim_to_retina_output(time_in_ms, sp, light_g_max, pre_computed_weights_folder, lstim=None, estim=None, delta_ve_folder="delta_ve_workspace", dt=0.01, debug=False, gpu=True, select_GPU=None, genn_directory=None):
    
    start = time.time()
    
    if isinstance(sp, str):
        sp = SimulationParam(sp)
    NUM_CR = np.load("{}/CR.npy".format(sp.xy_coord_folder)).shape[1]
        
    # if light stimulus not provided, set up constant background light (0.5 intensity)
    if not lstim:
        lstim = TimedArray(np.ones((time_in_ms, NUM_CR)) * 0.5, dt=1*ms)
    
    if estim:
        save_delta_ve(sp.imped, estim, sp.electrode_x, sp.electrode_y, sp.electrode_z, sp.electrode_diam, sp.xy_coord_folder, sp.z_coord_folder, delta_ve_folder, sp.implant_mode, dt=dt)
        data = retina_simulation(time_in_ms, sp, lstim, light_g_max=light_g_max, pre_computed_weights_folder=pre_computed_weights_folder, delta_ve_folder=delta_ve_folder, delta_ve_dt=dt, simulation_timestep=dt, debug=debug, gpu=gpu, select_GPU=select_GPU, genn_directory=genn_directory)
    else:
        data = retina_simulation(time_in_ms, sp, lstim, light_g_max=light_g_max, pre_computed_weights_folder=pre_computed_weights_folder, delta_ve_folder=None, delta_ve_dt=dt, simulation_timestep=dt, debug=debug, gpu=gpu, select_GPU=select_GPU, genn_directory=genn_directory)

    end = time.time()
    print("simulation time: {}".format(end-start))
    
    return data

def estim_to_retina_output(time_in_ms, sp, light_g_max, estim, pre_computed_weights_folder, light_intensity=0.5, delta_ve_folder="delta_ve_workspace", dt=0.01, debug=False, gpu=True, select_GPU=None, genn_directory=None):
    
    start = time.time()
    
    if isinstance(sp, str):
        sp = SimulationParam(sp)
    NUM_CR = np.load("{}/CR.npy".format(sp.xy_coord_folder)).shape[1]
    
    lstim = TimedArray(np.ones((time_in_ms, NUM_CR)) * light_intensity, dt=1*ms)
    
    save_delta_ve(sp.imped, estim, sp.electrode_x, sp.electrode_y, sp.electrode_z, sp.electrode_diam, sp.xy_coord_folder, sp.z_coord_folder, delta_ve_folder, sp.implant_mode, dt=dt)
    data = retina_simulation(time_in_ms, sp, lstim, light_g_max=light_g_max, pre_computed_weights_folder=pre_computed_weights_folder, delta_ve_folder=delta_ve_folder, delta_ve_dt=dt, simulation_timestep=dt, debug=debug, gpu=gpu, select_GPU=select_GPU, genn_directory=genn_directory)

    end = time.time()
    print("simulation time: {}".format(end-start))
    
    return data