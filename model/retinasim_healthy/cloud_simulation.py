import numpy as np
import pickle
from brian2 import *
from model.retinasim_healthy.simulation import stim_to_retina_output
from model.retinasim_healthy.simulation_utils import SimulationParam
from scipy.ndimage import gaussian_filter1d

# generate cloud stimulus res * res * num_timesteps saved to save_directory
def generate_cloud(res, num_timesteps, save_directory, mu=0.5, sigma=0.5*0.35):
    s = np.random.normal(mu, sigma, (res, res, num_timesteps))
    samples = gaussian_filter1d(gaussian_filter1d(s, sigma=2, axis=0), sigma=2, axis=1)
    np.save(save_directory, samples)
    return samples

# scale the coordinates of the photoreceptors to the magnitude of the coordinates of the stimulus
# note: only works in the case where the stimulus is coarser than the photoreceptor layout
def convert_coord(res, sp):
    xs = np.load("{}/{}.npy".format(sp.xy_coord_folder, "CR"))[0]
    ys = np.load("{}/{}.npy".format(sp.xy_coord_folder, "CR"))[1]
    if sp.network_size_option == "mini":
        xs_shifted = (xs+100)/200*res
        ys_shifted = (ys+100)/200*res
    else:
        xs_shifted = (xs+150)/300*res
        ys_shifted = (ys+150)/300*res
    xs_int = np.array(np.rint(xs_shifted), dtype=int)
    ys_int = np.array(np.rint(ys_shifted), dtype=int)
    xs_int[np.where(xs_int == res)] = res - 1
    ys_int[np.where(ys_int == res)] = res - 1
    return xs_int, ys_int, len(xs_int)

def cloud_to_retina_output(simulation_param_txt, save_folder, res=32, select_GPU=None, genn_directory=None):

    np.random.seed(1234)
    num_timesteps_per_batch = 500
    num_batches = 100
    num_timesteps = num_timesteps_per_batch * num_batches
    cloud_dt = 50 # ms
    sp = SimulationParam(simulation_param_txt)

    light = generate_cloud(res, num_timesteps, save_folder+"/stimulus.npy")
    xs_int, ys_int, NUM_CR = convert_coord(res, sp)

    # 10 batches, each batch contains 2500 timesteps
    for batch_number in range(num_batches):
        print("simulation with batch {}".format(batch_number))
        batch_light = light[:,:,num_timesteps_per_batch*batch_number:num_timesteps_per_batch*(batch_number+1)]
        batch_light_cr = np.transpose(np.array([batch_light[xs_int[i], ys_int[i], :] for i in range(NUM_CR)]))
        stim = TimedArray(batch_light_cr, dt=cloud_dt*ms)
        _, _, _, _, _, _, _, _, _, spikes_gl_on, spikes_gl_off = stim_to_retina_output(num_timesteps_per_batch*cloud_dt, simulation_param_txt, lstim=stim, dt=0.1, select_GPU=select_GPU, genn_directory=genn_directory)
        with open('{}/spikes_gl_on_{}.pickle'.format(save_folder, batch_number), 'wb') as f:
            pickle.dump(spikes_gl_on, f)
        with open('{}/spikes_gl_off_{}.pickle'.format(save_folder, batch_number), 'wb') as f:
            pickle.dump(spikes_gl_off, f)
    
if __name__ == "__main__":
    simulation_param_txt = "parameter/cottaris_original_2hz.txt"
    save_folder = "data/data-cloud-stim-full-2hz"
    res = 48
    select_GPU = 1
    genn_directory = "GeNNworkspace_{}".format(select_GPU)
    cloud_to_retina_output(simulation_param_txt, save_folder, res=res, select_GPU=select_GPU, genn_directory=genn_directory)