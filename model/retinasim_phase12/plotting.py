import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pickle
from brian2 import *

# https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def get_cmap():
    return get_continuous_cmap(['#141629', '#3A377A', '#828282', '#BE4151', '#F7E05C'])

def voltage_line_plot(data, vert_l=np.array([])):
    plt.figure(figsize=(15,5))
    for xc in vert_l:
        plt.axvline(x=xc, color='k', linestyle='--')
    plt.plot(data['t'], np.nan_to_num(data['V_value']))

# Cottaris Figure 5
def interp_y_t_plot_with_spike(path_to_interp_x_y, v_min, v_max, vert_l, path_to_spikes, cell_index):
    
    fig, axs = plt.subplots(2, 1, figsize=(15,4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    interp_x_y = np.load(path_to_interp_x_y)
    heatmap = axs[0].imshow(np.transpose(interp_x_y[:,:,50]), vmin=v_min, vmax=v_max, origin='lower', cmap=get_cmap(), aspect='auto', extent=[0,7000,0,100])
    axs[0].set_xticks([0,1000,2000,3000,4000,5000,6000,7000])
    axs[0].set_xticklabels([0,100,200,300,400,500,600,700])
    axs[0].set_yticks([0,20,40,60,80,100])
    axs[0].set_yticklabels([0,60,120,180,240,300])
    axs[0].set_ylabel('Space in y dimension (μm)', fontsize=10)
    for xc in vert_l:
        axs[0].axvline(x=xc*10, color='k', linestyle='--')
        
    with open(path_to_spikes, 'rb') as f:
        st = pickle.load(f)
    data = []
    for i in cell_index:
        st_example = st['t'][i]/ms
        st_example = (st_example[np.logical_and(st_example>300, st_example<1000)] - 300)*10
        data.append(st_example)
    data_c = np.concatenate([data], axis=0) # could break if the number of spikes are unequal
    axs[1].eventplot(data_c, colors='black', lineoffsets=1, linelengths=1)
    axs[1].yaxis.set_visible(False)
    axs[1].set_xlabel('Time (ms)', fontsize=10)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label('firing rate (Hz)', rotation=270)
    
    return fig, axs
    