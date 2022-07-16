from brian2 import *
import pickle
import numpy as np
from scipy import interpolate

def interp_wrapper(coord, data, method="rbf"):
    
    xi, yi = np.linspace(-150, 150, 100), np.linspace(-150, 150, 100)
    xi, yi = np.meshgrid(xi, yi)
    
    x_y_t = []
    
    if method == "cubic":
        coord = np.transpose(coord)
        for t in range(pol.shape[0]):
            zi = interpolate.griddata(coord, data[t], (xi, yi), method='cubic', fill_value=0)
            x_y_t.append(np.expand_dims(zi, axis=0))
            if t % 100 == 0:
                print(".", end="")
            
    elif method == "linear":
        coord = np.transpose(coord)
        for t in range(pol.shape[0]):
            zi = interpolate.griddata(coord, data[t], (xi, yi), method='linear', fill_value=0)
            x_y_t.append(np.expand_dims(zi, axis=0))
            if t % 100 == 0:
                print(".", end="")
            
    else:
        for t in range(data.shape[0]):
            rbf = interpolate.Rbf(coord[0], coord[1], data[t], function='linear')
            zi = rbf(xi, yi)
            x_y_t.append(np.expand_dims(zi, axis=0))
            if t % 100 == 0:
                print(".", end="")
        
    x_y_t_array = np.concatenate(x_y_t, axis=0)
    
    return x_y_t_array