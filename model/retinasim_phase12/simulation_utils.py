import numpy as np
import os
import pprint

# current is in uA and returns ve in millivolts
def get_ve(x0, y0, z0, a, x, y, z, current, imped):

    v0 = current * imped
    r = np.sqrt(np.power(x-x0, 2) + np.power(y-y0, 2))
    r = np.expand_dims(r, 2)
    d = z-z0
    d = np.expand_dims(d, 2)
    denominator = np.sqrt(np.power(r+a,2)+np.power(d,2))+np.sqrt(np.power(r-a,2)+np.power(d,2))
    ve = 2*v0/np.pi*np.arcsin(2*a/denominator)
    
    return ve

# uniformly randomly sample points on the sphere
# x, y, z should be in shape (num_cells, 1)
# current should be in shape (num_timesteps, 1)
# see https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
# also https://www-alg.ist.hokudai.ac.jp/~jan/randsphere.pdf
# also D. E. Knuth. The Art of Computer Programming, vol. 2: Seminumerical Algorithms.
def get_delta_ve(x0, y0, z0, a, x, y, z, r, current, imped, size=500):
   
    if current < 0:
        sign = 1
    else:
        sign = -1
            
    np.random.seed(1234) # set random seed to ensure reproducibility
    
    n = 3 # in 3 dimensions
    points = np.random.normal(size=(size, n)) 
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points = np.transpose(points)
    
    px = np.expand_dims(r*points[0], 0)
    py = np.expand_dims(r*points[1], 0)
    pz = np.expand_dims(r*points[2], 0)
    
    xs1, ys1, zs1 = x+px, y+py, z+pz
    xs2, ys2, zs2 = x-px, y-py, z-pz
    
    delta_ve = np.abs(get_ve(x0,y0,z0,a,xs1,ys1,zs1,current,imped)-get_ve(x0,y0,z0,a,xs2,ys2,zs2,current,imped))
    delta_ve = np.mean(delta_ve, axis=1)/2
    delta_ve = np.transpose(delta_ve)
    delta_ve = delta_ve*sign
    
    return delta_ve

def get_cell_delta_ve(name, estim, x0, y0, z0, a, imped, xy_coord_folder, z_coord_folder):
    x = np.load("{}/{}.npy".format(xy_coord_folder, name))[0]
    x = np.expand_dims(x, 1)
    y = np.load("{}/{}.npy".format(xy_coord_folder, name))[1]
    y = np.expand_dims(y, 1)
    z = np.squeeze(np.load("{}/{}.npy".format(z_coord_folder, name)))
    z = np.expand_dims(z, 1)
    # setting the soma radius according to different cell types
    if name == "GL_ON" or name == "GL_OFF":
        r = 13
    else:
        r = 3.5
    delta_ve = get_delta_ve(x0, y0, z0, a, x, y, z, r, estim, imped)
    return delta_ve

def save_delta_ve(imped, estim, x0, y0, z0, a, xy_coord_folder, z_coord_folder, save_folder, implant_mode, dt=0.01):
    '''
    Saves delta_ve (num_of_timesteps x num_of_cells) to disk for each cell type.
    Input: estim, a p2p pulse; x0, y0, z0: the coordinates of the electrode; a,
    the diameter of the electrode; xy_coord_folder: the folder containing 
    dendritic tree coordinates; z_coord_folder: the folder containing z 
    coordinates; save_folder, saving location; dt: estim will be downsampled 
    every dt ms (defaults to 0.01 ms)
    '''
    if implant_mode not in ["epiretinal", "subretinal", "cone_only", "bp_only"]:
        raise ValueError("The implant mode {} is not valid.".format(implant_mode))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    downsampled = estim[0, ::dt][0]
    u = np.unique(downsampled)
    cell_types = ["cr", "hrz", "bp_on", "bp_off", "am_wf_on", "am_wf_off", "am_nf_on", "gl_on", "gl_off"]
    for name in cell_types:
        if implant_mode == "subretinal" and name == "cr":
            continue
        if implant_mode == "cone_only" and name != "cr":
            continue
        if implant_mode == "bp_only" and (name != "bp_on" or name != "bp_off"):
            continue
        delta_ve_values_dict = {}
        for value in u:
            # round the floating point to use it as the key of the hashtable
            if round(value,4) not in delta_ve_values_dict:
                delta_ve_values_dict[round(value,4)] = get_cell_delta_ve(name.upper(), value, x0, y0, z0, a, imped, xy_coord_folder, z_coord_folder)
        delta_ve = [delta_ve_values_dict[round(current,4)] for current in downsampled]
        delta_ve = np.concatenate(delta_ve, axis=0)
        np.save("{}/{}.npy".format(save_folder, name.upper()), delta_ve)
        
def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError("Cannot covert {} to a bool".format(s))
        
class SimulationParam:
    
    def __init__(self, txt_file=None):
        
        # default values (calculation of electric potential)
        self.imped = 10 # retina tissue impedance in Cottaris (kOhm)
        self.electrode_x = 0 # the x-coordinate of the electrode position in Cottaris (um)
        self.electrode_y = 0 # the y-coordinate of the electrode position in Cottairs (um)
        self.electrode_z = -2 # the z-coordinate of the electrode position in Cottaris (um)
        self.electrode_diam = 80 # the diameter of the electrode in Cottaris (um)
        # epiretinal stimulates every cell class; 
        # subretinal stimulates every cell class except cones;
        # cone_only stimulates only cones;
        # bp_only stimulates only bipolar cells
        # implemented in the function save_delta_ve (skip the computation of delta_ve of the unstimulated cell classes) 
        # and in the function retina_simulation in retina_network.py (skip loading delta_ve of the unstimulated cell classes)
        self.implant_mode = "epiretinal"
        # the complete network (300 um by 300 um by 210 um) in Cottaris; 
        # the other option is "mini" (200 um by 200 um by 210 um)
        self.network_size_option = "full" 
        # coordinate folders will be set according to network_size_option
        self.xy_coord_folder = "dendritic-tree-coordinate"
        self.z_coord_folder = "z-coordinate"
        # by default cone cell class is included in the model
        self.cone_exists = True
        
        # default values (retinal network simulation)
        # neurons
        self.gl_on_leakage = 0.0206  # ON RGC's leakage conductance in Guo (msiemens/cm/cm)
        self.gl_off_leakage = 0.0479 # OFF RGC's leakage conductance in Guo (msiemens/cm/cm)
        self.gl_off_cond_h = 0.1429 # OFF RGC's hyperpolarization-activated current conductance in Guo (msiemens/cm/cm)
        self.gl_off_cond_cat = 0.1983 # OFF RGC's low-threshold voltage-activated calcium current conductance in Guo (msiemens/cm/cm)
        self.gl_off_tau_y_p1 = 588.2 # OFF RGC's hyperpolarization-activated current, the multiplicative factor in tau_h in Guo
        self.gl_off_tau_y_p2 = 10 # OFF RGC's hyperpolarization-activated current, the added number to V in tau_h in Guo
        self.gl_off_y_inf_p1 = 75 # OFF RGC's hyperpolarization-activated current, the added number to V in y_inf in Guo
        self.gl_off_cat_p1 = 28.8 # OFF RGC's low-threshold voltage-activated calcium current, the added number to V in alpha_mT and beta_mT in Guo
        self.gl_off_cat_p2 = 83.5 # OFF RGC's low-threshold voltage-activated calcium current, the added number to V in beta_hT, alpha_d and beta_d in Guo
        # synapses
        self.bp_on_gl_on_gmax = 2.5 # g_max in the synapse from ON bipolar to ON RGC in Cottaris
        self.bp_on_gl_on_sigma = 6 # sigma in the synapse from ON bipolar to ON RGC in Cottaris
        self.am_wf_on_gl_on_gmax = 2.0 # g_max in the synapse from ON wide-field amacrine to ON RGC in Cottaris
        self.am_wf_on_gl_on_factor = 1 # a multiplication factor before the synaptic input from ON wide-field amacrine to ON RGC
        self.bp_off_gl_off_gmax = 2.5 # g_max in the synapse from OFF bipolar to OFF RGC in Cottaris
        self.bp_off_gl_off_sigma = 6 # sigma in the synapse from OFF bipolar to OFF RGC in Cottaris
        self.am_wf_off_gl_off_gmax = 2.5 # g_max in the synapse from OFF wide-field amacrine to OFF RGC in Cottaris
        self.am_wf_off_gl_off_factor = 1 # a multiplication factor before the synaptic input from OFF wide-field amacrine to OFF RGC
        self.am_nf_on_gl_off_gmax = 2.0 # g_max in the synapse from ON narrow-field amacrine to OFF RGC in Cottaris
        self.am_nf_on_gl_off_factor = 1 # a multiplication factor before the synaptic input from ON narrow-field amacrine to OFF RGC
        
        # read txt file and replace the default values if necessary
        if txt_file:
            with open(txt_file) as file:
                for line in file:
                    split_line = line.rstrip().split(',')
                    if len(split_line) == 2:
                        if split_line[0] in vars(self).keys():
                            if split_line[0] == "network_size_option" or split_line[0] == "implant_mode":
                                setattr(self, split_line[0], split_line[1])
                            elif split_line[0] == "cone_exists":
                                setattr(self, split_line[0], str_to_bool(split_line[1]))
                            else:
                                setattr(self, split_line[0], float(split_line[1]))
                        else:
                            raise NameError("The attribute {} doesn't exist.".format(split_line[0]))
                        
        if self.network_size_option == "mini":
            self.xy_coord_folder = "dendritic-tree-coordinate-mini"
            self.z_coord_folder = "z-coordinate-mini"
    
    def __str__(self):
        return pprint.pformat(vars(self))
