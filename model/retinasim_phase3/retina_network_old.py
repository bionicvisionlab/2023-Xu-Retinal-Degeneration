from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
from scipy.spatial import distance_matrix
import brian2genn
import shutil

# parameters for synapses from neuron group 1 to neuron group 2
def get_distances(path_to_coord_1, path_to_coord_2, sigma_2, pre_computed_weights=None):
    coord_1 = np.load(path_to_coord_1)
    num_1 = coord_1.shape[1]
    coord_2 = np.load(path_to_coord_2)
    dist_mat = distance_matrix(np.transpose(coord_1), np.transpose(coord_2))
    D = dist_mat.flatten()
    if pre_computed_weights:
        weights = np.load(pre_computed_weights)
    else:
        weights = np.sum(np.exp(-dist_mat/sigma_2), axis=0)
    W = np.tile(weights, num_1)
    return D, W

# initial membrane voltages are the resting voltage when light intensity is 0.5
def retina_simulation(time_in_ms, sp, lstim, light_g_max=0.9, pre_computed_weights_folder=None, delta_ve_folder=None, delta_ve_dt=0.01, simulation_timestep=None, debug=False, gpu=True, select_GPU=None, genn_directory=None):
    
    xy_coord_folder = sp.xy_coord_folder
    if sp.cone_exists:
        NUM_CR = np.load("{}/CR.npy".format(xy_coord_folder)).shape[1]
        NUM_HRZ = np.load("{}/HRZ.npy".format(xy_coord_folder)).shape[1]
    NUM_BP_ON = np.load("{}/BP_ON.npy".format(xy_coord_folder)).shape[1]
    NUM_BP_OFF = np.load("{}/BP_OFF.npy".format(xy_coord_folder)).shape[1]
    NUM_AM_WF_ON = np.load("{}/AM_WF_ON.npy".format(xy_coord_folder)).shape[1]
    NUM_AM_WF_OFF = np.load("{}/AM_WF_OFF.npy".format(xy_coord_folder)).shape[1]
    NUM_AM_NF_ON = np.load("{}/AM_NF_ON.npy".format(xy_coord_folder)).shape[1]
    NUM_GL_ON = np.load("{}/GL_ON.npy".format(xy_coord_folder)).shape[1]
    NUM_GL_OFF = np.load("{}/GL_OFF.npy".format(xy_coord_folder)).shape[1]
    
    # set random seed to ensure reproducibility
    np.random.seed(100)
    
    BrianLogger.log_level_error()

    if genn_directory:
        set_device('genn', directory=genn_directory, use_GPU=gpu)
    else:
        set_device('genn', use_GPU=gpu)
        
    # select specific GPU if the index is provided
    if select_GPU is not None:
        prefs.devices.genn.cuda_backend.device_select = 'MANUAL'
        prefs.devices.genn.cuda_backend.manual_device = select_GPU
    
    if simulation_timestep:
        defaultclock.dt = simulation_timestep * ms

    # loading delta_ve for each cell type
    # if not provided, default to no electrical stimulation
    if delta_ve_folder:
        if sp.implant_mode == "epiretinal":
            if sp.cone_exists:
                delta_ve_cr = TimedArray(np.load("{}/CR.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
                delta_ve_hrz = TimedArray(np.load("{}/HRZ.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_bp_on = TimedArray(np.load("{}/BP_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_bp_off = TimedArray(np.load("{}/BP_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_wf_on = TimedArray(np.load("{}/AM_WF_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_wf_off = TimedArray(np.load("{}/AM_WF_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_nf_on = TimedArray(np.load("{}/AM_NF_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_on = TimedArray(np.load("{}/GL_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_off = TimedArray(np.load("{}/GL_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
        elif sp.implant_mode == "subretinal":
            if sp.cone_exists:
                delta_ve_hrz = TimedArray(np.load("{}/HRZ.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_bp_on = TimedArray(np.load("{}/BP_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_bp_off = TimedArray(np.load("{}/BP_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_wf_on = TimedArray(np.load("{}/AM_WF_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_wf_off = TimedArray(np.load("{}/AM_WF_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_am_nf_on = TimedArray(np.load("{}/AM_NF_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_on = TimedArray(np.load("{}/GL_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_off = TimedArray(np.load("{}/GL_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
        elif sp.implant_mode == "cone_only":
            if sp.cone_exists:
                delta_ve_cr = TimedArray(np.load("{}/CR.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            else:
                raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        elif sp.implant_mode == "bp_only":
            delta_ve_bp_on = TimedArray(np.load("{}/BP_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_bp_off = TimedArray(np.load("{}/BP_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
        else:
            raise ValueError("The implant mode {} is not valid.".format(sp.implant_mode))

    # cone
    if sp.cone_exists:
        cone_eqs = '''
        I_syn = G_syn*nsiemens * (E_syn - V) : amp
        I_e = G_m * delta_V_e / 2 : amp
        dV/dt = (-V+V_s)/tau : volt
        V_s = (G_m*(E_rest+delta_V_e/2) + (G_syn*nsiemens)*E_syn + G_syn_l*E_syn_l)/G_total : volt
        G_total = G_m + (G_syn*nsiemens) + G_syn_l : siemens
        tau = C_m/G_total : second
        C_m : farad
        G_m : siemens
        E_rest : volt
        E_syn : volt
        E_syn_l : volt
        G_syn : 1
        G_syn_l : siemens
        delta_V_e : volt
        V_delayed : 1
        V_value = V/mV : 1
        syn_delay : second
        '''
        cr = NeuronGroup(NUM_CR, cone_eqs, method='exponential_euler')
        cr.C_m = 80 * pfarad
        cr.G_m = 4.0 * nsiemens
        cr.E_rest = -50 * mV
        cr.V = np.random.randn(NUM_CR)*3*mV-46.8*mV
        cr.E_syn = -67 * mV
        cr.E_syn_l = -8 * mV

        cr.delta_V_e = 0 * mV
        cr.G_syn = 0
        cr.G_syn_l = 0 * nsiemens
        cr.syn_delay = 35 * ms
        cr.V_delayed = 0

    universal_eqs = '''
    I_syn = G_syn*nsiemens * (E_syn - V) : amp
    I_e = G_m * delta_V_e / 2 : amp
    dV/dt = (-V+V_s)/tau : volt
    V_s = (G_m*(E_rest+delta_V_e/2) + (G_syn*nsiemens)*E_syn)/G_total : volt
    G_total = G_m + (G_syn*nsiemens) : siemens
    tau = C_m/G_total : second
    C_m : farad (constant)
    G_m : siemens (constant)
    E_rest : volt (constant)
    E_syn : volt (constant)
    G_syn : 1
    delta_V_e : volt
    V_delayed : 1
    V_value = V/mV : 1
    syn_delay : second
    '''
    
    bp_universal_eqs = '''
    I_syn = G_syn*nsiemens * (E_syn - V) : amp
    I_e = G_m * delta_V_e / 2 : amp
    dV/dt = (-V+V_s)/tau : volt
    V_s = (G_m*(E_rest+delta_V_e/2) + (G_syn*nsiemens)*E_syn)/G_total : volt
    G_total = G_m + (G_syn*nsiemens) : siemens
    tau = C_m/G_total : second
    C_m : farad (constant)
    G_m : siemens (constant)
    E_rest : volt (constant)
    E_syn : volt (constant)
    G_syn : 1
    delta_V_e : volt
    V_delayed_am : 1
    V_delayed_gl : 1
    V_value = V/mV : 1
    syn_delay_am : second
    syn_delay_gl : second
    '''

    # horizontal
    if sp.cone_exists:
        hrz = NeuronGroup(NUM_HRZ, universal_eqs, method='exponential_euler')
        hrz.C_m = 210 * pfarad
        hrz.G_m = 2.5 * nsiemens
        hrz.E_rest = -65 * mV
        hrz.V = np.random.randn(NUM_HRZ)*3*mV-47.7*mV
        hrz.E_syn = 0 * mV

        hrz.delta_V_e = 0 * mV
        hrz.G_syn = 0
        hrz.syn_delay = 35 * ms
        hrz.V_delayed = 0

    # bipolar on
    bp_on = NeuronGroup(NUM_BP_ON, bp_universal_eqs, method='exponential_euler')
    bp_on.C_m = 50 * pfarad
    bp_on.G_m = 2.0 * nsiemens
    bp_on.E_rest = -45 * mV
    bp_on.V = np.random.randn(NUM_BP_ON)*3*mV-35*mV
    bp_on.E_syn = 0 * mV
    
    bp_on.delta_V_e = 0 * mV
    bp_on.G_syn = 0
    # synaptic delay of amacrine connections
    bp_on.syn_delay_am = 31 * ms
    bp_on.V_delayed_am = 0
    # synaptic delay of ganglion connections
    bp_on.syn_delay_gl = 35 * ms
    bp_on.V_delayed_gl = 0

    # bipolar off
    bp_off = NeuronGroup(NUM_BP_OFF, bp_universal_eqs, method='exponential_euler')
    bp_off.C_m = 50 * pfarad
    bp_off.G_m = 2.0 * nsiemens
    bp_off.E_rest = -45 * mV
    bp_off.V = np.random.randn(NUM_BP_OFF)*3*mV-44*mV
    bp_off.E_syn = 0 * mV
    
    bp_off.delta_V_e = 0 * mV
    bp_off.G_syn = 0
    # synaptic delay of amacrine connections
    bp_off.syn_delay_am = 31 * ms
    bp_off.V_delayed_am = 0
    # synaptic delay of ganglion connections
    bp_off.syn_delay_gl = 35 * ms
    bp_off.V_delayed_gl = 0

    # amacrine wide field on
    am_wf_on = NeuronGroup(NUM_AM_WF_ON, universal_eqs, method='exponential_euler')
    am_wf_on.C_m = 50 * pfarad
    am_wf_on.G_m = 2.0 * nsiemens
    am_wf_on.E_rest = -50 * mV
    am_wf_on.V = np.random.randn(NUM_AM_WF_ON)*3*mV-42*mV
    am_wf_on.E_syn = 0 * mV
    
    am_wf_on.delta_V_e = 0 * mV
    am_wf_on.G_syn = 0
    am_wf_on.syn_delay = 33 * ms
    am_wf_on.V_delayed = 0

    # amacrine wide field off
    am_wf_off = NeuronGroup(NUM_AM_WF_OFF, universal_eqs, method='exponential_euler')
    am_wf_off.C_m = 50 * pfarad
    am_wf_off.G_m = 2.0 * nsiemens
    am_wf_off.E_rest = -50 * mV
    am_wf_off.V = np.random.randn(NUM_AM_WF_OFF)*3*mV-34.5*mV
    am_wf_off.E_syn = 0 * mV
    
    am_wf_off.delta_V_e = 0 * mV
    am_wf_off.G_syn = 0
    am_wf_off.syn_delay = 33 * ms
    am_wf_off.V_delayed = 0

    # amacrine narrow field on
    am_nf_on = NeuronGroup(NUM_AM_NF_ON, universal_eqs, method='exponential_euler')
    am_nf_on.C_m = 50 * pfarad
    am_nf_on.G_m = 2.0 * nsiemens
    am_nf_on.E_rest = -50 * mV
    am_nf_on.V = np.random.randn(NUM_AM_NF_ON)*3*mV-47.6*mV
    am_nf_on.E_syn = 0 * mV
    
    am_nf_on.delta_V_e = 0 * mV
    am_nf_on.G_syn = 0
    am_nf_on.syn_delay = 33 * ms
    am_nf_on.V_delayed = 0

    # ganglion on
    gl_on_eqs = '''
    dV/dt = (1/C_m) * (I_e + I_syn - I_m) : volt
    # extracelllar current
    I_e = G_m * delta_V_e / 2 : amp
    # synaptic current
    I_syn_a = G_syn_a*nsiemens * (E_syn_a - V) : amp
    I_syn_b = G_syn_b*nsiemens * (E_syn_b - V) : amp
    I_syn = factor * I_syn_a + I_syn_b : amp
    # membrane current I_m
    I_m = I_Na + I_K + I_KA + I_Ca + I_K_Ca + I_h + I_CaT + I_L : amp
    # sodium
    I_Na = g_Na * sa * m**3 * h * (V - 35*mV) : amp
    dm/dt = alpha_m * (1-m) - beta_m * m : 1
    dh/dt = alpha_h * (1-h) - beta_h * h : 1
    alpha_m = 3.041/exprel(-0.1*(V/mV+30))/ms : Hz
    beta_m = 20 * exp(-(V/mV+55)/18)/ms : Hz
    alpha_h = 0.4 * exp(-(V/mV+50)/20)/ms : Hz
    beta_h = 6/(1+exp(-0.1*(V/mV+20)))/ms : Hz
    # calcium
    I_Ca = g_Ca * sa * c**3 * (V - E_Ca) : amp
    dc/dt = alpha_c * (1-c) - beta_c * c : 1
    alpha_c = 1.5/exprel(-0.1*(V/mV+13))/ms : Hz
    beta_c = 10 * exp(-(V/mV + 38)/18)/ms : Hz
    dCa/dt = -3/(2*F*radius) * (I_Ca + I_CaT) / sa - (Ca-0.0001*mM)/(13.75*ms) : mmolar
    E_Ca = R*T/(2*F) * log(1.8*mM/Ca) : volt  
    # non-inactivating potassium (delayed rectifier)
    I_K = g_K * sa * n**4 * (V + 72*mV) : amp
    dn/dt = alpha_n * (1-n) - beta_n * n : 1
    alpha_n = 0.2/exprel(-0.1*(V/mV+40))/ms : Hz
    beta_n = 0.4*exp(-(V/mV+50)/80)/ms : Hz
    # inactivating potassium (A type)
    I_KA = g_KA * sa * A**3 * h_A * (V + 72*mV) : amp
    dA/dt = alpha_A * (1-A) - beta_A * A : 1
    dh_A/dt = alpha_h_A * (1-h_A) - beta_h_A * h_A : 1
    alpha_A = 0.03/exprel(-0.1*(V/mV+90))/ms : Hz
    beta_A = 0.1 * exp(-(V/mV+30)/10)/ms : Hz
    alpha_h_A = 0.002 * exp(-(V/mV+70)/20)/ms : Hz
    beta_h_A = 0.03/(1+exp(-0.1*(V/mV+40)))/ms : Hz
    # calcium-activated potassium
    I_K_Ca = g_K_Ca * sa * (V + 72*mV) : amp
    g_K_Ca = g_K_Ca_max * (Ca/(0.001*mM))**2 / (1+(Ca/(0.001*mM))**2) : siemens/meter**2
    # hyperpolarization-activated non-selective cationic current
    I_h = g_h * sa * y * (V + 45.8*mV) : amp
    dy/dt = (1/tau_y) * (y_inf - y) : 1
    tau_y = 4649*(exp(0.01*(V/mV+20)))/(1+exp(0.2*(V/mV+20)))*ms : second
    y_inf = 1/(1+exp((V/mV+75)/5.5)) : 1
    # low-threshold voltage-activated calcium current (j_T is d_T in the paper)
    I_CaT = g_CaT * sa * m_T**3 * h_T * (V - E_Ca) : amp
    dm_T/dt = alpha_m_T * (1-m_T) - beta_m_T * m_T : 1
    dh_T/dt = alpha_h_T * (1-h_T-j_T) - beta_h_T * h_T : 1
    dj_T/dt = alpha_j_T * (1-h_T-j_T) - beta_j_T * j_T : 1
    alpha_m_T = 1/(1.7+exp(-(V/mV+28.8)/13.5))/ms : Hz
    beta_m_T = (1+exp(-(V/mV+63)/7.8))/(1.7+exp(-(V/mV+28.8)/13.5))/ms : Hz
    alpha_h_T = exp(-(V/mV+160.3)/17.8)/ms : Hz
    beta_h_T = alpha_h_T * (sqrt(0.25+exp((V/mV+83.5)/6.3))-0.5) : Hz
    alpha_j_T = (1+exp((V/mV+37.4)/30))/(240*(0.5+sqrt(0.25+exp((V/mV+83.5)/6.3))))/ms : Hz
    beta_j_T = alpha_j_T * sqrt(0.25+exp((V/mV+83.5)/6.3)) : Hz
    # leakage current
    I_L = g_L * sa * (V + 66.5*mV) : amp
    # other properties
    radius : meter (constant)
    sa = 4*pi*radius**2 : meter**2
    celsius_temp = 37 : 1
    T = celsius_temp*kelvin + zero_celsius : kelvin
    # parameters
    C_m : farad (constant)
    G_m : siemens (constant)
    E_syn_a : volt (constant)
    G_syn_a : 1
    E_syn_b : volt (constant)
    G_syn_b : 1
    factor : 1 (constant)
    delta_V_e : volt
    g_Na : siemens/meter**2 (constant)
    g_Ca : siemens/meter**2 (constant)
    g_K : siemens/meter**2 (constant)
    g_KA : siemens/meter**2 (constant)
    g_K_Ca_max : siemens/meter**2 (constant)
    g_h: siemens/meter**2 (constant)
    g_CaT : siemens/meter**2 (constant)
    g_L : siemens/meter**2 (constant)
    '''
    
    gl_on = NeuronGroup(NUM_GL_ON, gl_on_eqs, method='exponential_euler', threshold="V>-10*mV", refractory="V>-10*mV")
    gl_on.C_m = 50 * pfarad
    gl_on.G_m = 2.0 * nsiemens
    gl_on.E_syn_a = -70 * mV
    gl_on.E_syn_b = 0 * mV
    
    gl_on.delta_V_e = 0 * mV
    gl_on.G_syn_a = 0
    gl_on.G_syn_b = 0
    gl_on.factor = sp.am_wf_on_gl_on_factor

    # AIS (Guo)
    gl_on.g_Na = 1072 * msiemens /cm/cm
    gl_on.g_K = 40.5 * msiemens /cm/cm
    gl_on.g_KA = 94.5 * msiemens /cm/cm
    gl_on.g_Ca = 2.1 * msiemens /cm/cm
    gl_on.g_K_Ca_max = 0.04 * msiemens /cm /cm
    gl_on.g_h = 0.4287 * msiemens/cm/cm
    gl_on.g_CaT = 0.008 * msiemens/cm/cm
    gl_on.g_L = sp.gl_on_leakage * msiemens/cm/cm
    
    gl_on.radius = 13 * um

    # initial values

    rest_V = -66.5 * mV # reversal potential for the leakage current (Guo)
    init_V = np.random.randn(NUM_GL_ON)*3*mV+rest_V
    gl_on.V = init_V

    gl_on.m = 0.5
    gl_on.h = 0.5
    gl_on.c = 0.5
    gl_on.n = 0.5
    gl_on.A = 0
    gl_on.h_A = 1
    gl_on.y = 0.5
    gl_on.m_T = 0
    gl_on.h_T = 1
    gl_on.j_T = 1

    gl_on.Ca = 0.0001 * nM

    # ganglion off
    gl_off_eqs = '''
    dV/dt = (1/C_m) * (I_e + I_syn - I_m) : volt
    # extracelllar current
    I_e = G_m * delta_V_e / 2 : amp
    # synaptic current
    I_syn = factor_w * G_syn_aw*nsiemens * (E_syn_aw - V) + factor_n * G_syn_an*nsiemens * (E_syn_an - V) + G_syn_b*nsiemens * (E_syn_b - V) : amp
    # membrane current I_m
    I_m = I_Na + I_K + I_KA + I_Ca + I_K_Ca + I_h + I_CaT + I_L : amp
    # sodium
    I_Na = g_Na * sa * m**3 * h * (V - 35*mV) : amp
    dm/dt = alpha_m * (1-m) - beta_m * m : 1
    dh/dt = alpha_h * (1-h) - beta_h * h : 1
    alpha_m = 6/exprel(-0.1*(V/mV+30))/ms : Hz
    beta_m = 20 * exp(-(V/mV+55)/18)/ms : Hz
    alpha_h = 0.4 * exp(-(V/mV+50)/20)/ms : Hz
    beta_h = 6/(1+exp(-0.1*(V/mV+20)))/ms : Hz
    # calcium
    I_Ca = g_Ca * sa * c**3 * (V - E_Ca) : amp
    dc/dt = alpha_c * (1-c) - beta_c * c : 1
    alpha_c = 1.5/exprel(-0.1*(V/mV+13))/ms : Hz
    beta_c = 10 * exp(-(V/mV + 38)/18)/ms : Hz
    dCa/dt = -3/(2*F*radius) * (I_Ca + I_CaT) / sa - (Ca-0.0001*mM)/(55*ms) : mmolar
    E_Ca = R*T/(2*F) * log(1.8*mM/Ca) : volt  
    # non-inactivating potassium (delayed rectifier)
    I_K = g_K * sa * n**4 * (V + 68*mV) : amp
    dn/dt = alpha_n * (1-n) - beta_n * n : 1
    alpha_n = 0.2/exprel(-0.1*(V/mV+40))/ms : Hz
    beta_n = 0.4*exp(-(V/mV+50)/80)/ms : Hz
    # inactivating potassium (A type)
    I_KA = g_KA * sa * A**3 * h_A * (V + 68*mV) : amp
    dA/dt = alpha_A * (1-A) - beta_A * A : 1
    dh_A/dt = alpha_h_A * (1-h_A) - beta_h_A * h_A : 1
    alpha_A = 0.03/exprel(-0.1*(V/mV+90))/ms : Hz
    beta_A = 0.1 * exp(-(V/mV+30)/10)/ms : Hz
    alpha_h_A = 0.04 * exp(-(V/mV+70)/20)/ms : Hz
    beta_h_A = 0.6/(1+exp(-0.1*(V/mV+40)))/ms : Hz
    # calcium-activated potassium
    I_K_Ca = g_K_Ca * sa * (V + 68*mV) : amp
    g_K_Ca = g_K_Ca_max * (Ca/(0.001*mM))**2 / (1+(Ca/(0.001*mM))**2) : siemens/meter**2
    # hyperpolarization-activated non-selective cationic current
    I_h = g_h * sa * y * (V + 26.8*mV) : amp
    dy/dt = (1/tau_y) * (y_inf - y) : 1
    tau_y = {tau_y_p1}*(exp(0.01*(V/mV+{tau_y_p2})))/(1+exp(0.2*(V/mV+{tau_y_p2})))*ms : second
    y_inf = 1/(1+exp((V/mV+{y_inf_p1})/5.5)) : 1 
    # low-threshold voltage-activated calcium current (j_T is d_T in the paper)
    I_CaT = g_CaT * sa * m_T**3 * h_T * (V - E_Ca) : amp
    dm_T/dt = alpha_m_T * (1-m_T) - beta_m_T * m_T : 1
    dh_T/dt = alpha_h_T * (1-h_T-j_T) - beta_h_T * h_T : 1
    dj_T/dt = alpha_j_T * (1-h_T-j_T) - beta_j_T * j_T : 1
    alpha_m_T = 1/(1.7+exp(-(V/mV+{cat_p1})/13.5))/ms : Hz
    beta_m_T = (1+exp(-(V/mV+63)/7.8))/(1.7+exp(-(V/mV+{cat_p1})/13.5))/ms : Hz
    alpha_h_T = exp(-(V/mV+160.3)/17.8)/ms : Hz
    beta_h_T = alpha_h_T * (sqrt(0.25+exp((V/mV+{cat_p2})/6.3))-0.5) : Hz
    alpha_j_T = (1+exp((V/mV+37.4)/30))/(240*(0.5+sqrt(0.25+exp((V/mV+{cat_p2})/6.3))))/ms : Hz
    beta_j_T = alpha_j_T * sqrt(0.25+exp((V/mV+{cat_p2})/6.3)) : Hz
    # leakage current
    I_L = g_L * sa * (V + 70.5*mV) : amp
    # other properties
    radius : meter (constant)
    sa = 4*pi*radius**2 : meter**2
    celsius_temp = 37 : 1
    T = celsius_temp*kelvin + zero_celsius : kelvin
    # parameters
    C_m : farad (constant)
    G_m : siemens (constant)
    E_syn_aw : volt (constant)
    G_syn_aw : 1
    factor_w : 1 (constant)
    E_syn_an : volt (constant)
    G_syn_an : 1
    factor_n : 1 (constant)
    E_syn_b : volt (constant)
    G_syn_b : 1
    delta_V_e : volt
    g_Na : siemens/meter**2 (constant)
    g_Ca : siemens/meter**2 (constant)
    g_K : siemens/meter**2 (constant)
    g_KA : siemens/meter**2 (constant)
    g_K_Ca_max : siemens/meter**2 (constant)
    g_h: siemens/meter**2 (constant)
    g_CaT : siemens/meter**2 (constant)
    g_L : siemens/meter**2 (constant)
    '''.format(tau_y_p1=sp.gl_off_tau_y_p1, tau_y_p2=sp.gl_off_tau_y_p2, y_inf_p1=sp.gl_off_y_inf_p1, cat_p1=sp.gl_off_cat_p1, cat_p2=sp.gl_off_cat_p2)
    
    gl_off = NeuronGroup(NUM_GL_OFF, gl_off_eqs, method='exponential_euler', threshold="V>-10*mV", refractory="V>-10*mV")
    gl_off.C_m = 50 * pfarad
    gl_off.G_m = 2.0 * nsiemens
    gl_off.E_syn_aw = -70 * mV
    gl_off.E_syn_an = -80 * mV
    gl_off.E_syn_b = 0 * mV
    
    gl_off.delta_V_e = 0 * mV
    gl_off.G_syn_aw = 0
    gl_off.factor_w = sp.am_wf_off_gl_off_factor
    gl_off.G_syn_an = 0
    gl_off.factor_n = sp.am_nf_on_gl_off_factor
    gl_off.G_syn_b = 0
    
    # AIS (Guo)
    gl_off.g_Na = 249 * msiemens /cm/cm
    gl_off.g_K = 68.85 * msiemens /cm/cm
    gl_off.g_KA = 18.9 * msiemens /cm/cm
    gl_off.g_Ca = 1.6 * msiemens /cm/cm
    gl_off.g_K_Ca_max = 0.0474 * msiemens /cm /cm
    gl_off.g_h = sp.gl_off_cond_h * msiemens/cm/cm 
    gl_off.g_CaT = sp.gl_off_cond_cat * msiemens/cm/cm 
    gl_off.g_L = sp.gl_off_leakage * msiemens/cm/cm

    gl_off.radius = 13 * um

    # initial values
    
    rest_V = -70.5 * mV # reversal potential for the leakage current (Guo)
    init_V = np.random.randn(NUM_GL_OFF)*3*mV+rest_V
    gl_off.V = init_V

    gl_off.m = 0.5
    gl_off.h = 0.5
    gl_off.c = 0.5
    gl_off.n = 0.5
    gl_off.A = 0
    gl_off.h_A = 1
    gl_off.y = 0.5
    gl_off.m_T = 0
    gl_off.h_T = 1
    gl_off.j_T = 1

    gl_off.Ca = 0.0001 * nM

    # get delayed voltage
    if sp.cone_exists:
        cr.run_regularly("V_delayed = get_delayed_cr(i, timestep(t-syn_delay, dt))")
        hrz.run_regularly("V_delayed = get_delayed_hrz(i, timestep(t-syn_delay, dt))")
    bp_on.run_regularly("V_delayed_am = get_delayed_bp_on(i, timestep(t-syn_delay_am, dt))")
    bp_on.run_regularly("V_delayed_gl = get_delayed_bp_on(i, timestep(t-syn_delay_gl, dt))")
    bp_off.run_regularly("V_delayed_am = get_delayed_bp_off(i, timestep(t-syn_delay_am, dt))")
    bp_off.run_regularly("V_delayed_gl = get_delayed_bp_off(i, timestep(t-syn_delay_gl, dt))")
    am_wf_on.run_regularly("V_delayed = get_delayed_am_wf_on(i, timestep(t-syn_delay, dt))")
    am_wf_off.run_regularly("V_delayed = get_delayed_am_wf_off(i, timestep(t-syn_delay, dt))")
    am_nf_on.run_regularly("V_delayed = get_delayed_am_nf_on(i, timestep(t-syn_delay, dt))")

    # light-cone interaction
    if sp.cone_exists:
        cr.run_regularly("G_syn_l = (0 + ({}-0)*(1-lstim(t,i))) * nsiemens".format(light_g_max))

    # electrical stimulation
    if delta_ve_folder:
        if sp.implant_mode == "epiretinal":
            if sp.cone_exists:
                cr.run_regularly("delta_V_e = delta_ve_cr(t,i)*mV")
                hrz.run_regularly("delta_V_e = delta_ve_hrz(t,i)*mV")
            bp_on.run_regularly("delta_V_e = delta_ve_bp_on(t,i)*mV")
            bp_off.run_regularly("delta_V_e = delta_ve_bp_off(t,i)*mV")
            am_wf_on.run_regularly("delta_V_e = delta_ve_am_wf_on(t,i)*mV")
            am_wf_off.run_regularly("delta_V_e = delta_ve_am_wf_off(t,i)*mV")
            am_nf_on.run_regularly("delta_V_e = delta_ve_am_nf_on(t,i)*mV")
            gl_on.run_regularly("delta_V_e = delta_ve_gl_on(t,i)*mV")
            gl_off.run_regularly("delta_V_e = delta_ve_gl_off(t,i)*mV")
        elif sp.implant_mode == "subretinal":
            if sp.cone_exists:
                hrz.run_regularly("delta_V_e = delta_ve_hrz(t,i)*mV")
            bp_on.run_regularly("delta_V_e = delta_ve_bp_on(t,i)*mV")
            bp_off.run_regularly("delta_V_e = delta_ve_bp_off(t,i)*mV")
            am_wf_on.run_regularly("delta_V_e = delta_ve_am_wf_on(t,i)*mV")
            am_wf_off.run_regularly("delta_V_e = delta_ve_am_wf_off(t,i)*mV")
            am_nf_on.run_regularly("delta_V_e = delta_ve_am_nf_on(t,i)*mV")
            gl_on.run_regularly("delta_V_e = delta_ve_gl_on(t,i)*mV")
            gl_off.run_regularly("delta_V_e = delta_ve_gl_off(t,i)*mV")
        elif sp.implant_mode == "cone_only":
            if sp.cone_exists:
                cr.run_regularly("delta_V_e = delta_ve_cr(t,i)*mV")
            else:
                raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        elif sp.implant_mode == "bp_only":
            bp_on.run_regularly("delta_V_e = delta_ve_bp_on(t,i)*mV")
            bp_off.run_regularly("delta_V_e = delta_ve_bp_off(t,i)*mV")
        else:
            raise ValueError("The implant mode {} is not valid.".format(sp.implant_mode))
    
    # if debug mode is on, record more values
    if debug:
        if sp.cone_exists:
            cr_mon = StateMonitor(cr, ["V_value", "I_syn", "I_e"], record=True)
            hrz_mon = StateMonitor(hrz, ["V_value", "I_syn", "I_e"], record=True)
        bp_on_mon = StateMonitor(bp_on, ["V_value", "I_syn", "I_e"], record=True)
        bp_off_mon = StateMonitor(bp_off, ["V_value", "I_syn", "I_e"], record=True)
        am_wf_on_mon = StateMonitor(am_wf_on, ["V_value", "I_syn", "I_e"], record=True)
        am_wf_off_mon = StateMonitor(am_wf_off, ["V_value", "I_syn", "I_e"], record=True)
        am_nf_on_mon = StateMonitor(am_nf_on, ["V_value", "I_syn", "I_e"], record=True)
        gl_on_mon = StateMonitor(gl_on, ["V", "I_syn", "I_syn_a", "I_syn_b", "G_syn_a", "G_syn_b", "I_e"], record=True)
        gl_off_mon = StateMonitor(gl_off, ["V", "I_syn", "G_syn_aw", "G_syn_an", "G_syn_b", "I_e", "m", "n", "h"], record=True)
    else:
        if sp.cone_exists:
            cr_mon = StateMonitor(cr, "V_value", record=True)
            hrz_mon = StateMonitor(hrz, "V_value", record=True)
        bp_on_mon = StateMonitor(bp_on, "V_value", record=True)
        bp_off_mon = StateMonitor(bp_off, "V_value", record=True)
        am_wf_on_mon = StateMonitor(am_wf_on, "V_value", record=True)
        am_wf_off_mon = StateMonitor(am_wf_off, "V_value", record=True)
        am_nf_on_mon = StateMonitor(am_nf_on, "V_value", record=True)
        gl_on_mon = StateMonitor(gl_on, "V", record=True)
        gl_off_mon = StateMonitor(gl_off, "V", record=True)
        
    # spike monitors
    gl_on_spike_mon = SpikeMonitor(gl_on)
    gl_off_spike_mon = SpikeMonitor(gl_off)

    # implementation in CPP to get delayed voltage
    # if there is no data, the resting voltage is returned
    if sp.cone_exists:
        @implementation('cpp', '''
        float get_delayed_cr(int index, int step) {
            step = max(0, step);  // do not go beyond start
            if (brian::ARRAY_NAME.n > 0) { // any data?
                return brian::ARRAY_NAME(step, index);
            } else {
                return -46.8;
            }
        }
        '''.replace('ARRAY_NAME',
                    device.get_array_name(cr_mon.variables['V_value'],
                                          access_data=False)))
        @declare_types(index='integer', step='integer')
        @check_units(index=1, step=1, result=1)
        def get_delayed_cr(index, step):
            raise NotImplementedError('Use C++ target')

        @implementation('cpp', '''
        float get_delayed_hrz(int index, int step) {
            step = max(0, step);  // do not go beyond start
            if (brian::ARRAY_NAME.n > 0) { // any data?
                return brian::ARRAY_NAME(step, index);
            } else {
                return -47.7;
            }
        }
        '''.replace('ARRAY_NAME',
                    device.get_array_name(hrz_mon.variables['V_value'],
                                          access_data=False)))
        @declare_types(index='integer', step='integer')
        @check_units(index=1, step=1, result=1)
        def get_delayed_hrz(index, step):
            raise NotImplementedError('Use C++ target')

    @implementation('cpp', '''
    float get_delayed_bp_on(int index, int step) {
        step = max(0, step);  // do not go beyond start
        if (brian::ARRAY_NAME.n > 0) { // any data?
            return brian::ARRAY_NAME(step, index);
        } else {
            return -35.0;
        }
    }
    '''.replace('ARRAY_NAME',
                device.get_array_name(bp_on_mon.variables['V_value'],
                                      access_data=False)))
    @declare_types(index='integer', step='integer')
    @check_units(index=1, step=1, result=1)
    def get_delayed_bp_on(index, step):
        raise NotImplementedError('Use C++ target')

    @implementation('cpp', '''
    float get_delayed_bp_off(int index, int step) {
        step = max(0, step);  // do not go beyond start
        if (brian::ARRAY_NAME.n > 0) { // any data?
            return brian::ARRAY_NAME(step, index);
        } else {
            return -44.0;
        }
    }
    '''.replace('ARRAY_NAME',
                device.get_array_name(bp_off_mon.variables['V_value'],
                                      access_data=False)))
    @declare_types(index='integer', step='integer')
    @check_units(index=1, step=1, result=1)
    def get_delayed_bp_off(index, step):
        raise NotImplementedError('Use C++ target')

    @implementation('cpp', '''
    float get_delayed_am_wf_on(int index, int step) {
        step = max(0, step);  // do not go beyond start
        if (brian::ARRAY_NAME.n > 0) { // any data?
            return brian::ARRAY_NAME(step, index);
        } else {
            return -42.0;
        }
    }
    '''.replace('ARRAY_NAME',
                device.get_array_name(am_wf_on_mon.variables['V_value'],
                                      access_data=False)))
    @declare_types(index='integer', step='integer')
    @check_units(index=1, step=1, result=1)
    def get_delayed_am_wf_on(index, step):
        raise NotImplementedError('Use C++ target')

    @implementation('cpp', '''
    float get_delayed_am_wf_off(int index, int step) {
        step = max(0, step);  // do not go beyond start
        if (brian::ARRAY_NAME.n > 0) { // any data?
            return brian::ARRAY_NAME(step, index);
        } else {
            return -34.5;
        }
    }
    '''.replace('ARRAY_NAME',
                device.get_array_name(am_wf_off_mon.variables['V_value'],
                                      access_data=False)))
    @declare_types(index='integer', step='integer')
    @check_units(index=1, step=1, result=1)
    def get_delayed_am_wf_off(index, step):
        raise NotImplementedError('Use C++ target')

    @implementation('cpp', '''
    float get_delayed_am_nf_on(int index, int step) {
        step = max(0, step);  // do not go beyond start
        if (brian::ARRAY_NAME.n > 0) { // any data?
            return brian::ARRAY_NAME(step, index);
        } else {
            return -47.6;
        }
    }
    '''.replace('ARRAY_NAME',
                device.get_array_name(am_nf_on_mon.variables['V_value'],
                                      access_data=False)))
    @declare_types(index='integer', step='integer')
    @check_units(index=1, step=1, result=1)
    def get_delayed_am_nf_on(index, step):
        raise NotImplementedError('Use C++ target')

    synapse_params = '''
    D : 1
    W : 1
    g_min : 1
    g_max : 1
    V_50 : 1
    beta : 1
    sigma : 1
    '''

    increasing_synapse_eqs = '''
    G_syn_post = (1/W) * (g_min + (g_max-g_min) * (1-1/(1+exp((V_delayed_pre-(V_50))/(beta))))) * exp(-D/sigma) : 1 (summed)
    '''

    decreasing_synapse_eqs = '''
    G_syn_post = (1/W) * (g_min + (g_max-g_min) * (1/(1+exp((V_delayed_pre-V_50)/beta)))) * exp(-D/sigma) : 1 (summed)
    '''

    if sp.cone_exists:
        cr_hrz = Synapses(cr, hrz, model=synapse_params+increasing_synapse_eqs, method='exponential_euler')
        cr_hrz.connect()
        if pre_computed_weights_folder:
            cr_hrz.D, cr_hrz.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/HRZ.npy".format(xy_coord_folder), 10.5, pre_computed_weights = "{}/cr_hrz.npy".format(pre_computed_weights_folder))
        else:
            cr_hrz.D, cr_hrz.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/HRZ.npy".format(xy_coord_folder), 10.5)
        cr_hrz.g_min = 0.0 
        cr_hrz.g_max = 7.0 
        cr_hrz.V_50 = -43.0
        cr_hrz.beta = 2.0
        cr_hrz.sigma = 10.5

        hrz_cr = Synapses(hrz, cr, model=synapse_params+increasing_synapse_eqs, method='exponential_euler')
        hrz_cr.connect()
        hrz_cr.D, hrz_cr.W = get_distances("{}/HRZ.npy".format(xy_coord_folder), "{}/CR.npy".format(xy_coord_folder), 2.5)
        hrz_cr.g_min = 0.0
        hrz_cr.g_max = 3.0
        hrz_cr.V_50 = -29.5
        hrz_cr.beta = 7.4
        hrz_cr.sigma = 2.5

        cr_bp_on = Synapses(cr, bp_on, model=synapse_params+decreasing_synapse_eqs, method='exponential_euler')
        cr_bp_on.connect()
        if pre_computed_weights_folder:
            cr_bp_on.D, cr_bp_on.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/BP_ON.npy".format(xy_coord_folder), 3.85, pre_computed_weights = "{}/cr_bp_on.npy".format(pre_computed_weights_folder))
        else:
            cr_bp_on.D, cr_bp_on.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/BP_ON.npy".format(xy_coord_folder), 3.85)
        cr_bp_on.g_min = 0.1 
        cr_bp_on.g_max = 1.1 
        cr_bp_on.V_50 = -47
        cr_bp_on.beta = 1.7
        cr_bp_on.sigma = 3.85

        cr_bp_off = Synapses(cr, bp_off, model=synapse_params+increasing_synapse_eqs, method='exponential_euler')
        cr_bp_off.connect()
        if pre_computed_weights_folder:
            cr_bp_off.D, cr_bp_off.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/BP_OFF.npy".format(xy_coord_folder), 3.85, pre_computed_weights = "{}/cr_bp_on.npy".format(pre_computed_weights_folder))
        else:
            cr_bp_off.D, cr_bp_off.W = get_distances("{}/CR.npy".format(xy_coord_folder), "{}/BP_OFF.npy".format(xy_coord_folder), 3.85)
        cr_bp_off.g_min = 0.0 
        cr_bp_off.g_max = 3.75 
        cr_bp_off.V_50 = -41.5
        cr_bp_off.beta = 1.2
        cr_bp_off.sigma = 3.85

    bp_on_am_wf_on = Synapses(bp_on, am_wf_on, model=synapse_params+increasing_synapse_eqs.replace("V_delayed_pre", "V_delayed_am_pre"), method='exponential_euler')
    bp_on_am_wf_on.connect()
    if pre_computed_weights_folder:
        bp_on_am_wf_on.D, bp_on_am_wf_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/AM_WF_ON.npy".format(xy_coord_folder), 24, pre_computed_weights="{}/bp_on_am_wf_on.npy".format(pre_computed_weights_folder))
    else:
        bp_on_am_wf_on.D, bp_on_am_wf_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/AM_WF_ON.npy".format(xy_coord_folder), 24)
    bp_on_am_wf_on.g_min = 0.0 
    bp_on_am_wf_on.g_max = 1.0
    bp_on_am_wf_on.V_50 = -33.5
    bp_on_am_wf_on.beta = 3.0
    bp_on_am_wf_on.sigma = 24

    bp_off_am_wf_off = Synapses(bp_off, am_wf_off, model=synapse_params+increasing_synapse_eqs.replace("V_delayed_pre", "V_delayed_am_pre"), method='exponential_euler')
    bp_off_am_wf_off.connect()
    if pre_computed_weights_folder:
        bp_off_am_wf_off.D, bp_off_am_wf_off.W = get_distances("{}/BP_OFF.npy".format(xy_coord_folder), "{}/AM_WF_OFF.npy".format(xy_coord_folder), 24, pre_computed_weights="{}/bp_off_am_wf_off.npy".format(pre_computed_weights_folder))
    else:
        bp_off_am_wf_off.D, bp_off_am_wf_off.W = get_distances("{}/BP_OFF.npy".format(xy_coord_folder), "{}/AM_WF_OFF.npy".format(xy_coord_folder), 24)
    bp_off_am_wf_off.g_min = 0.0 
    bp_off_am_wf_off.g_max = 1.8 
    bp_off_am_wf_off.V_50 = -44.0
    bp_off_am_wf_off.beta = 3.0
    bp_off_am_wf_off.sigma = 24

    bp_on_am_nf_on = Synapses(bp_on, am_nf_on, model=synapse_params+increasing_synapse_eqs.replace("V_delayed_pre", "V_delayed_am_pre"), method='exponential_euler')
    bp_on_am_nf_on.connect()
    if pre_computed_weights_folder:
        bp_on_am_nf_on.D, bp_on_am_nf_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/AM_NF_ON.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/bp_on_am_nf_on.npy".format(pre_computed_weights_folder))
    else:
        bp_on_am_nf_on.D, bp_on_am_nf_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/AM_NF_ON.npy".format(xy_coord_folder), 6)
    bp_on_am_nf_on.g_min = 0.0 
    bp_on_am_nf_on.g_max = 0.2 
    bp_on_am_nf_on.V_50 = -35.0
    bp_on_am_nf_on.beta = 3.0
    bp_on_am_nf_on.sigma = 6

    bp_on_gl_on = Synapses(bp_on, gl_on, model=synapse_params+increasing_synapse_eqs.replace("G_syn_post", "G_syn_b_post").replace("V_delayed_pre", "V_delayed_gl_pre"), method='exponential_euler')
    bp_on_gl_on.connect()
    if pre_computed_weights_folder:
        bp_on_gl_on.D, bp_on_gl_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/GL_ON.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/bp_on_gl_on.npy".format(pre_computed_weights_folder))
    else:
        bp_on_gl_on.D, bp_on_gl_on.W = get_distances("{}/BP_ON.npy".format(xy_coord_folder), "{}/GL_ON.npy".format(xy_coord_folder), sp.bp_on_gl_on_sigma)
    bp_on_gl_on.g_min = 0.0 
    bp_on_gl_on.g_max = sp.bp_on_gl_on_gmax 
    bp_on_gl_on.V_50 = -33.5
    bp_on_gl_on.beta = 3.0
    bp_on_gl_on.sigma = sp.bp_on_gl_on_sigma

    am_wf_on_gl_on = Synapses(am_wf_on, gl_on, model=synapse_params+increasing_synapse_eqs.replace("G_syn_post", "G_syn_a_post"), method='exponential_euler')
    am_wf_on_gl_on.connect()
    if pre_computed_weights_folder:
        am_wf_on_gl_on.D, am_wf_on_gl_on.W = get_distances("{}/AM_WF_ON.npy".format(xy_coord_folder), "{}/GL_ON.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/am_wf_on_gl_on.npy".format(pre_computed_weights_folder))
    else:
        am_wf_on_gl_on.D, am_wf_on_gl_on.W = get_distances("{}/AM_WF_ON.npy".format(xy_coord_folder), "{}/GL_ON.npy".format(xy_coord_folder), 6)
    am_wf_on_gl_on.g_min = 0.0 
    am_wf_on_gl_on.g_max = sp.am_wf_on_gl_on_gmax 
    am_wf_on_gl_on.V_50 = -42.5
    am_wf_on_gl_on.beta = 2.5
    am_wf_on_gl_on.sigma = 6

    bp_off_gl_off = Synapses(bp_off, gl_off, model=synapse_params+increasing_synapse_eqs.replace("G_syn_post", "G_syn_b_post").replace("V_delayed_pre", "V_delayed_gl_pre"), method='exponential_euler')
    bp_off_gl_off.connect()
    if pre_computed_weights_folder:
        bp_off_gl_off.D, bp_off_gl_off.W = get_distances("{}/BP_OFF.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/bp_off_gl_off.npy".format(pre_computed_weights_folder))
    else:
        bp_off_gl_off.D, bp_off_gl_off.W = get_distances("{}/BP_OFF.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), sp.bp_off_gl_off_sigma) 
    bp_off_gl_off.g_min = 0.0 
    bp_off_gl_off.g_max = sp.bp_off_gl_off_gmax 
    bp_off_gl_off.V_50 = -44.0
    bp_off_gl_off.beta = 3.0
    bp_off_gl_off.sigma = sp.bp_off_gl_off_sigma

    am_wf_off_gl_off = Synapses(am_wf_off, gl_off, model=synapse_params+increasing_synapse_eqs.replace("G_syn_post", "G_syn_aw_post"), method='exponential_euler')
    am_wf_off_gl_off.connect()
    if pre_computed_weights_folder:
        am_wf_off_gl_off.D, am_wf_off_gl_off.W = get_distances("{}/AM_WF_OFF.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/am_wf_off_gl_off.npy".format(pre_computed_weights_folder))
    else:
        am_wf_off_gl_off.D, am_wf_off_gl_off.W = get_distances("{}/AM_WF_OFF.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), 6)
    am_wf_off_gl_off.g_min = 0.0 
    am_wf_off_gl_off.g_max = sp.am_wf_off_gl_off_gmax
    am_wf_off_gl_off.V_50 = -34.4
    am_wf_off_gl_off.beta = 2.5
    am_wf_off_gl_off.sigma = 6

    am_nf_on_gl_off = Synapses(am_nf_on, gl_off, model=synapse_params+increasing_synapse_eqs.replace("G_syn_post", "G_syn_an_post"), method='exponential_euler')
    am_nf_on_gl_off.connect()
    if pre_computed_weights_folder:
        am_nf_on_gl_off.D, am_nf_on_gl_off.W = get_distances("{}/AM_NF_ON.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), 6, pre_computed_weights="{}/am_nf_on_gl_off.npy".format(pre_computed_weights_folder))
    else:
        am_nf_on_gl_off.D, am_nf_on_gl_off.W = get_distances("{}/AM_NF_ON.npy".format(xy_coord_folder), "{}/GL_OFF.npy".format(xy_coord_folder), 6)
    am_nf_on_gl_off.g_min = 0.0 
    am_nf_on_gl_off.g_max = sp.am_nf_on_gl_off_gmax 
    am_nf_on_gl_off.V_50 = -47.5
    am_nf_on_gl_off.beta = 2.0
    am_nf_on_gl_off.sigma = 6
    
    run(time_in_ms*ms)

    if sp.cone_exists:
        data_cr = cr_mon.get_states()
        data_hrz = hrz_mon.get_states()
    data_bp_on = bp_on_mon.get_states()
    data_bp_off = bp_off_mon.get_states()
    data_am_wf_off = am_wf_off_mon.get_states()
    data_am_wf_on = am_wf_on_mon.get_states()
    data_am_nf_on = am_nf_on_mon.get_states()
    data_gl_off = gl_off_mon.get_states()
    data_gl_off['V_value'] = data_gl_off['V']/mV
    data_gl_on = gl_on_mon.get_states()
    data_gl_on['V_value'] = data_gl_on['V']/mV

    spikes_gl_on = gl_on_spike_mon.all_values()    
    spikes_gl_off = gl_off_spike_mon.all_values()

    device.reinit()
    device.activate()
    
    if genn_directory:
        shutil.rmtree(genn_directory)
    else:
        shutil.rmtree("GeNNworkspace")

    if sp.cone_exists:
        return [data_cr, data_hrz, data_bp_on, data_bp_off, data_am_wf_on, data_am_wf_off, data_am_nf_on, data_gl_on, data_gl_off, spikes_gl_on, spikes_gl_off]
    else:
        return [None, None, data_bp_on, data_bp_off, data_am_wf_on, data_am_wf_off, data_am_nf_on, data_gl_on, data_gl_off, spikes_gl_on, spikes_gl_off]
