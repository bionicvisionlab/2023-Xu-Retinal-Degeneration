from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
from scipy.spatial import distance_matrix
import brian2genn
import shutil

# initial membrane voltages are the resting voltage when light intensity is 0.5
def retina_simulation(time_in_ms, sp, light_g_max=0.9, pre_computed_weights_folder=None, delta_ve_folder=None, delta_ve_dt=0.01, simulation_timestep=None, debug=False, gpu=True, select_GPU=None, genn_directory=None):
    
    xy_coord_folder = sp.xy_coord_folder
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
            delta_ve_gl_on = TimedArray(np.load("{}/GL_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_off = TimedArray(np.load("{}/GL_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
        elif sp.implant_mode == "subretinal":
            delta_ve_gl_on = TimedArray(np.load("{}/GL_ON.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
            delta_ve_gl_off = TimedArray(np.load("{}/GL_OFF.npy".format(delta_ve_folder)), dt=delta_ve_dt*ms)
        elif sp.implant_mode == "cone_only":
            raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        elif sp.implant_mode == "bp_only":
            raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        else:
            raise ValueError("The implant mode {} is not valid.".format(sp.implant_mode))

    # ganglion on
    gl_on_eqs = '''
    dV/dt = (1/C_m) * (I_e - I_m) : volt
    # extracelllar current
    I_e = G_m * delta_V_e / 2 : amp
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

    gl_on.delta_V_e = 0 * mV
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
    dV/dt = (1/C_m) * (I_e - I_m) : volt
    # extracelllar current
    I_e = G_m * delta_V_e / 2 : amp
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
    factor_w : 1 (constant)
    factor_n : 1 (constant)
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
    
    gl_off.delta_V_e = 0 * mV
    gl_off.factor_w = sp.am_wf_off_gl_off_factor
    gl_off.factor_n = sp.am_nf_on_gl_off_factor
    
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

    # electrical stimulation
    if delta_ve_folder:
        if sp.implant_mode == "epiretinal":
            gl_on.run_regularly("delta_V_e = delta_ve_gl_on(t,i)*mV")
            gl_off.run_regularly("delta_V_e = delta_ve_gl_off(t,i)*mV")
        elif sp.implant_mode == "subretinal":
            gl_on.run_regularly("delta_V_e = delta_ve_gl_on(t,i)*mV")
            gl_off.run_regularly("delta_V_e = delta_ve_gl_off(t,i)*mV")
        elif sp.implant_mode == "cone_only":
            raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        elif sp.implant_mode == "bp_only":
            raise ValueError("The implant mode {} conflicts the retina setup.".format(sp.implant_mode))
        else:
            raise ValueError("The implant mode {} is not valid.".format(sp.implant_mode))
    
    # if debug mode is on, record more values
    if debug:
        gl_on_mon = StateMonitor(gl_on, ["V", "I_syn", "I_syn_a", "I_syn_b", "G_syn_a", "G_syn_b", "I_e"], record=True)
        gl_off_mon = StateMonitor(gl_off, ["V", "I_syn", "G_syn_aw", "G_syn_an", "G_syn_b", "I_e", "m", "n", "h"], record=True)
    else:
        gl_on_mon = StateMonitor(gl_on, "V", record=True)
        gl_off_mon = StateMonitor(gl_off, "V", record=True)
        
    # spike monitors
    gl_on_spike_mon = SpikeMonitor(gl_on)
    gl_off_spike_mon = SpikeMonitor(gl_off)
    
    run(time_in_ms*ms)

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

    return [None, None, None, None, None, None, None, data_gl_on, data_gl_off, spikes_gl_on, spikes_gl_off]
