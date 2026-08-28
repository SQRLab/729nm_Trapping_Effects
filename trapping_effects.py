import numpy as np


### CONSTANTS ###
c = 2.99792458*10**8 # speed of light (m/s)
epsilon_0 = 1/(4*np.pi*10**-7*c**2) # permittivity of free space (C^2/N/m^2)
alpha_0 = 7.2973525376*10**(-3) # fine structure constant
e = 1.602176487*10**(-19) # electron charge (C)
h_bar = 6.62606896*10**(-34)/(2*np.pi) # reduced planck constant (J*s)
k_b = 1.3806504*10**(-23)


### Clebsch Gordan Coefficient Lookup Tables ###
cg_05 = {-1.5 : np.sqrt(1/5), -0.5 : np.sqrt(2/5), 0.5: np.sqrt(3/5), 1.5 : np.sqrt(4/5), 2.5 : 1} # m=1/2 lookup table


### HELPER FUNCTIONS ###
def Clebsch_Gordan(m, m_prime):
    # lookup the appropriate Clebsch Gordan coefficient
    if m == 0.5:
        # For m=1/2, use the m=1/2 lookup table
        return cg_05.get(m_prime)
    
    elif m == -0.5:
        # For m=-1/2, use the m=1/2 lookup table with m'->-m'
        return cg_05.get(-m_prime)

    else: 
        raise ValueError("This m value has not been implemented")

def Geometric_Factors(delta_m, phi, gamma=None, p=None):
    # compute the geometrical factor for a given transition
    # phi: angle between magnetic field and wave propogation
    # gamma: angle between magnetic field and polarization when projected onto the plane of incidence 
    #        if gamma is not specified, assumes circular polarization
    # p: ±1 for either sigma+ or sigma- polarized light
    #    only specify when circularly polarized
    # from the kirchmair thesis
    if gamma is None:
        if np.abs(delta_m) == 0:
            return 1/np.sqrt(8)*np.abs(np.sin(2*phi))
        elif np.abs(delta_m) == 1:
            return 1/np.sqrt(12)*np.abs(np.cos(phi)-delta_m*p*np.cos(2*phi))
        elif np.abs(delta_m) == 2:
            return 1/np.sqrt(12)*np.abs(0.5*np.sin(2*phi)+np.sin(phi))
    else:
        if np.abs(delta_m) == 0:
            return 0.5*np.abs(np.cos(gamma)*np.sin(2*phi))
        elif np.abs(delta_m) == 1:
            return 1/np.sqrt(6)*np.abs(np.cos(gamma)*np.cos(2*phi)+1j*np.sin(gamma)*np.cos(phi))
        elif np.abs(delta_m) == 2:
            return 1/np.sqrt(6)*np.abs(0.5*np.cos(gamma)*np.sin(2*phi)+1j*np.sin(gamma)*np.sin(phi))

def P2I(power, waist_size):
    # converts from power (measurable) to intensity
    return 2*power/(np.pi*waist_size**2)

def rabi_off_axis(r, beam_waist, rabi_initial):
    return rabi_initial*np.exp(-r**2/beam_waist**2)


### FUNCTIONS ###
def rabi_frequency(E_0, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma=None, p=None):
    # calculate the rabi frequency for a given set of parameters
    delta_m = np.abs(m_prime - m)
    cg_coeff = Clebsch_Gordan(m, m_prime)
    geo_term = Geometric_Factors(delta_m, phi, gamma, p)
    return e*E_0/(2*h_bar)*np.sqrt(15*excited_linewidth/(c*alpha_0*wave_vec_mag**3))*cg_coeff*geo_term

def rabi_frequency_power(power, waist_size, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma=None, p=None):
    # calculate the rabi frequency for a given set of parameters excluding E_0
    E_0 = np.sqrt(2*P2I(power, waist_size)/(c*epsilon_0))
    return rabi_frequency(E_0, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma, p)

def find_power(target_rabi, waist_size, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma=None, p=None):
    # find the power necessary to achieve a target rabi frequency given a specific waist size
    delta_m = np.abs(m_prime - m)
    cg_coeff = Clebsch_Gordan(m, m_prime)
    geo_term = Geometric_Factors(delta_m, phi, gamma, p)
    return c*epsilon_0*np.pi/4*(target_rabi*waist_size/(e/(2*h_bar)*np.sqrt(15/(c*alpha_0*wave_vec_mag**3)*excited_linewidth)*cg_coeff*geo_term))**2

def find_intensity(target_rabi, waist_size, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma=None, p=None):
    # find the intensity necessary to achieve a target rabi frequency given a specific waist size
    return P2I(find_power(target_rabi, waist_size, excited_linewidth, wave_vec_mag, m, m_prime, phi, gamma, p), waist_size)

def first_order_dev(r, dr, beam_waist, rabi_initial):
    # finds the first order standard deviation of rabi frequency given some pointing instability dr and distance off axis r
    return 2*r*dr/beam_waist**2 * rabi_off_axis(r, beam_waist, rabi_initial)

def second_order_dev(dr, beam_waist, rabi_initial):
    # finds the second order standard deviation of rabi frequency assuming an on-axis pointing instability dr
    return np.sqrt(2)*rabi_initial*dr**2/beam_waist**2

def single_qubit_gate_fidelity(dOmega, t):
    # find the single qubit gate fidelity for a pulse of time t and rabi frequency deviation of dOmega
    return 1- (dOmega * t)**2/4