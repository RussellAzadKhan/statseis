"""
Miscellaneous functions
"""


def clustering_event_rate(k, t, c, p, r, q):
    """
    Shearer 2012. clustering behaviour expected for separable time and distance power laws,
        i.e., an equation where the event rate, D, is given by the equation where κ is a scaling constant, 
        t is time, r is distance, c and p are the Omori's law parameters, and q is the distance decay parameter. 
        This equation predicts uniform time decay regardless of distance and uniform distance decay regardless of time.
    """
    D = k*(t + c)**(-p)*r**(-q)
    return D
    
def productivity_law(m, m1, Q, alpha=1):
    """
    Shearer 2012. average number of direct (first generation) aftershocks,
    Nasl following an event of magnitude m follows a productivity law
    where m1 is the minimum magnitude earthquake that triggers other earthquakes,
    Q is an aftershock productivity parameter (denoted k by Sornette and Werner [2005b]),
    and alpha is a parameter that determines the rate of increase in the number of aftershocks
    observed for larger main shock magnitudes. Provides the average number
    of direct aftershocks for a given magnitude main shock and typically will have non-integer values. 
    """
    N_asl = Q*10**(alpha*(m-m1))
    return N_asl 

def GR_law(a, b, m):
    """
    Gutenberg-Richter (G-R) distribution.
    “where m is the magnitude, a is related to the total number of earthquakes,
     and b (the b-value) determines the relative number of large quakes compared 
     to small quakes and is generally observed to lie between 0.8 and 1.2." (Shearer 2012). 
    """
    N = 10**(a-b*m)
    return N

def rand_event_mag(m1=0, m2=5.5):
    """
    Shearer 2012. How event magnitudes are generated in computer simulations.
    """
    x_r = np.random.uniform(10**(m1-m2), 1)
    m_r = m1 - np.log10(x_r)
    return m_r

# def branching_ratio(Q, b, m2, m1):
#     """
#     Shearer 2012. Triggering productivity can be defined in terms of the branching ratio, n, given by equation,
#     which gives the ratio of the average number of first generation aftershocks to the number of background events 
#     (simplified in this case by using an approximation for large m2). This parameter is used by 
#     Helmstetter and Sornette [2003b], Helmstetter et al. [2003], and Sornette and Werner [2005a, 2005b]. 
#     For the G-R magnitude limitsm1 = 0 and m2 = 5.5, following the method described in Shearer [2012]
#     we obtain n = 0.39 in order to satisfy Båth's law [Båth, 1965], the observation that the largest aftershock is, 
#     on average, 1.2 magnitudes smaller than its main shock.
#     """
#     n = Q*b*np.log(10)*(m2-m1)
#     return n
    
def aftershock_rate(t, c=0.001, p=1):
    """
    Shearer 2012. The time delay following the triggering event can be determined from Omori's Law,
    where D is the aftershock rate, t is the time delay, c is a constant the defines the flattening 
    of the aftershock decay curve at short times, and p is the decay constant, which is often assumed to be one.
    For the simulations presented here,c = 0.001 day (86 s) and p = 1.
    """
    D = (t + c)**-p
    return D

def aftershock_distance_decay(r, q):
    """
    Shearer 2012. Following Felzer and Brodsky [2006], I assume that the distance, r, 
    from the triggering event to the aftershock obeys the power law where D is the aftershock rate 
    and q is the distance decay constant. After a random r is drawn from this distribution, 
    the aftershock location is assigned as a random location on a sphere of radius r centered on the triggering event, 
    excluding any portions of the sphere that are above the surface or below an assigned maximum depth of seismicity 
    (30 km for the simulations presented here).
    """
    D = r**-q
    return D

### Heimisson 2019

def K_t(tau_t, A, sigma_t, tau_0, sigma_0, alpha):
    """
    (Heimisson, 2019) Stress history-dependent integral kernel function in terms of shear and normal stress.
    """
    K_t = np.exp((tau_t/A*sigma_t) - (tau_0/A*sigma_0)*(sigma_t/sigma_0)**(alpha/A))


def K_t_approximation(S_t, A, sigma_0, tau_t, tau_0, sigma_t, alpha):
    """
    (Heimisson, 2019) Stress history-dependent integral kernel function in terms of shear and normal stress
    Coulomb stress approximation.
    """
    delta_tau_t = tau_t - tau_0
    delta_sigma_t = sigma_t - sigma_0 # my assumption, not explicitly stated
    S_t = delta_tau_t - (tau_0/sigma_0 - alpha)*delta_sigma_t
    K_t = np.exp(S_t/(A*sigma_0))

def cumulative_event_count(A, sigma_0, Tau_r, K):
    """
    (Heimisson, 2019) Intergral expression for the cumulative number of events  N.
    A is a constitutive parameter that relates to the rate dependence of friction, 
    and σ0 is the initial normal stress acting on a population. K(t) is a stress 
    history-dependent integral kernel function that can be written out explicitly 
    in terms of shear τ(t) and normal stress σ(t)
    """
    t_alpha = A*sigma_0/Tau_r
    N_r = t_alpha*np.log((1/t_alpha)*K+1)
    return N_r

def seismicity_rate_intergral(A, sigma_0, Tau_r, K):
    """
    (Heimisson, 2019) Integral expression for the seismicity rate R.
    A is a constitutive parameter that relates to the rate dependence of friction, 
    and σ0 is the initial normal stress acting on a population. K(t) is a stress 
    history-dependent integral kernel function that can be written out explicitly 
    in terms of shear τ(t) and normal stress σ(t)
    """
    t_alpha = A*sigma_0/Tau_r
    R_r = K/(1+(1/t_alpha)*K) 

