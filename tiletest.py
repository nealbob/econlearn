"""Examples"""

# Authors:  Neal Hughes <neal.hughes@anu.edu.au>

def consumption_savings(T, alpha, sigma, rho, simple=0):
    
    import numpy as np

    eta = np.random.normal(size=T, loc=0, scale=sigma)
    epsilon = np.random.rand(T)
    if simple > 0:
        epsilon = np.ones(T)*simple

    delta = 1

    Y = np.zeros(T)
    C = np.zeros(T)
    U = np.zeros(T)
    K = np.zeros(T+1)
    Z = np.zeros(T+1)

    Z[0] = 1
    K[0] = 1

    for t in range(T):
        
        # Production
        Y[t] = Z[t] * (K[t]**alpha)
        # Consumption
        C[t] = epsilon[t] * Y[t] 

        # State transition
        K[t+1] = Y[t] - C[t]
        
        # Need to check functional form for Z
        Z[t+1] = np.exp(rho * np.log(Z[t]) + eta[t])
   

    # Payoff
    U = np.log(C)
    
    # Stack the samples 
    XA = np.vstack([C, K[0:T], Z[0:T]]).T
    X1 = np.vstack([K[1::], Z[1::]]).T

    print 'Mean welfare: ' + str(np.mean(U))

    return [XA, X1, U, Y]


