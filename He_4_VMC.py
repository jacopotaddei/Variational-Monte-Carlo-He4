import numpy as np
from time import process_time



'''Il codice presenta ancora il problema della dipendenza del valore di energia 
del ground state dal volume iniziale (L=3 in questo caso) nel quale sono generate
le particelle. Questo credo sia ereditato dalla struttura della funzione d'onda,
infatti phi in questo problema è una funzione dei tre parametri variazionali
e della sola distanza internucleonica. Dato che si calcola l'energia locale allora
quando il volume è molto grande tutti gli step iniziali del metropolis vanno a campionare
regioni in cui la funzione d'onda è molto piccola e questo è dovuto alla soppressione
esponenziale che ha come argomento proprio la distanza fra le particelle. Dato che 
l'energia cinetica per l'energia locale è della forma kin/wf allora i contributi
in queste regioni crollano drasticamente sotto zero.
Un possibile modo per evitare questo problema potrebbe essere proprio quello di 
attivare il loop del metropolis algoritm che "sposta" le particelle senza campionare
l'energia nei primi N_nel punti. In questo modo la dipendenza dal volume per L grandi 
viene risolta in quanto si troncano valori di energia che restituiscono una sottostima 
del ground state.'''




#Generate random coordinates for the particles
def coordinate_generator():
    #Dimension of the volume
    L=3
    Matrix=np.random.uniform(-L,L, size=(N_particles,3))
    return Matrix
#Creation of the wave function as a function of the internucleons distances
def wave_function(R, Alfa, Beta, Gamma):
    phi=1
    for i in range(N_particles-1):
        for j in range(i+1, N_particles):
            r2=np.sum((R[i]-R[j])**2)
            phi*=np.exp(-Gamma*r2)*(1+Alfa*np.exp(-Beta*r2))
    return phi
#Laplacian calculator
def laplacian(R):
    # "Infinitesimal" increment to compute the second derivatives
    h = 0.001
    laplacian = np.zeros(N_particles)
    #Finite difference based loop to compute the laplacian
    for i in range(N_particles):
        #reset the laplacian for each particle
        lap = 0
        for j in range(3):
            
            # Shifted coordintates by +h
            R_plus = np.copy(R)
            R_plus[i,j] += h
            phi_plus = wave_function(R_plus, Alfa, Beta, Gamma)
            # Shifted coordinates by -h
            R_minus = np.copy(R)
            R_minus[i,j] -= h
            phi_minus = wave_function(R_minus, Alfa, Beta, Gamma)
            # Finite difference second derivatives
            lap += (phi_plus + phi_minus - 2*wave_function(R, Alfa, Beta, Gamma)) / (h**2)
        #Laplacian for each particle
        laplacian[i] = lap
    return laplacian
#Local Energy computator
def local_energy(R):
    pot=0
    # h_bar^2/2m in femptometers and MeV
    h2div2m=20.74
    #Internucleons potential
    for i in range(N_particles-1):
        for j in range(i+1,N_particles):
            r2=np.sum((R[i]-R[j])**2)
            pot+=1000*np.exp(-3*r2)-163.35*np.exp(-1.05*r2)-21.5*np.exp(-0.6*r2)-83*np.exp(-0.8*r2)-11.5*np.exp(-0.4*r2)
    #Kinetic Energy
    Laplacian=np.sum(laplacian(R))
    kin=-h2div2m*Laplacian
    #Local Energy
    E_local=kin/wave_function(R, Alfa, Beta, Gamma)+pot
    return E_local

#Metropolis Algoritm
def metropolis(R, N_met, N_nel=0):
    #Inizializing the counters
    e, e2, rej=0,0,0
    #METROPOLIS STEP
    step=0.4
    
    #Test for the "Anti-Thermalizzation" loop that should avoid the dependance from the starting volume, "nel" is for "non evaluated loops"
    for i in range(N_nel):
        #Random small shift for each coordinate
        deltaR=step*np.random.uniform(-1,1, size=(N_particles, 3))
        #Effective shift of particles
        R_try=R+deltaR
        #Probabilities: P=prob. of the old wf, P_try=prob. on the new move
        P=(wave_function(R, Alfa, Beta, Gamma))**2
        P_try=(wave_function(R_try, Alfa, Beta, Gamma))**2
        #Stochastic acceptance parameter
        x_mute=np.random.uniform(0,1)
        if (P_try/P)>x_mute:
            #Move accepted, R_try becomes the new matrix of coordinates R
            R=R_try
        else: pass
    
    #Metropolis Algoritm
    for i in range(N_met):
        #Random small shift for each coordinate
        deltaR=step*np.random.uniform(-1,1, size=(N_particles, 3))
        #Effective shift of particles
        R_try=R+deltaR
        #Probabilities: P=prob. of the old wf, P_try=prob. on the new move
        P=(wave_function(R, Alfa, Beta, Gamma))**2
        P_try=(wave_function(R_try, Alfa, Beta, Gamma))**2
        #Stochastic acceptance parameter
        x_mute=np.random.uniform(0,1)
        if (P_try/P)>x_mute:
            #Move accepted, R_try becomes the new matrix of coordinates R
            R=R_try
        else: 
            #Move rejected, the particles remain in R and I count how many times this happens
            rej+=1
        #Computation of the local energy and of the local energy squared
        e+=local_energy(R)
        e2+=(local_energy(R))**2
    #rejection ratio to check if the sampling is correct
    R_Ratio=100*rej/N_met 
    #Averages for the metropolis
    E=e/N_met
    E2=e2/N_met
    #variance
    Var_E=(E2-E**2)/N_met
    return E, Var_E, R_Ratio
#Function that starts from the same coordinates R, N_walk random walkers 
def let_them_walk(R,N_walk):
    E, Var, rej = 0, 0, 0
    for i in range(N_walk):
        E_m, Var_m, rej_m=metropolis(R, N_met, N_nel)
        #Averages over walkers
        E+=E_m/N_walk
        Var+=Var_m
        rej+=rej_m/N_walk
    #Propagation of the error from each set. This correspond to sqrt(sum(sigma^2)/N^2)
    dE=np.sqrt(Var)/N_walk
    return E, dE, rej

'''______________________________________________________________________'''
start = process_time()
#Number of particles, in that case 4 since we are considering the four nucleons
N_particles=4
#Number of loops for metropolis algoritm, in other words, number of step of the random walk
N_met, N_nel = 1000, 0
#Number of random walkers
N_walk=10


#Alfa, Beta, Gamma  = -0.7191, 2.0520, 0.08597 
Alfa, Beta, Gamma =-0.7191, 2.0520, 0.08597

# Matrix of the coordinates
R=coordinate_generator()
#Starting of the random walks

Energy, dEnergy, RejRatio= let_them_walk(R, N_walk)
print('The energy is:(%.2f'%Energy, ' +- %.2f' %dEnergy,')', 'with rejection ratio of the Metropolis algoritm of: %.1f' %RejRatio,'%')

end = process_time()
print('Tempo=', end-start)