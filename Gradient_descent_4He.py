import numpy as np
from time import process_time

'''Questo invece è uno dei tentativi di implementazione dell'algoritmo di discesa
del gradiente. Il problema principale sta proprio nel calcolo del gradiente stesso.
I precedenti tentativi sono stati:
    -Calcolo della variazione dell'energia dovuta allo shift dei parametri 
     dalla funzione let_them_walk, ovvero dalla media delle random walk del
     Metropolis algoritm. Chiaramente fallimentare in quanto essendo un gradiente
     calcolato su delle quantità mediate non è fedele all'effettivo shift dell'
     energia.
     Quindi:
     E_alfa=let_them_walk(R,N_walk, Alfa+h, Beta, Gamma)
     E_beta=let_them_walk(R,N_walk, Alfa, Beta+h, Gamma)
     ecc.
    - Calcolo utilizzando lo stesso ragionamento direttamente con l'energia di una 
    singola random walk. Quindi:
    E_alfa=Metropolis(R,N_met, Alfa+h, Beta, Gamma)
    ecc.
    Fallimentare per le stesse ragioni
    -Calcolo considerando la variazione nella local_energy direttamente, tuttavia
    a causa del fatto che in regioni dove la funzione d'onda ha valori piccoli il
    contributo di energia cinetica per l'energia locale crolla a valori molto negativi
    a causa della definizione stessa di energia locale (E_kin/wf) e quindi l'algoritmo
    di discesa porta i parametri proprio in queste zone. Allontanando i parametri dal
    loro valore ottimizzato piuttosto che avvicinarli.
Il codice presentato sotto invece calcola questa variazione su ogni step della random walk.
Quindi partendo dalle 4 particelle iniziali, direttamente all'interno dell'algoritmo
Metropolis viene calcolata l'energia della configurazione e le tre energie relative
allo shift infinitesimo dei parametri.
I problemi di questo algoritmo sono molteplici:
    -Il minimo dei parametri non viene trovato dall'algoritmo in quanto la funzione 
    d'onda risulta molto sensibile anche a variazioni infinitesime dei parametri.
    Il contributo della variazione del parametro Gamma (che moltiplica l'intera funzione
    d'onda di correlazione) è molto grande. Questo significa che il lerning ratio per 
    questo parametro è molto piccolo e comunque l'algoritmo restituisce risultati contrastanti.
    Questo si può testare fissando i tre parametri senza variarli ad ogni ciclo, e si può
    notare come le componenti del gradiente cambino di segno. Questo significa che talvolta
    l'algoritmo sugggerisce di aumentare il valore di un determinato parametro e poi sempre
    sullo stesso valore di diminuirlo.
    Un altro problema è la variazione sul parametro beta (e anche in parte su quello gamma, evitato
    in questo caso dal lering ratio eccessivamente piccolo), infatti essendo un fattore moltiplicativo
    all'interno di un esponenziale, quando l'algoritmo suggerisce di diminuire il valore succede che
    beta assuma valori negativi, questo cambia il segno dell'esponenziale che diventa un contributo
    divergente alla funzione d'onda.
    Questo rende fallimentare anche questo approccio.
    '''
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
def laplacian(R, Alfa, Beta, Gamma):
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
def local_energy(R, Alfa, Beta, Gamma):
    pot=0
    # h_bar^2/2m in femptometers and MeV
    h2div2m=20.74
    #Internucleons potential
    for i in range(N_particles-1):
        for j in range(i+1,N_particles):
            r2=np.sum((R[i]-R[j])**2)
            pot+=1000*np.exp(-3*r2)-163.35*np.exp(-1.05*r2)-21.5*np.exp(-0.6*r2)-83*np.exp(-0.8*r2)-11.5*np.exp(-0.4*r2)
    #Kinetic Energy
    Laplacian=np.sum(laplacian(R, Alfa, Beta, Gamma))
    kin=-h2div2m*Laplacian
    #Local Energy
    E_local=kin/wave_function(R, Alfa, Beta, Gamma)+pot
    return E_local

#Metropolis Algoritm
def metropolis(R, N_met, N_nel=0):
    #Inizializing the counters
    e, e2, rej=0,0,0
    #Energies for the gradient
    e_alfa, e_beta, e_gamma =0,0,0
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
        e+=local_energy(R,Alfa,Beta, Gamma)
        e2+=(local_energy(R, Alfa, Beta, Gamma))**2
        #Computation of energies for Shifted parameters__________________________________________________________
        e_alfa+=local_energy(R, Alfa+h, Beta, Gamma)
        e_beta+=local_energy(R, Alfa, Beta+h, Gamma)
        e_gamma+=local_energy(R, Alfa, Beta, Gamma+h)
    #rejection ratio to check if the sampling is correct
    R_Ratio=100*rej/N_met 
    #Averages for the metropolis
    E=e/N_met
    E2=e2/N_met
    #Averages for the gradient
    E_alfa=e_alfa/N_met
    E_beta=e_beta/N_met
    E_gamma=e_gamma/N_met
    #variance
    Var_E=(E2-E**2)/N_met
    return E, Var_E, R_Ratio, E_alfa, E_beta, E_gamma
#Function that starts from the same coordinates R, N_walk random walkers 
def let_them_walk(R,N_walk):
    E, Var, rej = 0, 0, 0
    E_alfa,E_beta,E_gamma=0,0,0
    for i in range(N_walk):
        E_m, Var_m, rej_m, E_a, E_b, E_g=metropolis(R, N_met, N_nel)
        #Averages over walkers
        E+=E_m/N_walk
        Var+=Var_m
        rej+=rej_m/N_walk
        #Averages over walkers for shifted parameters
        E_alfa+=E_a/N_walk
        E_beta+=E_b/N_walk
        E_gamma+=E_g/N_walk
    #Propagation of the error from each set. This correspond to sqrt(sum(sigma^2)/N^2)
    dE=np.sqrt(Var)/N_walk
    return E, dE, rej, E_alfa, E_beta, E_gamma

'''______________________________________________________________________'''
start = process_time()
#Number of particles, in that case 4 since we are considering the four nucleons
N_particles=4
#Number of loops for metropolis algoritm, in other words, number of step of the random walk
N_met, N_nel = 3000, 0
#Number of random walkers
N_walk=10
#Step for the gradient
h=0.00001

#Limits of parameters
L_alfa=5
L_beta=5
L_gamma=0.2

#Starting parameters
##Alfa, Beta, Gamma  = -0.7191, 2.0520, 0.08597
Alfa, Beta, Gamma = np.random.uniform(-L_alfa,L_alfa), np.random.uniform(0.001,L_beta), np.random.uniform(0.001,L_gamma)
print('Starting parameters:(%.3f' %Alfa, ', %.3f' %Beta, ', %.3f' %Gamma,')')

# Matrix of the coordinates
R=coordinate_generator()
#learning ratios
l_alfa=0.01
l_beta=0.01
l_gamma=0.00001

for i in range(10):
    E, dE, rej, E_a, E_b, E_c=metropolis(R, N_met)
    print('The energy is:%2.f' %E)
    #Components of the gradient
    G_a=(E_a-E)/h
    G_b=(E_b-E)/h
    G_g=(E_c-E)/h
    print('Gradient= %.2f'%G_a,'%.2f'% G_b, '%.2f'% G_g)
    #Modification of parameters- Gradient descent
    Alfa-=l_alfa*G_a
    Beta-=l_beta*G_b
    Gamma-=l_gamma*G_g
    print('Intermediate parameters:(%.3f' %Alfa, ', %.3f' %Beta, ', %.3f' %Gamma,')')
print('Final parameters:(%.3f' %Alfa, ', %.3f' %Beta, ', %.3f' %Gamma,')')
print('Energy=', E)
end = process_time()
print('Tempooooo=', end-start)