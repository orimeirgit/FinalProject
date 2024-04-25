### see https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
###  0: charged hadrons
###  1: electrons      
###  2: muons          
###  3: neutral hadrons
###  4: photons    
###  5: residual
### -1: neutrinos

pdgid_class_dict = {
    -211:   0,  211:  0,  # pi+-
    -321:   0,  321:  0,  # kaon+-
    -411:   0,  411:  0,  # D+-
    -431:   0,  431:  0,  # D_s+-
    -521:   0,  521:  0,  # B+-
    -2212:  0, 2212:  0,  # proton
    -3112:  0, 3112:  0,  # sigma-
    -3312:  0, 3312:  0,  # xi+-
    -3222:  0, 3222:  0,  # sigma+ 
    -3334:  0, 3334:  0,  # omega
    -11:    1,   11:  1,  # e
    -13:    2,   13:  2,  # mu
    -111:   3,  111:  3,  # pi0
                130:  3,  # K0L
                310:  3,  # K0S
    -2112:  3, 2112:  3,  # neutrons
    -3122:  3, 3122:  3,  # lambda
    -3322:  3, 3322:  3,  # xi0
    22:     4,            # photon
    1000010020:  0,       # deuteron 
    1000010030:  0,       # triton
    1000010040:  0,       # alpha
    1000020030:  0,       # He3
    1000020040:  0,       # He4
    1000030040:  0,       # Li6
    1000030050:  0,       # Li7
    -999:   5,            # residual
    -12:   -1,   12: -1,  # nu_e
    -14:   -1,   14: -1,  # nu_mu
    -16:   -1,   16: -1,  # nu_tau
}