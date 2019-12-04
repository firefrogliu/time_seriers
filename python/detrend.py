import numpy as np
from PyEMD import EMD, Visualisation, EEMD, CEEMDAN

def moving_average(a, t=3) :
    ma = np.cumsum(a, dtype=float)
    ma[t:] = ma[t:] - ma[:-t]
    ma[:t-1] = a[:t-1]

    ma = ma[t-1:]/t
    
    ma_residual = a[t-1:] - ma
    return ma, ma_residual


def emd_detrend(S, max_imf = 9):


    print('s', S.shape)
    S = S.reshape((S.shape[0],))
    t = np.arange(0, S.shape[0])

    # In case of EMD
    emd = EMD()
    emd.emd(S,t, max_imf)

    imfs, res = emd.get_imfs_and_residue()

    
    # In general:
    # components = EMD()(S)
    # imfs, res = components[:-1], components[-1]

    
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()

    
    imfs = np.transpose(imfs)
    res = res.reshape((res.shape[0],1))

    imfs_res = np.append(imfs, res, axis=1)
    print('imfs_res', imfs_res.shape)
    return imfs_res

def ceemd_detrend(S, max_imf = 8):


    print('s', S.shape)
    S = S.reshape((S.shape[0],))
    t = np.arange(0, S.shape[0])

    # In case of EMD
    ceemd = CEEMDAN()
    components = ceemd.ceemdan(S,t, max_imf)

    #imfs, res = ceemd.get_imfs_and_residue()

    
    # In general:

    imfs, res = components[:-1], components[-1]

    
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()

    
    imfs = np.transpose(imfs)
    res = res.reshape((res.shape[0],1))

    imfs_res = np.append(imfs, res, axis=1)
    print('imfs_res', imfs_res.shape)
    return imfs_res



import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  

def vmd_detrend(f):
#. Time Domain 0 to T  
    # T = 1000  
    # fs = 1/T  
    # t = np.arange(1,T+1)/T  
    # freqs = 2*np.pi*(t-0.5-fs)/(fs)  

    # #. center frequencies of components  
    # f_1 = 2  
    # f_2 = 24  
    # f_3 = 288  

    # #. modes  
    # v_1 = (np.cos(2*np.pi*f_1*t))  
    # v_2 = 1/4*(np.cos(2*np.pi*f_2*t))  
    # v_3 = 1/16*(np.cos(2*np.pi*f_3*t))  

    # f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)  

    #. some sample parameters for VMD  
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 5              # 3 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  


    #. Run actual VMD code  
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  

    u = np.transpose(u)

    print('u', u.shape)
    print('u_hat', u_hat.shape)
    print('omega,', omega.shape)

    #plt.plot(u)
    #plt.plot(u_hat)
    #plt.plot(omega)
    #plt.show()
    return u

if __name__ == '__main__':

    a = np.array([1,7,9,2,3,6,2,8,4,5])
    print(a[0:-1])
    


    # ma, ma_residual = moving_average(a)
    # print(a)
    # print(ma)
    # print(ma_residual)


    #ceemd_detrend(a)
    # emd_detrend(a)
    vmd_detrend(a)
