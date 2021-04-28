# plot utils para SINDy y curso DNL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D

def solve(func,t,x0,method='DOP853',args=None):
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def findperiod(t,x):
    peaks, _ = find_peaks(x)
    per = np.diff(t[peaks])
    return np.mean(per)

def plot2D_test(t,x_test):    
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(t, x_test[:, 0], "r", label="$x_0$", **plot_kws)
    axs[0].plot(t, x_test[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel="$x_k$")
    axs[1].plot(x_test[:, 0], x_test[:, 1], "r", label="$x_k$", **plot_kws)
    axs[1].legend()
    axs[1].set(xlabel="$x_0$", ylabel="$x_1$")
    
    
def plot2D_labels(t,x,labels,ranges=[[-1,1],[-1,1]]):    
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(t, x[:, 0], "r", label="$x_0$", **plot_kws)
    axs[0].plot(t, x[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel="$x_k$")
    axs[1].plot(x[:, 0], x[:, 1], "r", label="$x_k$", **plot_kws)
    axs[1].legend()
    axs[1].plot(x[0, 0], x[0, 1], "ro")
    axs[1].set(xlabel="$x_0$", ylabel="$x_1$",title=labels,xlim=ranges[0],ylim=ranges[1])    
    axs[1].grid()

def plot2D_labels_fft(t,x,fmax,labels,ranges=[[-1,1],[-1,1]]):    
    plot_kws = dict(linewidth=2)
    fig = plt.figure(figsize=(18, 8))
    gd = gridspec.GridSpec(2, 2)
    axs1 = plt.subplot(gd[0,0])
    axs2 = plt.subplot(gd[1,0])
    axs3 = plt.subplot(gd[:,1])
    axs1.plot(t[::10], x[::10, 0], "r", label="$x_0$", **plot_kws)
    axs1.plot(t[::10], x[::10, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs1.legend()
    axs1.set(xlabel="t", ylabel="$x_k$")
    axs3.plot(x[:, 0], x[:, 1], "r", label="$x_k$", **plot_kws)
    axs3.legend()
    axs3.plot(x[0, 0], x[0, 1], "ro")
    axs3.set(xlabel="$x_0$", ylabel="$x_1$",title=labels,xlim=ranges[0],ylim=ranges[1])    
    axs3.grid()
    fnyq = 0.5/(t[1]-t[0])
    df = fnyq/len(t)
    f = np.arange(0,fnyq,df)
    y0 = np.abs(fft(x[:,0]))
    y1 = np.abs(fft(x[:,1]))
    axs2.plot(f, 20*np.log10(y0), "r", label="$fft(x_0)$", **plot_kws)
    axs2.plot(f, 20*np.log10(y1), "b", label="$fft(x_1)$", alpha=0.4, **plot_kws)
    axs2.legend()
    axs2.set(xlabel="t", ylabel="$fft amplitude (dB)$",title="FFT transform",xlim=[0,fmax])    
    
    
def solve_plot(system,pars,xini,tmax,dt,ranges=[[-1,1],[-1,1]],fmax=1.0,wfft=False):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    x = solve(system, t, xini, args=args, method='RK45')
    if wfft:
        plot2D_labels_fft(t,x,fmax,labels,ranges)
    else:    
        plot2D_labels(t,x,labels,ranges)    

    
 # doble oscilador    
def plot4D_test(t,x_test):    
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    print(axs)
    for n in range(2):
        xa = "$x_" + str(2*n+0) + "$"
        xb = "$x_" + str(2*n+1) + "$"
        axs[n,0].plot(t, x_test[:, 2*n+0], "r", label=xa, **plot_kws)
        axs[n,0].plot(t, x_test[:, 2*n+1], "b", label=xb, alpha=0.5, **plot_kws)
        axs[n,0].legend()
        axs[n,0].set(xlabel="t", ylabel="x")
        axs[n,1].plot(x_test[:, 2*n+0], x_test[:, 2*n+1], "r", label=xa + xb, **plot_kws)
        axs[n,1].legend()
        axs[n,1].set(xlabel=xa, ylabel=xb)  

def plot4D_labels(t,x,labels,ranges,curves=[]):    
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    print(axs)
    for n in range(2):
        xa = "$x_" + str(2*n+0) + "$"
        xb = "$x_" + str(2*n+1) + "$"
        axs[n,0].plot(t, x[:, 2*n+0], "r", label=xa, **plot_kws)
        axs[n,0].plot(t, x[:, 2*n+1], "b", label=xb, alpha=0.5, **plot_kws)
        axs[n,0].legend()
        axs[n,0].set(xlabel="t", ylabel="x")
        axs[n,1].plot(x[:, 2*n+0], x[:, 2*n+1], "r", label=xa + xb, **plot_kws)
        axs[n,1].legend()
        axs[n,1].plot(x[0, 2*n+0], x[0, 2*n+1], "ro")
        axs[n,1].set(xlabel=xa, ylabel=xb,title=labels,xlim=ranges[n,0],ylim=ranges[n,1])  
        axs[n,1].grid()
    # bifurcaciones
    for c in curves:
        axs[1,1].plot(c[0],c[1])        
        
 
# grafica train y sim y compara 
def testsim(model,func,t,x0, method='DOP853'):
    x_test = solve(func, t, x0, method=method)
    x_sim = model.simulate(x0, t)
    return x_test, x_sim
    
def plot2D_testsim(t,x_test,x_sim):    
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(t, x_test[:, 0], "r", label="$x_0$", **plot_kws)
    axs[0].plot(t, x_test[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs[0].plot(t, x_sim[:, 0], "k--", label="model", **plot_kws)
    axs[0].plot(t, x_sim[:, 1], "k--")
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel="$x_k$")
    axs[1].plot(x_test[:, 0], x_test[:, 1], "r", label="$x_k$", **plot_kws)
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
    axs[1].legend()
    axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
    
def train_mu(mu_stable, mu_unstable, func, t, x0_stable, x0_unstable, eps, method='DOP853'):
    n_ics = mu_stable.size + 2 * mu_unstable.size
    x_train = [np.zeros((t.size, 3)) for i in range(n_ics)]
    ic_idx = 0
    for mu in mu_stable:
        x = solve(lambda t, x: func(x, mu), t, x0_stable, method=method)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        ic_idx += 1
    for mu in mu_unstable:
        x = solve(lambda t, x: func(x, mu), t, x0_unstable, method=method)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        ic_idx += 1
        x = solve(lambda t, x: func(x, mu), t, x0_stable, method=method)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        ic_idx += 1
    return x_train

def testsim_mu(mu_stable, mu_unstable, model, func, t, x0_stable, x0_unstable, method='DOP853'):
    n_ics = mu_stable.size + 2 * mu_unstable.size
    x_test = [np.zeros((t.size, 3)) for i in range(n_ics)]
    x_sim = [np.zeros((t.size, 3)) for i in range(n_ics)]
    ic_idx = 0
    for mu in mu_stable:
        x_test[ic_idx][:, 0:2] = solve(lambda t, x: func(x, mu), t, x0_stable, method=method)
        x_sim[ic_idx] = model.simulate(np.array(x0_stable + [mu]), t)
        x_test[ic_idx][:, 2] = mu
        ic_idx += 1
    for mu in mu_unstable:
        x_test[ic_idx][:, 0:2] = solve(lambda t, x: func(x, mu), t, x0_unstable, method=method)
        x_sim[ic_idx] = model.simulate(np.array(x0_unstable + [mu]), t)
        x_test[ic_idx][:, 2] = mu
        ic_idx += 1
        x_test[ic_idx][:, 0:2] = solve(lambda t, x: func(x, mu), t, x0_stable, method=method)
        x_sim[ic_idx] = model.simulate(np.array(x0_stable + [mu]), t)
        x_test[ic_idx][:, 2] = mu
        ic_idx += 1
    return x_test, x_sim

# Plot results
def plot3D_testsim(x_test,x_sim):
    n_ics = np.shape(x_test)[0]
    fig = plt.figure(figsize=(28, 8))
    plot_kws=dict(alpha=0.75, linewidth=1)
    ax = fig.add_subplot(121, projection="3d")
    for i in range(n_ics):
        if i > 2 and i % 2 == 0:
            ax.plot(
                x_test[i][:, 2], x_test[i][:, 0], x_test[i][:, 1], "r", **plot_kws)
        else:
            ax.plot(x_test[i][:, 2], x_test[i][:, 0], x_test[i][:, 1], "b", **plot_kws)
    ax.set(title="Full Simulation", xlabel="$\mu$", ylabel="x", zlabel="y")
    ax = fig.add_subplot(122, projection="3d")
    for i in range(n_ics):
        if i > 2 and i % 2 == 0:
            ax.plot(x_sim[i][:, 2], x_sim[i][:, 0], x_sim[i][:, 1], "r", **plot_kws)
        else:
            ax.plot(x_sim[i][:, 2], x_sim[i][:, 0], x_sim[i][:, 1], "b", **plot_kws)
    ax.set(title="Identified System", xlabel="$\mu$", ylabel="x", zlabel="y")
    

