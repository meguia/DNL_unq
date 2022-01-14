# plot utils para SINDy y curso DNL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist,squareform
from scipy.signal import find_peaks
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D

def solve(func,t,x0,method='DOP853',args=None):
    """ Simple function for calling solve_ivp with system func, initial condition  x0 and time evaluation t
    """ 
    dt = np.abs(t[1]-t[0])
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args,max_step=dt,dense_output=True)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def var_apply(x,var):
    """ Custom function for applying function in string var to array x
    """
    if var=='angle':
        return (x+np.pi)%(2*np.pi)-np.pi
    if var=='cos':
        return np.cos(x%(2*np.pi))
    if var=='sin':
        return np.sin(x%(2*np.pi))
    else:
        return x

def findperiod(t,x):
    """ Simple function for finding mean period of signal x(t) with a single peak for cycle
    """    
    peaks, _ = find_peaks(x)
    per = np.diff(t[peaks])
    return np.mean(per)

### MAPS

def map_plot(f, xini, N, *pars):
    """ 2D Plot of a map defined by f (and parameters *pars) with initial condition xini and N steps
    """
    fig, ax = plt.subplots(figsize=(20,5))
    for x in xini:
        xn = [x]
        for t in range(N):
            x = f(x,*pars)
            xn.append(x)
        ax.plot(xn,'-o');
    plt.grid()  
    plt.show()

def cobweb(f, xini, N, *pars, fscale=1.25):
    '''
    Cobweb Plot of the map defined by f (and parameters *pars) with initial condition xini and N steps
    '''
    if type(xini) not in [list,np.ndarray]:
        xini = [xini]
    (xmax,xmin)=[0,0]
    fig, ax = plt.subplots(figsize=(20,10))
    tabcolors = list(mcolors.TABLEAU_COLORS)
    for m,x in enumerate(xini):
        xn = [x]
        for n in range(N):
            x = f(x,*pars)
            xn.append(x)
        (xmax,xmin) = [max(np.max(xn),xmax), min(np.min(xn),xmin)]
        clr = tabcolors[m%10]
        ax.scatter(xn[0],0,30,clr)
        ax.scatter(xn[:-1],xn[1:],10,clr)
        ax.scatter(xn[1:],xn[1:],10,clr)
        ax.vlines(xn[0],0,xn[1],clr,linewidths=0.5)
        ax.hlines(xn[1:],xn[:-1],xn[1:],clr,linewidths=0.5)
        ax.vlines(xn[1:-1],xn[1:-1],xn[2:],clr,linewidths=0.5)
    xrange=xmax-xmin
    # arrays para graficar x y f(x) con 300 pts en el grafico
    xarr = np.linspace(xmin-(fscale-1)*xrange,xmax+(fscale-1)*xrange,num=300)
    yarr = f(xarr,*pars)
    (ymax,ymin) = [max(np.max(yarr),xarr[-1]), min(np.min(yarr),xarr[0])]
    ax.plot(xarr,yarr,'-k')
    ax.plot(xarr,xarr,'--k')
    if xarr[0]<0<xarr[-1]:
        ax.spines['left'].set_position('zero')
        ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()

def orbitdiag(f, xini, Tini, Tfin, parlist, fscale=1,msize=5,AMAX=1e2,alphaval=1):
    ''' Grafica el diagrama de orbitas del mapa dado por f para las condiciones iniciales 
    xini luego de un transitorio de Tini pasos. parlist es la lista de valores de parametros
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    xmin = 0
    xmax = 0
    for p in parlist:
        for x in xini:
            t = 0
            while t<Tini and np.abs(x)<AMAX:
                x = f(x,p)
                t += 1
            xn = [x]  
            while t<Tfin and np.abs(x)<AMAX:
                x = f(x,p)
                xn.append(x)
                t += 1   
            if (t>=Tini):    
                ax.plot([p]*len(xn),xn,'.', markersize=msize,alpha=alphaval); 
                xmax = max(np.max(xn),xmax)
                xmin = min(np.min(xn),xmin)
    ax.set_ylim([xmin*fscale,xmax*fscale])
    plt.show()

### FLOWS

def plot1D_labels(t,x,labels,ranges=[-1,1],varname=None):    
    """ Time evolution, x vs t plot with labels
    """
    plot_kws = dict(linewidth=2)
    if varname is None:
        varname = 'x'
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    axs.plot(t, x[:, 0], "b", label=varname, **plot_kws)
    axs.set(xlabel="$t$", ylabel=varname,title=labels,ylim=ranges) 
    axs.legend()
    axs.grid()    
    plt.show()

def plot1D_labels_fft(t,x,labels,ranges=[[-1,1],[-1,1]],var='',fmax=None):
    """ Time evolution and FFT spectrum of single variable x
    """
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(t, x[:,0], "r", label=var+' x', **plot_kws)
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel=var+' x')
    fnyq = 0.5/(t[1]-t[0])
    if fmax is None:
        fmax=fnyq
    df = 2*fnyq/len(t)
    f = np.arange(0,fnyq,df)
    y = np.abs(fft(x[:,0]))
    y = y[:len(f)]
    axs[1].plot(f, 20*np.log10(y), "r", label="$fft$", **plot_kws)
    axs[1].legend()
    axs[1].set(xlabel="t", ylabel="$fft amplitude (dB)$",title="FFT transform",xlim=[0,fmax])        
    plt.show()

def plot2D_labels(t,x,labels,ranges=[[-1,1],[-1,1]],var=[0,1]): 
    """ Time evolution and phase plot of variables x[var[0]], x[var[1]]
    """
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    for n in range(len(x[0,:])):
        axs[0].plot(t, x[:,n], label='x_'+str(n), alpha=1-n*0.2,**plot_kws)
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel="$x_k$")
    axs[1].plot(x[:,var[0]], x[:,var[1]], "r", label="$x_k$", **plot_kws)
    axs[1].legend()
    axs[1].plot(x[0,var[0]], x[0,var[1]], "ro")
    axs[1].set(xlabel='x_'+str(var[0]), ylabel='x_'+str(var[1]),title=labels,xlim=ranges[0],ylim=ranges[1])    
    axs[1].grid()
    plt.show()


def plot2D_labels_fft(t,x,labels,ranges=[[-1,1],[-1,1]],fmax=None,var=[0,1]):    
    """ Time evolution, Phase plot and FFT spectrum of variables x[var[0]], x[var[1]]
    """
    plot_kws = dict(linewidth=2)
    fig = plt.figure(figsize=(18, 8))
    gd = gridspec.GridSpec(2, 2)
    axs1 = plt.subplot(gd[0,0])
    axs2 = plt.subplot(gd[1,0])
    axs3 = plt.subplot(gd[:,1])
    axs1.plot(t, x[:, var[0]], "r", label="$x_0$", **plot_kws)
    axs1.plot(t, x[:, var[1]], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs1.legend()
    axs1.set(xlabel="t", ylabel="$x_k$")
    axs3.plot(x[:, var[0]], x[:, var[1]], "r", label="$x_k$", **plot_kws)
    axs3.legend()
    axs3.plot(x[0, var[0]], x[0, var[1]], "ro")
    axs3.set(xlabel="$x_0$", ylabel="$x_1$",title=labels,xlim=ranges[0],ylim=ranges[1])    
    axs3.grid()
    fnyq = 0.5/(t[1]-t[0])
    if fmax is None:
        fmax=fnyq
    df = 2*fnyq/len(t)
    f = np.arange(0,fnyq,df)
    y0 = np.abs(fft(x[:,var[0]]))
    y1 = np.abs(fft(x[:,var[1]]))
    y0 = y0[:len(f)]
    y1 = y1[:len(f)]
    axs2.plot(f, 20*np.log10(y0), "r", label="$fft(x_0)$", **plot_kws)
    axs2.plot(f, 20*np.log10(y1), "b", label="$fft(x_1)$", alpha=0.4, **plot_kws)
    axs2.legend()
    axs2.set(xlabel="f", ylabel="$fft amplitude (dB)$",title="FFT transform",xlim=[0,fmax])    
    plt.show()



def solve_plot(system,pars,xini,tmax,dt,ranges=None,wfft=False,var=None,method='RK45',trans=None,fmax=None):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    x = solve(system, t, xini, args=args, method=method)
    if ranges is None:
        if len(xini)==1:
            ranges = [np.min(x[:,0]),np.max(x[:,0])]
        else:
            ranges = [[np.min(x[:,0]),np.max(x[:,0])],[np.min(x[:,1]),np.max(x[:,1])]]
    if trans is not None:
        t0 = int(trans/dt)
        x = x[t0:,:]
        t = t[t0:]
    x = var_apply(x,var)
    if len(xini) == 1:
        if wfft:
            plot1D_labels_fft(t,x,labels,ranges,var,fmax)
        else: 
            plot1D_labels(t,x,labels,ranges,var)
    else:    
        if wfft:
            plot2D_labels_fft(t,x,labels,ranges,fmax)
        else:    
            plot2D_labels(t,x,labels,ranges)  
    plt.show()

    


def solve_plot1D_multiple(system,pars,xini_array,tmax,dt,xrange=[-1,1],var=None,method='RK45'):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    axs.grid()  
    for n,xini in enumerate(xini_array):
        x = solve(system, t, [xini], args=args, method=method)
        x = var_apply(x,var)
        axs.plot(t[:len(x)], x[:, 0], **plot_kws)
        axs.set_ylim(xrange)
    axs.set(xlabel="$t$", ylabel="$x_0$",title=labels)
    plt.show()


def solve_plot1D_dual(system,pars,xini,tmax,dt,xrange=[-1,1],fmax=1.0,var=None,method='RK45'):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    x0 = np.linspace(xrange[0],xrange[1],100)
    f = system(t,x0,*args)
    x = solve(system, t, xini, args=args, method=method)
    x = var_apply(x,var)
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(x0, f[0], "k", label="$f(x)$", **plot_kws)
    axs[0].plot(x0, 0*f[0], "r", label="$0$", **plot_kws)
    axs[0].plot(x[:,0], 0*x[:,0], "b", label="$x$", **plot_kws)
    axs[0].set(xlabel="$x0$", ylabel="$f(x_0)$",title=labels) 
    axs[0].set_xlim(xrange)
    axs[0].grid()   
    axs[1].plot(t[:len(x)], x[:, 0], "b", label="$x_0$", **plot_kws)
    axs[1].set(xlabel="$t$", ylabel="$x_0$",title=labels) 
    axs[1].legend()
    axs[1].set_ylim(xrange)
    axs[1].grid()
    plt.show()

# diagrama de bifurcaciones para flujos 1D y 2D
def bifurcation_diag(system, pars, xini_list, tmax, dt, parval, parlist,vi=0,xrange=[-1,1],msize=5,var=None,method='RK45'):
    ''' Grafica el diagrama de bifurcaciones del flujo de systems evolucionando para atras 
    y para adelante en el tiempo y usando xrange como bound.
    pars es la lista de parametros fijos parval es el nombre del parametro a variar y parlist es la lista de 
    valores de ese parametro
    Para 2D se puede cambiar el indice de la variable en vi
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    ax.grid()
    t = np.arange(0, tmax, dt)
    for p in parlist:
        print(p)
        pars[parval]=p
        args = tuple(pars.values())
        for xini in xini_list:
            x = solve(system, t, xini, args=args, method='RK45')
            pt = x[-1,vi]
            if pt<xrange[1] and pt>xrange[0]:
                x = var_apply(x,var)
                ax.plot(p,pt,'b.', markersize=msize); 
            x = solve(system, -t, xini, args=args, method='RK45')
            pt = x[-1,vi]
            if pt<xrange[1] and pt>xrange[0]:
                x = var_apply(x,var)
                ax.plot(p,pt,'r.', markersize=msize);
    plt.show()


def solve_plot2D_multiple(system,pars,xini_array,tmax,dt,ranges=[[-1,1],[-1,1]],method='RK45'):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    plot_kws = dict(linewidth=0.5)
    fig = plt.figure(figsize=(18, 8))
    gd = gridspec.GridSpec(2, 2)
    axs1 = plt.subplot(gd[0,0])
    axs2 = plt.subplot(gd[1,0])
    axs3 = plt.subplot(gd[:,1])
    axs1.grid()  
    axs3.grid()  
    for n,xini in enumerate(xini_array):
        x = solve(system, t, xini, args=args, method=method)
        axs1.plot(t[:len(x)], x[:, 0], **plot_kws)
        axs2.plot(t[:len(x)], x[:, 1], **plot_kws)
        axs1.set_ylim(ranges[0])
        axs2.set_ylim(ranges[1])
        axs3.plot(x[:, 0], x[:, 1], **plot_kws)
        axs3.set_xlim(ranges[0])
        axs3.set_ylim(ranges[1])
    axs1.set(xlabel="$t$", ylabel="$x_0$",title=labels) 
    axs2.set(xlabel="$t$", ylabel="$x_1$",title=labels) 
    axs3.set(xlabel="$x_0$", ylabel="$x_1$",title=labels)
    plt.show()

def solve_plot2D_nulclinas(system,pars,xini,tmax,dt,ranges=[[-1,1],[-1,1]],method='RK45'):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    plot_kws = dict(linewidth=2)
    fig, ax = plt.subplots(figsize=(20,10))
    delta = (ranges[0][1]-ranges[0][0])/100
    xrange = np.arange(ranges[0][0],ranges[0][1], delta)
    yrange = np.arange(ranges[1][0],ranges[1][1], delta)
    X, Y = np.meshgrid(xrange,yrange)
    Z = system(0,[X,Y],*args)
    ax.contourf(xrange,yrange,Z[0],levels=[-1000,0,1000],alpha=0.1,colors=('r','g'))
    ax.contourf(xrange,yrange,Z[1],levels=[-1000,0,1000],alpha=0.1,colors=('b','y'))
    ax.contour(xrange,yrange,Z[0],[0],colors='r')
    ax.contour(xrange,yrange,Z[1],[0],colors='b')
    ax.grid()  
    x = solve(system, t, xini, args=args, method=method)
    ax.plot(x[:, 0], x[:, 1], 'k', **plot_kws)
    ax.set(xlabel="$x$", ylabel="$y$",title=labels)
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    plt.show()

def solve_plot2D_quiver(system,pars,dx,dy,xini=None,tmax=0,dt=0,scale=None,ranges=[[-1,1],[-1,1]],method='RK45'):
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    plot_kws = dict(width=0.001)
    fig, ax = plt.subplots(figsize=(20,15))
    delta = (ranges[0][1]-ranges[0][0])/100
    xrange = np.arange(ranges[0][0],ranges[0][1], delta)
    yrange = np.arange(ranges[1][0],ranges[1][1], delta)
    X, Y = np.meshgrid(xrange,yrange)
    Z = system(0,[X,Y],*args)
    ax.contourf(xrange,yrange,Z[0],levels=[-1000,0,1000],alpha=0.1,colors=('r','g'))
    ax.contourf(xrange,yrange,Z[1],levels=[-1000,0,1000],alpha=0.1,colors=('b','y'))
    ax.contour(xrange,yrange,Z[0],[0],colors='r')
    ax.contour(xrange,yrange,Z[1],[0],colors='b')
    ax.grid()  
    x = np.arange(ranges[0][0],ranges[0][1], dx)
    y = np.arange(ranges[1][0],ranges[1][1], dy)
    X, Y = np.meshgrid(x,y)
    UV = system(0,[X,Y],*args)
    r = np.power(np.add(np.power(UV[0],2), np.power(UV[1],2)),0.5)
    ax.quiver(X,Y,UV[0]/r,UV[1]/r, scale=scale,**plot_kws) 
    ax.set(xlabel="$x$", ylabel="$y$",title=labels)
    if xini is not None:
        t = np.arange(0, tmax, dt)
        x = solve(system, t, xini, args=args, method=method)
        ax.plot(x[:, 0], x[:, 1],'k')
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    plt.show()    


def recurrence_plot(system,pars,xini,tmax,dt,var=0,eps=1e-6,method='RK45',trans=None,period=None):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    xv = solve(system, t, xini, args=args, method=method)
    x = xv[:,var]
    if trans is not None:
        t0 = int(trans/dt)
        x = x[t0:]
        t = t[t0:]
    steps = 10    
    dst = pdist(xv[t0:,:])
    dst = np.floor(dst/eps)/steps
    dst[dst>steps] = steps
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].plot(t, x, "r", label="$x$", **plot_kws)
    axs[0].legend()
    axs[0].grid()
    axs[0].set(xlabel="t", ylabel="$x$")
    if period is not None:
        ncycles = int(np.ceil((t[-1]-t[0])/period))
        for n in range(ncycles):
            axs[1].plot(t,t-n*period,'r',linewidth=0.3)
            axs[1].plot(t,t+n*period,'r',linewidth=0.3)
    axs[1].set_xlim([t[0],t[-1]])
    axs[1].set_ylim([t[0],t[-1]])
    axs[1].contour(t,t,squareform(dst))
    axs[1].set(xlabel="$t$", ylabel="$t$",title=labels)
    plt.show()        

def butterfly(system,pars,xini,tmax,dt,eps=1e-6,method='RK45'):
    """ Calcula la divergencia entre dos condiciones iniciales en 
    xini y xini + eps. Grafica las dos primeras variables y la distancia euclidiana 
    en funcion del tiempo para mostra el efecto 'mariposa'
    """
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    x1 = solve(system, t, xini, args=args, method=method)
    xini[0] = xini[0]+eps
    x2 = solve(system, t, xini, args=args, method=method)
    dx = np.linalg.norm(x2-x1,axis=-1)
    plot_kws = dict(linewidth=0.5)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))    
    axs[0].plot(x1[:,0],x1[:,1], "r", label="$x1$", **plot_kws)
    axs[0].plot(x2[:,0],x2[:,1], "b", label="$x2$", **plot_kws)
    axs[0].legend()
    axs[0].grid()
    axs[0].set(xlabel="x", ylabel="$y$")
    axs[1].semilogy(t,dx, "r", label="$dist$", **plot_kws)
    axs[1].set(xlabel="$t$", ylabel="$dist$",title=labels)
    plt.show()        



def solve_plot2D_linear(A,xini_array,tmax,dt,ranges=[[-1,1],[-1,1]],method='RK45'):
    t = np.arange(0, tmax, dt)
    pars = {'a':A[0,0],'b':A[0,1],'c':A[1,0],'d':A[1,1]}
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    tr = np.trace(A)
    det = np.linalg.det(A)
    maxtr = 1.3*np.max([np.abs(tr),2*np.sqrt(np.abs(det))])
    maxdet = 1.3*np.max([np.abs(det),tr**2/4])
    trv = np.linspace(-maxtr,maxtr,100)
    trdet = np.square(trv)/4
    axs[0].plot(tr,det,'or')
    axs[0].plot(trv,trdet,'-b')
    axs[0].plot(trv,trv*0,'-g')
    axs[0].plot([0,0],[0,maxdet],'-k')
    axs[0].grid()
    axs[0].set(xlabel="$Tr(A)$", ylabel="$Det(A)$") 
    axs[0].text(0, -maxdet/2, 'Saddle')
    axs[0].text(maxtr/4, maxdet/2, 'Foco\nRepulsor')
    axs[0].text(-maxtr/4, maxdet/2, 'Foco\nAtractor')
    axs[0].text(3*maxtr/4, maxdet/4, 'Nodo\nRepulsor')
    axs[0].text(-3*maxtr/4, maxdet/4, 'Nodo\nAtractor')
    axs[0].set_xlim([-maxtr,maxtr])
    axs[0].set_ylim([-maxdet,maxdet])
    def system(t,x,a,b,c,d):
        return[
            a*x[0]+b*x[1],
            c*x[0]+d*x[1]
        ]
 
    for n,xini in enumerate(xini_array):
        x = solve(system, t, xini, args=args, method=method)
        axs[1].plot(x[:, 0], x[:, 1], **plot_kws)
    axs[1].set_xlim(ranges[0])
    axs[1].set_ylim(ranges[1])
    axs[1].set(xlabel="$x_0$", ylabel="$x_1$",title=labels)
    plt.show()

def butterfly(system,pars,xini,tmax,dt,eps=1e-6,method='RK45'):
    t = np.arange(0, tmax, dt)
    args = tuple(pars.values())
    labels = ','.join([it[0]+ ' = ' + str(it[1]) for it in pars.items()])
    x1 = solve(system, t, xini, args=args, method=method)
    xini[0] = xini[0]+eps
    x2 = solve(system, t, xini, args=args, method=method)
    dx = np.linalg.norm(x2-x1,axis=-1)
    plot_kws = dict(linewidth=0.5)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))    
    axs[0].plot(x1[:,0],x1[:,1], "r", label="$x1$", **plot_kws)
    axs[0].plot(x2[:,0],x2[:,1], "b", label="$x2$", **plot_kws)
    axs[0].legend()
    axs[0].grid()
    axs[0].set(xlabel="x", ylabel="$y$")
    axs[1].semilogy(t,dx, "r", label="$dist$", **plot_kws)
    axs[1].set(xlabel="$t$", ylabel="$dist$",title=labels)
    plt.show()        


### SINDY

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

