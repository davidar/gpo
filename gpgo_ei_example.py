from gpgo_ei import *
from plot_utils import *

from time import sleep

# GPGO-EI (1D)


ion() # interactive mode

def gpgo_ei_1d(f, xs, ts, showit=True, noise=0.001):
    xs = list(xs); ys = map(f,xs)
    mei = Inf # maximum expected improvement
    while mei > noise:
        mean,var = mvg_cond(xs,ys,ts)
        sd = sqrt(var)
        ei = EI(mean, sd, max(ys))
        mei = max(ei); t = ts[argmax(ei)]
        
        if showit:
            clf()
            plot(xs,ys,'x')
            errbars(ts, mean, sd)
            plot(ts,f(ts),':')
            plot(ts, ei, 'r')
            plot(ts, ei/mei, 'r:')
            plot(t, mei, 'rx')
            #show()
            draw(); sleep(1)
        
        xs.append(t); ys.append(f(t))
    return xs,ys


# $\sin(x)$ objective function
gpgo_ei_1d(sin, [0,10], linspace(0,10,101))

# Objective function sampled from GP
f = GPmem(1)
gpgo_ei_1d(f, [0,10], linspace(0,10,101))


# GPGO-EI (2D)

ioff() # non-interactive

def gpgo_ei_2d(f, D, N1=60, N2=30, showit=True, noise=0.001):
    D = list(D); Y = list(f(D))
    #ax = Axes3D(figure())
    mei = Inf # maximum expected improvement
    while mei > noise:
        T,X1,X2 = grid2d(0,D[1][0],N1, 0,D[1][1],N2) # test points
        mean,var = mvg_cond(asarray(D), ascolumn(Y), T)
        sd = sqrt(var)
        
        ei = EI(mean, sd, max(Y))
        mei = max(ei)
        t = T[argmax(ei)]
        
        if showit:
            Z = mean.reshape(N1,N2).T
            V = var.reshape(N1,N2).T; S = sqrt(V)
            F = f(T).reshape(N1,N2).T
            E = absolute(F - Z)
            
            #clf()
            ax = plot_2d_approx(X1, X2, Z, S, F)
            zlo = ax.get_zlim()[0]
            ax.plot(asarray(D)[:,0], asarray(D)[:,1], zlo, 'o')
            ax.plot([t[0]], [t[1]], zlo, 'or')
            ax.contour(X1,X2,Z,20, offset=zlo)
            ax.contour(X1,X2,F,20, offset=zlo, linestyles='dotted', linewidths=2)
            show()
            #ax.draw(); sleep(10)
        
        D.append(t); Y.append(f(atleast_2d(t)))
    
    return asarray(D), asarray(Y)

# $z = \sin(x) + \sin(y)$ objective function
def sinsin(X):
    X = atleast_2d(X)
    return sin(X[:,0]) + sin(X[:,1])
gpgo_ei_2d(sinsin, [[0,0], [10,5]])

# Objective function sampled from GP
gpgo_ei_2d(GPmem(2), [[0,0], [10,5]], 30,15)



#### vs Simulated Annealing

# GPGO-EI and simulated annealing run on 25 objective functions sampled from 5-dimensional GP.

ioff() # non-interactive

class Fmem(object):
    def __init__(self, f, lower=-Inf, upper=Inf):
        self.xs = []
        self.ys = []
        self.f = f
        self.lower = lower
        self.upper = upper
    def __call__(self, x):
        if any(asarray(x) < asarray(self.lower)) or            any(asarray(x) > asarray(self.upper)):
            y = -Inf
        else:
            y = self.f(x)
            self.xs.append(x)
            self.ys.append(y)
        return y

for dims in [1,2,5,10,20,50,100]:
    print dims,
    
    maxeval=150
    gpgo_avg = zeros(maxeval)
    anneal_avg = zeros(maxeval)
    
    N = 25
    for _ in xrange(N):
        g = GPmem(dims)
        
        xs,ys = gpgo_ei(g, [[0]*dims,[10]*dims], lower=[0]*dims, upper=[10]*dims, maxeval=maxeval)
        rmax = array(list(maximum.accumulate(ys))[:maxeval] + [max(ys)] * (maxeval-len(ys)))
        plot(rmax, 'b', alpha=0.2)
        gpgo_avg += rmax
        
        f = Fmem(g, [0]*dims, [10]*dims)
        anneal(lambda x: -f(x), [0]*dims, lower=[0]*dims, upper=[10]*dims, maxeval=maxeval)
        rmax = array(list(maximum.accumulate(squeeze(f.ys)))[:maxeval] + [max(f.ys)] * (maxeval-len(f.ys)))
        plot(rmax, 'r', alpha=0.2)
        anneal_avg += rmax
        
        print '.',
    
    print
    
    plot(gpgo_avg / N, 'b', linewidth=2)
    plot(anneal_avg / N, 'r', linewidth=2)
    ylim(1,3)
    show()
