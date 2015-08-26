from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def errbars(ts, mean, sd):
    plot(ts, mean, 'k')
    plot(ts, mean + 2*sd, '--b')
    plot(ts, mean - 2*sd, '--b')

def norm01(Z):
    zmin = Z.min(); zmax = Z.max()
    return (Z - zmin) / (zmax - zmin)

def fc_alpha(Z, A, cmap=cm.jet):
    C = cmap(Z)
    for i in xrange(C.shape[0]):
        for j in xrange(C.shape[1]):
            C[i,j][3] = A[i,j]
    return C

def plot_2d_approx(X1, X2, Z, S, F):
    ax = Axes3D(figure())
    ax.plot_surface(X1,X2,Z, rstride=1, cstride=1,
            facecolors=fc_alpha(norm01(Z), 1 - .975*(S / S.max())))
    E = absolute(F - Z)
    ax.plot_surface(X1,X2,F, rstride=1, cstride=1,
            facecolors=fc_alpha(ones_like(F), .1*(E / E.max()), cm.Blues))
    zlo = 2 * ax.get_zlim()[0]; ax.set_zlim(bottom=zlo)
    ax.contour(X1,X2,Z,20, offset=zlo)
    ax.contour(X1,X2,F,20, offset=zlo, linestyles='dotted', linewidths=2)
    return ax
