from mpl_toolkits.mplot3d import Axes3D
from gp import *
from plot_utils import *

K = K_SE

# One-dimensional posterior samples
xs = linspace(0,10,100)
mean = zeros_like(xs)
cov = K(xs)
for i in range(7):
    plot(xs, multivariate_normal(mean, cov))
show()

# Two-dimensional sample
Nx = 45; Ny = 30
D,X,Y = grid2d(0,15,Nx, 0,10,Ny)
Z = multivariate_normal([0]*(Nx*Ny), K(D))
Z = Z.reshape(Nx,Ny).T # cols for x, rows for y
contour(X,Y,Z,20)
ax = Axes3D(figure())
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
show()

# 2D test function
X,Y = mgrid[0:10:100j,0:10:100j]
#Z = bivariate_normal(X,Y, mux=1,muy=1) + bivariate_normal(X,Y, mux=-1,muy=-1)
Z = sin(X) + sin(Y)
contour(X,Y,Z)
show()

# Posterior given data from $y = \sin(x)$
xs = linspace(0,10,7)
ys = sin(xs)
ts = linspace(0,10,100)

mean,var = mvg_cond(xs, ys, ts)
errbars(ts, mean, sqrt(var)); show()

mean,cov = mvg_cond(xs, ys, ts, full=True)
for i in xrange(1000):
    plot(ts, multivariate_normal(mean, cov), 'k', alpha=.01)
show()

# 2D posterior given data from $z = \sin(x) + \sin(y)$
N1 = 60; N2 = 30
D,D1,D2 = grid2d(0,10,5, 0,5,4) # training data inputs
Y = sin(D[:,0]) + sin(D[:,1])   #    "      "   outputs
T,X1,X2 = grid2d(0,10,N1, 0,5,N2) # test points
mean,var = mvg_cond(D, Y, T)

Z = mean.reshape(N1,N2).T
V = var.reshape(N1,N2).T; S = sqrt(V)
F = sin(X1) + sin(X2)

plot(D[:,0], D[:,1], 'x')
contour(X1,X2,Z,20)
contour(X1,X2,F,20, linestyles='dotted', linewidths=2)

plot_2d_approx(X1, X2, Z, S, F)
show()
