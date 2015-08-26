from pylab import *
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import fmin_cg
from scipy.linalg import pinvh
from scipy.stats import lognorm

def ascolumn(xs):
    if rank(xs) < 2:
        return atleast_2d(xs).T
    return asarray(xs)

def allpairs(xs,ys):
    return array([[x,y] for x in xs for y in ys])

def grid2d(xmin, xmax, xnum, ymin, ymax, ynum):
    xs = linspace(xmin,xmax,xnum)
    ys = linspace(ymin,ymax,ynum)
    D = allpairs(xs,ys) # observation matrix (Nx2)
    X,Y = meshgrid(xs,ys)
    return D,X,Y

def eye_like(mat):
    dims = mat.shape
    return eye(*dims)

def dqf(A,B):
    # calculate diag(A * B * A.T)
    # stackoverflow.com/q/14758283
    return einsum('ij,ij->i', dot(A,B), A)

def sameshape(a,b):
    return asarray(a).reshape(shape(b))

class GPmem(object):
    def __init__(self, dim):
        self.dim = dim
        self.d = {tuple([0]*dim) : randn()}
    def __call__(self, xs):
        if self.dim > 1: xs = atleast_2d(xs)
        else:            xs = atleast_1d(xs)
        mean,cov = mvg_cond(self.d.keys(), self.d.values(), xs, full=True)
        ys = multivariate_normal(atleast_1d(mean), atleast_2d(cov))
        self.update(xs, ys)
        return ys
    def update(self, xs, ys):
        for x,y in zip(xs,ys):
            self.d[tuple(atleast_1d(x))] = y


### Squared Exponential covariance matrix

# $$ K_{ij} = \exp\left(-\tfrac{1}{2} \left| \mathbf{x}^{(i)} - \mathbf{y}^{(j)} \right|^2\right) $$

def K_SE(xs, ys=None, l=1, deriv=False, wrt='l'):
    l = asarray(l)
    sig = 1 #l[0]
    #l = l[1:]
    xs = ascolumn(xs)
    if ys is None:
        d = squareform(pdist(xs/l, 'sqeuclidean'))
    else:
        ys = ascolumn(ys)
        d = cdist(xs/l, ys/l, 'sqeuclidean')
    cov = exp(-d/2)
    if not deriv: return sig * cov

    grads = []
    if wrt == 'l':
        #grads.append(cov) # grad of sig
        for i in xrange(shape(xs)[1]):
            if ys is None:
                grad = sig * cov * squareform(pdist(ascolumn(xs[:,i]), 'sqeuclidean'))
            else:
                grad = sig * cov * cdist(ascolumn(xs[:,i]), ascolumn(ys[:,i]), 'sqeuclidean')
            grad /= l[i] ** 3
            grads.append(grad)
        return sig * cov, grads
    elif wrt == 'y':
        if shape(xs)[0] != 1: print '*** x not a row vector ***'
        jac = sig * cov * ((ys - xs) / l**2).T
        return sig * cov, jac

# Matern
def K_Matern(xs, ys=None, l=1):
    l = asarray(l)
    xs = ascolumn(xs)
    if ys is None:
        r = squareform(pdist(xs, 'euclidean'))
    else:
        ys = ascolumn(ys)
        r = cdist(xs, ys, 'euclidean')
    #r = sqrt(5.) * r / l
    #cov = (1 + r * (1 + r/3)) * exp(-r)
    r = sqrt(3.) * r / l
    cov = (1 + r) * exp(-r)
    return cov


### Multivariate Gaussian conditional distribution

# conditional distribution $\mathcal{N}(\boldsymbol\mu,\Sigma)$ of test points $\mathbf{t}$ given data $(\mathbf{x},\mathbf{y})$
# \begin{align}
# \boldsymbol\mu &= A \mathbf{y} \\
# \Sigma &= K_\mathbf{tt} - A K_\mathbf{xt} \\
# \text{where } A &= K_\mathbf{tx} (K_\mathbf{xx} + \sigma_\mathrm{noise}^2 I)^{-1}
# \end{align}
# [[GPML](http://gaussianprocess.org/gpml), p16]

class GaussCond(object):
    def __init__(self, xs, ys, noise=0.001, l=1, K=K_SE):
        self.xs = xs
        self.l = l
        self.K = K
        Kxx = self.K(xs, l=self.l)
        self.KxxI = pinvh(Kxx + (noise**2) * eye_like(Kxx))
        self.KxxI_ys = self.KxxI.dot(ys)
    
    def __call__(self, ts, full=False):
        Ktx = self.K(ts, self.xs, l=self.l)
        mean = Ktx.dot(self.KxxI_ys)
        if full: # full covariance matrix
            cov = self.K(ts, l=self.l) - Ktx.dot(self.KxxI).dot(Ktx.T)
            return mean, cov
        else: # predictive variance (diag cov)
            # TODO assuming diag(K(ts)) = [1,1,...]
            var = 1 - dqf(Ktx, self.KxxI)
            return squeeze(mean), squeeze(var)

def mvg_cond(xs, ys, ts, full=False, noise=0.001):
    return GaussCond(xs, ys, noise)(ts, full)


# set hyperparms by max-likelihood

def trace_prod(A,B):
    # trace(A.dot(B))
    # http://stackoverflow.com/q/18854425
    return einsum('ij,ji->', A, B)

def hpml(xs, ys, l0=1, noise=0.001, K=K_SE):
    xs = asarray(xs); ys = ascolumn(ys)
    def nll(l): # negative log likelihood
        #if l < 0.001: return 1e10
        Kxx = K(xs, l=l)
        Kxx += (noise**2) * eye_like(Kxx)
        res = (ys.T).dot(pinvh(Kxx)).dot(ys) + slogdet(Kxx)[1]
        res = squeeze(res)
        #print l,res
        return res
    def nll_prime(l):
        Kxx,Kps = K(xs, l=l, deriv=True)
        Kxx += (noise**2) * eye_like(Kxx)
        KxxI = pinvh(Kxx)
        a = KxxI.dot(ys)
        aaT = outer(a,a) # a . a.T
        KI_aaT = KxxI - aaT # K^-1 - aaT
        res = []
        for Kp in Kps:
            grad = trace_prod(KI_aaT, Kp)
            res.append(grad)
        return asarray(res)
    #l = fmin_cg(nll, l0, maxiter=10, disp=False, epsilon=.001)
    #l = fmin_cg(nll, l0, disp=False, epsilon=.001)
    l = fmin_cg(nll, l0, fprime=nll_prime, disp=False)#, maxiter=10, disp=False)
    best_nll = nll(l)
    nlls = set([int(best_nll/noise)])
    for i in xrange(20):
        cur_l0 = lognorm.rvs(1, size=size(l0))
        cur_l = fmin_cg(nll, cur_l0, fprime=nll_prime, disp=False)
        cur_nll = nll(cur_l)
        nlls.add(int(cur_nll/noise))
        if cur_nll < best_nll:
            #print 'LL up by', best_nll - cur_nll
            best_nll = cur_nll
            l = cur_l
    #print len(nlls), 'suff. uniq. LL optima:', sorted([x*noise for x in nlls])
    return absolute(l), len(nlls)

