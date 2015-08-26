import sys
from pylab import *
from gp import *

from scipy.stats import norm
from scipy.optimize import anneal


### Expected Improvement

# given current maximum $y$:
# \begin{align}
# \mathbb{E}I &= \mathbb{E} (f(\mathbf{x}) - y)^+ \\
# &= (\boldsymbol\mu - y) \Phi(\mathbf{z}) + \boldsymbol\sigma \cdot \phi(\mathbf{z}) \\
# \text{where } \mathbf{z} &= \frac{\boldsymbol\mu - y}{\boldsymbol\sigma}
# \end{align}
# [[arXiv:1012.2599](http://arxiv.org/abs/1012.2599), p13]

def EI(mean, sd, maxpt):
    mean = asarray(mean); sd = asarray(sd); maxpt = asscalar(maxpt)
    Z = (mean - maxpt) / sd
    ei = (mean - maxpt) * norm.cdf(Z) + sd * norm.pdf(Z)
    return nan_to_num(ei)

### High-Dimensional GPGO-EI

#ion()

def gpgo_ei(f, D, lower, upper, dim, maxeval=Inf, noise=0.001, target=Inf, l=1, K=K_SE):
    D = list(D); Y = list(f(D))

    numopts_ll = []; numopts_ei = []
    
    mei = Inf # maximum expected improvement
    x0 = D[-1] # last sample
    iters = 0
    while mei > noise and iters < maxeval and max(Y) < target:
    #while iters < maxeval and max(Y) < target:
        posterior = GaussCond(asarray(D), ascolumn(Y), l=l, K=K)
        maxY = max(Y)
        
        def neg_ei(x):
            if any(asarray(x) < asarray(lower)) or \
               any(asarray(upper) < asarray(x)):
                return Inf
            #print x
            mean,var = posterior(atleast_2d(x))
            ei = EI(mean, sqrt(var), maxY)
            #print 'EI', ei
            return -ei

        def neg_ei_deriv(x):
            mean,var = posterior(atleast_2d(x))
            s = asarray(sqrt(var))
            mean = asarray(mean); maxpt = asscalar(maxY)
            Z = (mean - maxpt) / s
            cov,jac = K(atleast_2d(x), D, l=l, deriv=True, wrt='y')
            #print x, D, jac, posterior.KxxI, cov.T, s
            ds = -jac.dot(posterior.KxxI).dot(cov.T) / s
            dZ = (jac.dot(posterior.KxxI).dot(ascolumn(Y)) - Z*ds) / s
            dEI = (Z*norm.cdf(Z) + norm.pdf(Z))*ds + s*norm.cdf(Z)*dZ
            dEI = squeeze(dEI.T)
            #print 'dEI', dEI
            return -dEI
        
        #x0,status = anneal(neg_ei, x0, lower=lower, upper=upper, maxeval=100)
        x0 = fmin_cg(neg_ei, x0, fprime=neg_ei_deriv, disp=False)
        mei = -neg_ei(x0)
        meis = set()
        if not isinf(mei): meis.add(int(mei/noise))
        #for xstart in D:
        for i in xrange(20):
            cur_x0 = fmin_cg(neg_ei, 10*rand(dim)-5, fprime=neg_ei_deriv, disp=False)
            cur_mei = -neg_ei(cur_x0)
            if not isinf(cur_mei): meis.add(int(cur_mei/noise))
            if cur_mei > mei:
                #print 'MEI up by', cur_mei - mei
                x0 = cur_x0
                mei = cur_mei
        #print len(meis), 'suff. uniq. EI optima:', sorted([x*noise for x in meis])
        numopts_ei.append(len(meis))
        #if mei > 10000: print posterior(atleast_2d(x0)), maxY, target
        D.append(x0); Y.append(f(atleast_2d(x0)))
        l, numopt_ll = hpml(D, Y, l, K=K) # hyperparam
        numopts_ll.append(numopt_ll)

        print 'MEI =', '%.7f' % mei, '@', x0, '\t HYPER =', absolute(squeeze(l))
        sys.stdout.flush()
        #print>>sys.stderr, '\tl =', '%5.1f' % abs(asscalar(l)), '\t \r',
        #print>>sys.stderr, '\tl =', absolute(squeeze(l)), '\t \r',
        #print>>sys.stderr, 'HYPER =', absolute(squeeze(l))
        iters += 1
    
    print '*** EI:', numopts_ei
    print '*** LL:', numopts_ll
    sys.stdout.flush()
    #plot(numopts_ei, 'r')
    #plot(numopts_ll, 'b')
    #draw()
    #print>>sys.stderr, '*** DONE ***'
    return asarray(D), asarray(Y)
