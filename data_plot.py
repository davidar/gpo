from pylab import *
import random

def jitter(d, fmt='ko'):
    n = len(d)
    x = range(n) + .25*randn(n) + 1
    y = asarray(d) + .25*randn(n)
    plot(x, y, fmt, alpha=.5)

def shlist(dat):
	# f1-5, f6-9, f10-14, f15-19, f20-24
	res = [(d,'ko') for ds in dat[0:5] for d in ds] + \
          [(d,'ro') for ds in dat[5:9] for d in ds] + \
          [(d,'go') for ds in dat[9:14] for d in ds] + \
          [(d,'bo') for ds in dat[14:19] for d in ds] + \
          [(d,'yo') for ds in dat[19:24] for d in ds]
	random.shuffle(res)
	return res
