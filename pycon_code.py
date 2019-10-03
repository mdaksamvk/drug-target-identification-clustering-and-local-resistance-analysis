# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:13:53 2019

@author: Md Aksam
"""

from numba import cuda
print(cuda.gpus)

from __future__ import division
from numba import cuda
import numpy as np
import numpy as np
import os
from numba import guvectorize, float32, cuda, int64
import numba
import numpy as np
from time import time
from contextlib import contextmanager
from math import sin
import pkg_resources, os

os.chdir("C:\\Users\\Md Aksam\\Downloads\\drug-target-identification-clustering-and-local-resistance-analysis-master\\drug-target-identification-clustering-and-local-resistance-analysis-master")
inp=np.genfromtxt('input.csv',delimiter=',')
n1=np.genfromtxt('matrix.csv',delimiter=',')

# set up input data
rows = 58   # shape[0]
cols = 6   # shape[1]
one = np.ones(rows*cols, dtype=np.int64).reshape(rows, cols)
for i in range(inp.shape[0]):
    for j in range(inp.shape[1]):
        one[i,j] = inp[i,j]

@guvectorize(['void(int64[:], int64[:])'],'(n)->(n)', target="cuda")       
def test(x1,c1):
    #x1=x1[:2]
    #for i in x1:
    i=0
    e1=x1[i]
    e2=x1[i+1]
    x2=n1[e1]
    y2=n1[e2];
#return x2,y2
#a=func(x2,y2)
    temp1,temp2,temp3,temp4,temp5,temp6=0,0,0,0,0,0;
    for i in range(0,7):
        if x2[i]==1 and y2[i]==1:
            temp1=temp1+1;
        elif x2[i]==1 and y2[i]==0:
            temp2=temp2+1;
        elif x2[i]==0 and y2[i]==1:
            temp3=temp3+1;
    for i in range(8,82):
        if x2[i]==1 and y2[i]==1:
            temp4=temp4+1;
        elif x2[i]==1 and y2[i]==0:
            temp5=temp5+1;
        elif x2[i]==0 and y2[i]==1:
            temp6=temp6+1;
    c1[0]=int64(temp1)
    c1[1]=int64(temp2)
    c1[2]=int64(temp3)
    c1[3]=int64(temp4)
    c1[4]=int64(temp5)
    c1[5]=int64(temp6)
    
    a=[temp1,temp2,temp3,temp4,temp5,temp6]
    for i in range(c1.shape[0]):
#       for j in range(0,5):
        c1[i] = a[i]
    return a

# Copy the arrays to the device
n1= cuda.to_device(n1)
dev_inp = cuda.to_device(one)             # alloc and copy input data

test(dev_inp,dev_inp)             # invoke the gufunc

dev_inp.copy_to_host(one)
# Copy the result back to the host
C = dev_inp.copy_to_host()


#thread and parallel
def test(x1,c1):
    i=0
    e1=x1[i]
    e2=x1[i+1]
    x2=n1[e1]
    y2=n1[e2];
    temp1,temp2,temp3,temp4,temp5,temp6=0,0,0,0,0,0;
    for i in range(0,7):
        if x2[i]==1 and y2[i]==1:
            temp1=temp1+1;
        elif x2[i]==1 and y2[i]==0:
            temp2=temp2+1;
        elif x2[i]==0 and y2[i]==1:
            temp3=temp3+1;
    for i in range(8,82):
        if x2[i]==1 and y2[i]==1:
            temp4=temp4+1;
        elif x2[i]==1 and y2[i]==0:
            temp5=temp5+1;
        elif x2[i]==0 and y2[i]==1:
            temp6=temp6+1;
    c1[0]=int64(temp1)
    c1[1]=int64(temp2)
    c1[2]=int64(temp3)
    c1[3]=int64(temp4)
    c1[4]=int64(temp5)
    c1[5]=int64(temp6)
    
    a=[temp1,temp2,temp3,temp4,temp5,temp6]
    for i in range(c1.shape[0]):
#       for j in range(0,5):
        c1[i] = a[i]
    return a

#timer function
@contextmanager
def timer(name=""):
    ts = time()
    yield
    te = time()
    print("Elapsed time for %s: %s" % (name, te - ts))
    
#single thread, multi thread and GPU  
gus1 = guvectorize(['void(int64[:], int64[:])'],'(n)->(n)')
gup1 = guvectorize(['void(int64[:], int64[:])'],'(n)->(n)', target="parallel")
guc1 = guvectorize(['void(int64[:], int64[:])'],'(n)->(n)', target="cuda") 

gvecs1 = gus1(test)
gvecp1 = gup1(test)
gvecc1 = guc1(test)

with timer("single-thread, scalar y"):
    s0d = gvecs1(one)
    
with timer("multi-thread, scalar y"):
    p0d = gvecp1(one)
    
with timer("GPU, scalar y"):
    c0d = gvecc1(one)
