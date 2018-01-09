import numpy as np 
n=np.genfromtxt('matrix1.csv',delimiter=',') 
import numpy as ap 
a=ap.genfromtxt('inmat.csv',delimiter=' ')
import numpy as bp 
b=bp.genfromtxt('outmat.csv',delimiter=' ')  
k=n[:,0:7] 
k1=n[:,8:83] 
p=[sum(k[i]) for i in range(83)] 
p1=[sum(k1[i]) for i in range(83)] 
cluster1=[] 
cluster2=[] 
cluster3=[] 
cluster4=[]
for i in range(len(p)): 
 if p1[i]<=5*p[i] and a[i]<=3 and b[i]<=3: 
  cluster1.append(i) 
 elif p1[i]<=6*p[i] and a[i]<=3 and b[i]<=3: 
  cluster2.append(i) 
 elif p1[i]<=7*p[i] and a[i]<=3 and b[i]<=3: 
  cluster3.append(i) 
 else: 
   cluster4.append(i) 
len(cluster1) 
len(cluster2)  
len(cluster3) 
len(cluster4)
print("cluster1")
cluster1
print("cluster2")
cluster2
print("cluster3")
cluster3
print("cluster4")
cluster4