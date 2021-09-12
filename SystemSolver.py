import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt
import random as rd

def model(z,t,u):
    x = z[0]
    y = z[1]
    dxdt = 0.8*x*(1-x-y)+u
    dydt = 0.4*y*(1-x-y) + 0.05*np.sin(0.5*t)
    dzdt = [dxdt,dydt]
    return dzdt

def model_with_error(z,t,u):
    x = z[0] + rd.uniform(-0.02,0.02)
    y = z[1]
    dxdt = 0.8*x*(1-x-y)+u
    dydt = 0.4*y*(1-x-y) + 0.05*np.sin(0.5*t)
    dzdt = [dxdt,dydt]
    return dzdt

def model_opt(z,t,u):
    x = z[0] 
    y = z[1]
    dxdt = 0.8*x*(1-x-y)+u
    dydt = 0.4*y*(1-x-y) + 0.05*np.sin(0.5*t)
    dzdt = [dxdt,dydt]
    return dzdt

z0 = [0.4,0.3]

n = 100

t = np.linspace(0,5,n)

u = np.zeros(n)
u_opt = np.zeros(n)
u[21:30] = 0.1
u[60:70]=0.2

x = np.empty_like(t)
y = np.empty_like(t)

x[0] = z0[0]
y[0] = z0[1]

for i in range(1,n):
    tspan = [t[i-1],t[i]]
    z = odeint(model,z0,tspan,args=(u[i],))
    x[i] = z[1][0]
    y[i] = z[1][1]
    z0 = z[1] 

z0_e = [0.4,0.3]
x_e =np.empty_like(t)
y_e = np.empty_like(t)
x_e[0] = z0_e[0]
y_e[0] = z0_e[1]

def chi(t_0,t_k,t):
    if t>=t_0 and t<=t_k:
        return 1
    else:
        return 0

def integrand(s,tau,t):
    return chi(tau,t,s)*(1-y[s]-2*x[s])

def beta(tau,t):
    integral = quad(integrand,0,100,args=(tau,t))
    return 2.7**integral

for i in range(1,n):
    tspan = [t[i-1],t[i]]
    z_e = odeint(model_with_error,z0_e,tspan,args=(u[i],))
    x_e[i] = z_e[1][0]
    y_e[i] = z_e[1][1]
    z0_e = z_e[1] 

from scipy import interpolate

def f(x):
    x_points = [0,0.25,0.5,0.75,1.
    ,1.25,1.5,1.75,2.
    ,2.25,2.5,2.75,3.
    ,3.25,3.5,3.75,4.
    ,4.25,4.5,4.75,5.]
    y_points = [0.01,0.005,0.007,0.02,0.035,
    0.07,0.04,0.01,0.005,
    0.003,0.005,0.03,0.075,
    0.15,0.1,0.03,0.01,
    0.0033,0.001,0.0001,0.00001]
    tck = interpolate.splrep(x_points,y_points)
    return interpolate.splev(x,tck)
u_opt = f(t)

z0_opt = [0.4,0.3]
x_opt =np.empty_like(t)
y_opt = np.empty_like(t)
x_opt[0] = z0_opt[0]
y_opt[0] = z0_opt[1]

for i in range(1,n):
    tspan = [t[i-1],t[i]]
    z_opt = odeint(model_opt,z0_opt,tspan,args=(u_opt[i],))
    x_opt[i] = z_opt[1][0]
    y_opt[i] = z_opt[1][1]
    z0_opt = z_opt[1] 


plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,u_opt,'g-', label='u_opt(t)')
plt.plot(t,x,'b:',label='x_1(t)')
plt.plot(t,x_e,'b--',label = 'y(t)')
plt.plot(t,x_opt,'b-', label='x_1_opt(t)')
plt.plot(t,y,'r:',label='x_2(t)')
plt.plot(t,y_e,'r--', label = 'y_2(t)')
plt.plot(t,y_opt,'r-',label='x_2_opt')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()