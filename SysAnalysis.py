import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

class DynamicM:
    def __init__(self,a1,a2,b,q,t,k0,x0,k):
        self.changeData(a1,a2,b,q,t,k0,x0,k)

    def changeData(self,a1,a2,b,q,t,k0,x0,k):
        self.a1=a1
        self.a2=a2
        self.b=b
        self.q=q
        self.T0=t
        self.k0=k0
        self.k=k
        self.x0 = np.matrix([[0],[0],[0]])
        self.t_range = np.arange(0,300 + self.T0, self.T0)
        self.A = np.matrix([[0,1,0],[0,0,1],[-1,-self.a1,-self.a2]])
        self.B = np.matrix([[0],[0],[self.b]])
        self.C = np.matrix([1,0,0])

    def CalcY(self):
        u =4
        F = np.identity(3)
        for index in range(1, q+1):
            F +=np.linalg.matrix_power(self.A * T0,index)/float(math.factorial(index))
        G = np.dot(np.dot(F-np.identity(3),np.linalg.inv(self.A)),B)
        x_next = np.dot(F,self.x0) + (G*u)
        vector_arr1 = []
        vector_arr2 = []
        vector_arr3 = []

        X_arr1 = []
        Y_arr1 = []
        X_arr2 = []
        Y_arr2 = []
        X_arr3 = []
        Y_arr3 = []
        file = open('output.txt','a')
        print("success")
        sys.stdout = file
        plt.xlabel('t - time')
        plt.ylabel('y(t) - output process')
        for n in range(1,self.k):
            x_prev = x_next
            x_next = np.dot(F,x_prev)+(G*u)
            Y_arr1.append(float(np.dot(self.C,x_next)))
            vector_arr1.append(x_next)
            X_arr1.append(n*self.T0)
        self.y1 = Y_arr1
        plt.plot(X_arr1, Y_arr1)
        x_next = np.dot(F,self.x0) + (G*u)
        for n in range(1,self.k):
            if (n>=int(self.k/2)):
                u=-1
            x_prev = x_next
            x_next = np.dot(F,x_prev)+(G*u)
            Y_arr2.append(float(np.dot(self.C,x_next)))
            vector_arr2.append(x_next)
            X_arr2.append(n*self.T0)
        self.y2 = Y_arr2
        plt.plot(X_arr2, Y_arr2)
        x_next = np.dot(F,self.x0) + (G*u)
        for n in range(1,self.k):
            if (n<int(self.k*1/3)):
                u=1
            if (n>=int(self.k/3) and n<int(self.k*2/3)):
                u=-1
            if (n>=int(self.k*2/3)):
                u = 1
            x_prev = x_next
            x_next = np.dot(F,x_prev)+(G*u)
            Y_arr3.append(float(np.dot(self.C,x_next)))
            vector_arr3.append(x_next)
            X_arr3.append(n*self.T0)
        self.y3 = Y_arr3
        plt.plot(X_arr3, Y_arr3)
        plt.show()
        draw(vector_arr1)
        draw(vector_arr2)
        draw(vector_arr3)

        print("First observation: \n X: ",X_arr1,"\n Y:",Y_arr1, "\n")
        print("Second observation: \n X: ",X_arr2,"\n Y:",Y_arr2, "\n")
        print("Third observation: \n X: ",X_arr3,"\n Y:",Y_arr3, "\n")
        file.close()

def draw(vector_arr):
        first = []
        second = []
        third = []
        x_coord =[0 for x in range(9999)]
        i = 0
        for n in range(9999):
            x_coord[n] = i
            i+=0.02               

        for x in vector_arr:
            first.append(float(x[0]))
            second.append(float(x[1]))
            third.append(float(x[2]))
        plt.plot(x_coord, first,label = 'x1')
        plt.plot(x_coord, second,label = 'x2')
        plt.plot(x_coord, third,label = 'x3')
        plt.show()
            
n=3
m=1
l=1
a1 = 3 #random.randrange(1,11,1)
a2 = 1 #random.randrange(1,11,1)
b = 1
A = np.matrix([[0,1,0],[0,0,1],[-1,-a1,-a2]])
B = np.matrix([[0],[0],[b]])
C = np.matrix([1,0,0])
Trand = random.randrange(1, 1000,1)
T0 = 0.02 #(1/float(Trand))
q = 10 #random.randrange(2,11,1)
model = DynamicM(a1,a2,b,q,T0,0,0,10000)
model.CalcY()
