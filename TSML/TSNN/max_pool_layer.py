import numpy as np

class maxpool:
    def __init__(self,nodes,indims,solver="Basic",seed=False):
        pass
    
    def forwardprop(self,A):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])
    
        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z

    def forwardprop_dropout(self,A,p):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])
    
        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z
    
    def backprop(self,eta, Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def momentum_backprop(self,eta, mu,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def nesterov_forwardprop(self, A,mu):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])
    
        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z

    def nesterov_forwardprop_dropout(self, A,mu,p):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])
    
        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z


    def nesterov_backprop(self,eta, mu,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def ada_backprop(self,eta,Dnext,nesterov,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def RMS_prop(self,eta,gamma,Dnext, nesterov,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def adamoptimizer(self,eta,mu,gamma,t,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def momentum_RMS_prop(self,eta,mu,gamma,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def momentum_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def nesterov_RMS_prop(self,eta,mu,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D

    def nesterov_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        o0=int(Dnext.shape[0]*2)
        o1=int(Dnext.shape[1])
    
        D = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                D[i:i+2,j][self.locations[cX,j].astype(int)] = Dnext[cX,j]
            cX+=1
        return D
