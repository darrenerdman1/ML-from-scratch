import numpy as np

class SPP:
    def __init__(self,nodes,indims,solver="Basic",seed=False):
        pass
    
    def forwardprop(self,A):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1
    
        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))
        return np.vstack((out4,out2,out1))    
    
    def forward_dropoutprop(self,A):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1
    
        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))

        return np.vstack((out4,out2,out1))
    
    
    
    def backprop(self,eta, Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    
    def momentum_backprop(self,eta, mu,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def nesterov_forwardprop(self, A,mu):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1
    
        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))

        return np.vstack((out4,out2,out1))

    def nesterov_forwardprop_dropout(self, A,mu,p):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1
    
        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))

        return np.vstack((out4,out2,out1))


    def nesterov_backprop(self,eta, mu,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def ada_backprop(self,eta,Dnext,nesterov,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def RMS_prop(self,eta,gamma,Dnext, nesterov,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def adamoptimizer(self,eta,mu,gamma,t,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def momentum_RMS_prop(self,eta,mu,gamma,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def momentum_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def nesterov_RMS_prop(self,eta,mu,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    

    def nesterov_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        D = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                D[i:i+s,j][self.locations[cX,j]] = Dnext[cX,j]
            cX+=1
        return D
    
