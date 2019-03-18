import numpy as np
from scipy.signal import correlate

class convolution:
    def __init__(self,filters,filtersize,indims,solver="Basic",seed=False):
        self.solver=solver.lower()
        self.filters=filters
        self.filtersize=filtersize
        self.indims=indims
        self.seed=seed
        self.weights=np.array([])
        self.biases=np.array([])
        self.momentum_solvers=['momentum','nesterov','nesterovrms','nesterovada','momentumada','momentumrms']
        self.eta_solvers=["adagrad","rmsprop",'nesterovrms','nesterovada','momentumada','momentumrms']


    def Activation(self,Z):
        return Z*(Z>0)

    def Act_derivative(self,Z):
        return Z>0

    def weightinit(self,seed=False):
        if seed!=False:
            np.random.seed(seed)

        self.weights=np.random.randn(self.filtersize,self.indims,self.filters)*np.sqrt(2/(self.indims+self.filters))
        self.biases=np.random.randn(1,self.filters)*np.sqrt(2/(self.indims+self.filters))


        if self.solver in self.momentum_solvers:
            self.delw=np.zeros((self.filtersize,self.indims,self.filters))
            self.delb=np.zeros((1,self.filters))

        if self.solver in self.eta_solvers:
            self.Gw=np.ones((self.filtersize,self.indims,self.filters))
            self.Gb=np.ones((1,self.filters))

        if self.solver=='adam':
            self.mw=np.zeros((self.filtersize,self.indims,self.filters))
            self.mb=np.zeros((1,self.filters))
            self.vw=np.zeros((self.filtersize,self.indims,self.filters))
            self.vb=np.zeros((1,self.filters))

    def forwardprop(self, A):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        for k in range(self.filters):
            self.H[:,k]=correlate(padA,self.weights[:,:,k], mode="valid").reshape(-1)
        self.H+=self.biases
        self.Z=self.Activation(self.H)
        return self.Z

    def forwardprop_dropout(self,A,p):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        for k in range(self.filters):
            self.H[:,k]=correlate(padA,self.weights[:,:,k], mode="valid").reshape(-1)
        self.H+=self.biases
        self.Z=self.Activation(self.H)
        return self.Z

    def backprop(self,eta, Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def momentum_backprop(self,eta, mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def nesterov_forwardprop(self, A,mu):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        print(self.H.shape)
        print(self.weights.shape[0])
        print(padA.shape)
        for k in range(self.filters):
            self.H[:,k]=correlate(padA,self.weights[:,:,k], mode="valid").reshape(-1)
        self.H+=self.biases
        self.Z=self.Activation(self.H)
        return self.Z
    
    def nesterov_forwardprop_dropout(self, A,mu,p):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        print(self.H.shape)
        print(self.weights.shape[0])
        print(padA.shape)
        for k in range(self.filters):
            self.H[:,k]=correlate(padA,self.weights[:,:,k], mode="valid").reshape(-1)
        self.H+=self.biases
        self.Z=self.Activation(self.H)
        return self.Z

    def nesterov_backprop(self,eta, mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def ada_backprop(self,eta,Dnext,nesterov,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def RMS_prop(self,eta,gamma,Dnext, nesterov,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def adamoptimizer(self,eta,mu,gamma,t,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.mw[:,k,l]=(mu*self.mw[:,k,l]+(1-mu)*(correlate(self.A[:,k],derivterm[:,l], mode="valid")))/(1+mu**t)
                self.vw[:,k,l]=(gamma*self.vw[:,k,l]+(1-gamma)*(correlate(self.A[:,k],derivterm[:,l], mode="valid"))**2)/(1+gamma**t)
  
        self.mb=(mu*self.mb+(1-mu)*(np.sum(derivterm, axis=0)))/(1+mu**t)
        self.vb=(gamma*self.vb+(1-gamma)*(np.sum(derivterm, axis=0))**2)/(1+gamma**t)
        self.biases-=eta*(np.sum(derivterm, axis=0))
        adam_etaw=eta/np.sqrt(self.vw+1e-9)
        adam_etab=eta/np.sqrt(self.vb+1e-9)
        self.weights-=adam_etaw*self.mw
        self.biases-=adam_etab*self.mb
        return D

    def momentum_RMS_prop(self,eta,mu,gamma,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def momentum_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def nesterov_RMS_prop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D

    def nesterov_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivative(self.H)
        D=np.empty((Dnext.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            D[:,k]=correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=eta*correlate(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=eta*(np.sum(derivterm, axis=0))
        return D