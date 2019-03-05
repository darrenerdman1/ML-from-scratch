import numpy as np

class tanh:
    def __init__(self,nodes,indims,solver="Basic",scaleweights=True,seed=False):
        self.solver=solver.lower()
        self.nodes=nodes
        self.indims=indims
        self.scaleweights=scaleweights
        self.seed=seed
        self.weights=np.array([])
        self.biases=np.array([])

    def weightinit(self):
        if self.scaleweights:
            self.weights=np.random.randn(self.indims,self.nodes)*np.sqrt(2/(self.indims+self.nodes))
            self.biases=np.random.randn(1,self.nodes)*np.sqrt(2/(self.indims+self.nodes))
        else:
            self.weights=np.random.randn(self.indims,self.nodes)
            self.biases=np.random.randn(1,self.nodes)
        if self.solver=="momentum" or self.solver=="nesterov":
            self.delw=np.zeros((self.indims,self.nodes))
            self.delb=np.zeros((1,self.nodes))
        elif self.solver == "adagrad" or self.solver=="rmsprop":
            self.Gw=np.ones((self.indims,self.nodes))
            self.Gb=np.ones((1,self.nodes))
        elif self.solver=='adam':
            self.mw=np.zeros((self.indims,self.nodes))
            self.mb=np.zeros((1,self.nodes))
            self.vw=np.zeros((self.indims,self.nodes))
            self.vb=np.zeros((1,self.nodes))

    def forwardprop(self, A):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        self.H=self.A@self.weights+self.biases
        self.Z=np.tanh(self.H)
        return self.Z

    def backprop(self,eta, Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.weights-=eta*(self.A.T@(derivterm))
        self.biases-=eta*np.sum(derivterm, axis=0)
        return D

    def momentum_backprop(self,eta, mu,Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.delw=mu*self.delw-eta*(self.A.T@(derivterm))
        self.delb=mu*self.delb-eta*np.sum(derivterm, axis=0)
        self.weights+=self.delw
        self.biases+=self.delb
        return D

    def nesterov_forwardprop(self, A,mu):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        self.weights+=mu*self.delw
        self.biases+=mu*self.delb
        self.H=self.A@self.weights+self.biases
        self.Z=np.tanh(self.H)
        return self.Z

    def nesterov_backprop(self,eta, mu,Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.weights-=mu*self.delw
        self.biases-=mu*self.delb
        self.delw=mu*self.delw-eta*(self.A.T@(derivterm))
        self.delb=mu*self.delb-eta*np.sum(derivterm, axis=0)
        self.weights+=self.delw
        self.biases+=self.delb
        return D

    def ada_backprop(self,eta,Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.Gw=self.Gw+(self.A.T@(derivterm))**2
        self.Gb=self.Gb+(np.sum(derivterm, axis=0))**2
        ada_etaw=eta/np.sqrt(self.Gw+1e-9)
        ada_etab=eta/np.sqrt(self.Gb+1e-9)
        self.weights-=ada_etaw*(self.A.T@(derivterm))
        self.biases-=ada_etab*np.sum(derivterm, axis=0)
        return D

    def RMS_prop(self,eta,gamma,Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.Gw=gamma*self.Gw+(1-gamma)*(self.A.T@(derivterm))**2
        self.Gb=gamma*self.Gb+(1-gamma)*(np.sum(derivterm, axis=0))**2
        rms_etaw=eta/np.sqrt(self.Gw+1e-9)
        rms_etab=eta/np.sqrt(self.Gb+1e-9)
        self.weights-=rms_etaw*(self.A.T@(derivterm))
        self.biases-=rms_etab*np.sum(derivterm, axis=0)
        return D

    def adamoptimizer(self,eta,mu,gamma,t,Dnext):
        derivterm=Dnext*(1-(self.Z)**2)
        D=derivterm@self.weights.T
        self.mw=(mu*self.mw+(1-mu)*(self.A.T@(derivterm)))/(1+mu**t)
        self.mb=(mu*self.mb+(1-mu)*(np.sum(derivterm, axis=0)))/(1+mu**t)
        self.vw=(gamma*self.vw+(1-gamma)*(self.A.T@(derivterm))**2)/(1+gamma**t)
        self.vb=(gamma*self.vb+(1-gamma)*(np.sum(derivterm, axis=0))**2)/(1+gamma**t)
        adam_etaw=eta/np.sqrt(self.vw+1e-9)
        adam_etab=eta/np.sqrt(self.vb+1e-9)
        self.weights-=adam_etaw*self.mw
        self.biases-=adam_etab*self.mb
        return D
