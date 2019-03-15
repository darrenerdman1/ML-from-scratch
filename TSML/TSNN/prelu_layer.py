import numpy as np

class pReLU:
    def __init__(self,nodes,indims,solver="Basic", seed=False):
        self.solver=solver.lower()
        self.nodes=nodes
        self.indims=indims
        self.seed=seed
        self.weights=np.array([])
        self.biases=np.array([])
        self.p=np.array([])
        self.momentum_solvers=['momentum','nesterov','nesterovrms','nesterovada','momentumada','momentumrms']
        self.eta_solvers=["adagrad","rmsprop",'nesterovrms','nesterovada','momentumada','momentumrms']

    def Activation(self,p,Z):
        return Z*(Z>0)+Z*(Z<=0)*p

    def Act_derivativeP(self,p,Z):
        return (Z*(Z<=0))

    def Act_derivativeX(self,p,Z):
        return (Z>0+(Z<=0)*p)

    def weightinit(self,seed=False):
        if seed!=False:
            np.random.seed(seed)

        self.weights=np.random.randn(self.indims,self.nodes)*np.sqrt(2/(self.indims+self.nodes))
        self.biases=np.random.randn(1,self.nodes)*np.sqrt(2/(self.indims+self.nodes))
        self.p=np.random.randn(1,self.nodes)*np.sqrt(2/(self.indims+self.nodes))

        if self.solver in self.momentum_solvers:
            self.delw=np.zeros((self.indims,self.nodes))
            self.delb=np.zeros((1,self.nodes))
            self.delp=np.zeros((1,self.nodes))

        if self.solver in self.eta_solvers:
            self.Gw=np.ones((self.indims,self.nodes))
            self.Gb=np.ones((1,self.nodes))
            self.Gp=np.ones((1,self.nodes))

        if self.solver=='adam':
            self.mw=np.zeros((self.indims,self.nodes))
            self.mb=np.zeros((1,self.nodes))
            self.mp=np.zeros((1,self.nodes))
            self.vw=np.zeros((self.indims,self.nodes))
            self.vb=np.zeros((1,self.nodes))
            self.vp=np.zeros((1,self.nodes))

    def forwardprop(self, A):
        self.A=A
        if not self.weights.size:
            self.weightinit()
        self.H=self.A@self.weights+self.biases
        return self.Activation(self.p,self.H)

    def forwardprop_dropout(self, A,p):
        self.M=np.random.rand(*A.shape)<p
        self.A=A*self.M/p
        if not self.weights.size:
            self.weightinit()
        self.H=self.A@self.weights+self.biases
        return self.Activation(self.p,self.H)


    def backprop(self,eta, Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        self.weights -= eta*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.biases -= eta*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.p -= eta*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        return D

    def momentum_backprop(self,eta, mu,Dnext,lam1,lam2):
        derivterm = Dnext*self.Act_derivativeX(self.p,self.H)
        D = derivterm@self.weights.T
        self.delw = mu*self.delw-eta*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb = mu*self.delb-eta*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp = mu*self.delp-eta*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        self.weights += self.delw
        self.biases += self.delb
        self.p+=self.delp
        return D

    def nesterov_forwardprop(self, A,mu):
        self.A = A
        if not self.weights.size:
            self.weightinit()
        self.weights += mu*self.delw
        self.biases += mu*self.delb
        self.p += mu*self.delp
        self.H = self.A@self.weights+self.biases
        return self.Activation(self.p,self.H)

    def nesterov_forwardprop_dropout(self, A,mu,p):
        if not self.M.size:
            self.M=np.random.rand(*A.shape)<p
        self.A=A*self.M/p
        if not self.weights.size:
            self.weightinit()
        self.weights+=mu*self.delw
        self.biases+=mu*self.delb
        self.H=self.A@self.weights+self.biases
        return self.Activation(self.p,self.H)

    def nesterov_backprop(self,eta, mu,Dnext,lam1,lam2):
        derivterm = Dnext*self.Act_derivativeX(self.p,self.H)
        D = derivterm@self.weights.T
        dw=self.delw
        db=self.delb
        dp=self.delp
        self.delw = mu*self.delw-eta*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb = mu*self.delb-eta*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp = mu*self.delp-eta*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)

        self.weights -= mu*dw
        self.biases -= mu*db
        self.p -= mu*dp
        self.weights += self.delw
        self.biases += self.delb
        self.p+=self.delp
        return D

    def ada_backprop(self,eta,Dnext,nesterov,lam1,lam2):
        derivterm = Dnext*self.Act_derivativeX(self.p,self.H)
        D = derivterm@self.weights.T
        self.Gw = self.Gw+(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)**2
        self.Gb = self.Gb+(np.sum(derivterm, axis=0))**2
        self.Gp = self.Gp + (np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0))**2
        ada_etaw = eta/np.sqrt(self.Gw+1e-9)
        ada_etab = eta/np.sqrt(self.Gb+1e-9)
        ada_etap = eta/np.sqrt(self.Gp+1e-9)
        if not nesterov:
            self.weights -= ada_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
            self.biases -= ada_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
            self.p -= ada_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        return D

    def RMS_prop(self,eta,gamma,Dnext,nesterov,lam1,lam2):
        derivterm = Dnext*self.Act_derivativeX(self.p,self.H)
        D = derivterm@self.weights.T
        self.Gw = gamma*self.Gw+(1-gamma)*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)**2
        self.Gb = gamma*self.Gb+(1-gamma)*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)**2
        self.Gp = gamma*self.Gp +(1-gamma)*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)**2
        rms_etaw = eta/np.sqrt(self.Gw+1e-9)
        rms_etab = eta/np.sqrt(self.Gb+1e-9)
        rms_etap = eta/np.sqrt(self.Gp+1e-9)
        if not nesterov:
            self.weights -= rms_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
            self.biases -= rms_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
            self.p -= rms_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        return D

    def adamoptimizer(self,eta,mu,gamma,t,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        self.mw=(mu*self.mw+(1-mu)*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights))/(1+mu**t)
        self.mb=(mu*self.mb+(1-mu)*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases))/(1+mu**t)
        self.mp = (mu*self.mp +(1-mu)*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p))/(1+mu**t)
        self.vw=(gamma*self.vw+(1-gamma)*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)**2)/(1+gamma**t)
        self.vb=(gamma*self.vb+(1-gamma)*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)**2)/(1+gamma**t)
        self.vp = (gamma*self.vp +(1-gamma)*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)**2)/(1+gamma**t)
        adam_etaw=eta/np.sqrt(self.vw+1e-9)
        adam_etab=eta/np.sqrt(self.vb+1e-9)
        adam_etap=eta/np.sqrt(self.vp+1e-9)
        self.weights-=adam_etaw*self.mw
        self.biases-=adam_etab*self.mb
        self.p-=adam_etap*self.mp
        return D


    def momentum_RMS_prop(self,eta,mu,gamma,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        self.Gw=(gamma*self.Gw+(1-gamma)*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)**2)
        self.Gb=(gamma*self.Gb+(1-gamma)*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)**2)
        self.Gp = gamma*self.Gp +(1-gamma)*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)**2
        rms_etaw=eta/np.sqrt(self.Gw+1e-9)
        rms_etab=eta/np.sqrt(self.Gb+1e-9)
        rms_etap = eta/np.sqrt(self.Gp+1e-9)
        self.delw=mu*self.delw-rms_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb=mu*self.delb-rms_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp=mu*self.delp-rms_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        self.weights+=self.delw
        self.biases+=self.delb
        self.p+=self.delp
        return D

    def momentum_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        self.Gw=(self.Gw+(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)**2)
        self.Gb=(self.Gb+(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)**2)
        self.Gp = gamma*self.Gp +(1-gamma)*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)**2
        ada_etaw=eta/np.sqrt(self.Gw+1e-9)
        ada_etab=eta/np.sqrt(self.Gb+1e-9)
        ada_etap=eta/np.sqrt(self.Gp+1e-9)
        self.delw=mu*self.delw-ada_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb=mu*self.delb-ada_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp=mu*self.delp-ada_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        self.weights+=self.delw
        self.biases+=self.delb
        self.p+=self.delp
        return D

    def nesterov_RMS_prop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        dw=self.delw
        db=self.delb
        dp=self.delp
        rms_etaw=eta/np.sqrt(self.Gw+1e-9)
        rms_etab=eta/np.sqrt(self.Gb+1e-9)
        rms_etap = eta/np.sqrt(self.Gp+1e-9)
        self.delw=mu*self.delw-rms_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb=mu*self.delb-rms_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp=mu*self.delp-rms_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        self.weights-=mu*dw
        self.biases-=mu*db
        self.p-=mu*dp
        self.weights+=self.delw
        self.biases+=self.delb
        self.p+=self.delp
        return D

    def nesterov_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        derivterm=Dnext*self.Act_derivativeX(self.p,self.H)
        D=derivterm@self.weights.T
        dw=self.delw
        db=self.delb
        dp=self.delp
        ada_etaw=eta/np.sqrt(self.Gw+1e-9)
        ada_etab=eta/np.sqrt(self.Gb+1e-9)
        ada_etap=eta/np.sqrt(self.Gp+1e-9)
        self.delw=mu*self.delw-ada_etaw*(self.A.T@(derivterm)+lam1*np.sign(self.weights)+lam2*self.weights)
        self.delb=mu*self.delb-ada_etab*(np.sum(derivterm, axis=0)+lam1*np.sign(self.biases)+lam2*self.biases)
        self.delp=mu*self.delp-ada_etap*(np.sum(Dnext*self.Act_derivativeP(self.p,self.H), axis=0)+lam1*np.sign(self.p)+lam2*self.p)
        self.weights-=mu*dw
        self.biases-=mu*db
        self.p-=mu*dp
        self.weights+=self.delw
        self.biases+=self.delb
        self.p+=self.delp
        return D
