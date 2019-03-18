import numpy as np

class flatten:
    def __init__(self,nodes,indims,solver="Basic",seed=False):
        pass
    
    def forwardprop(self,A):
        self.dims=A.shape
        return A.flatten().reshape(1,-1)
    
    def forwardprop_dropout(self,A):
        self.dims=A.shape
        return A.flatten().reshape(1,-1)
    
    def backprop(self,eta, Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def momentum_backprop(self,eta, mu,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def nesterov_forwardprop(self, A,mu):
        self.dims=A.shape
        return A.flatten().reshape(1,-1)
    
    def nesterov_forwardprop_dropout(self, A,mu,p):
        self.dims=A.shape
        return A.flatten().reshape(1,-1)

    def nesterov_backprop(self,eta, mu,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def ada_backprop(self,eta,Dnext,nesterov,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def RMS_prop(self,eta,gamma,Dnext, nesterov,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def adamoptimizer(self,eta,mu,gamma,t,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def momentum_RMS_prop(self,eta,mu,gamma,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def momentum_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def nesterov_RMS_prop(self,eta,mu,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)

    def nesterov_ada_backprop(self,eta,mu,Dnext,lam1,lam2):
        return Dnext.reshape(*self.dims)