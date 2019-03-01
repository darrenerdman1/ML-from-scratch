

import numpy as np


class LinearRegression:
    def __init__(self,indims,outdims,solver="Gradient Descent"):
        self.indims=indims
        self.outdims=outdims
        self.solver=solver
        self.weights=np.array([])
        self.Xmins=np.array([])
        self.trained=False
        if solver !="Gradient Descent" and solver !="Closed Form":
            print("Not a valid solver. Must be 'Gradient Descent' or 'Closed Form'")
            
    def cost(self,y,y_hat):
        return ((y_hat-y).T@(y_hat-y))/len(y)
    
    def frob_cost(self,y,y_hat):
        return np.trace((y_hat-y).T@(y_hat-y))/len(y)

    def weightinit(self):
        self.weights=np.random.randn(self.indims+1,self.outdims)
            
    def predict(self, X,normalize=True):
        b = np.sort(X,axis=0)
        if not np.where((b[1:] != b[:-1]).sum(axis=0)+1<20):
            print("At least one of your features has less than 20 unique "\
                  +"values. Perhaps try one hot encoding.")

        elif self.solver=="Gradient Descent" and normalize:
            X=self.norm(X,self.Xmins,self.Xmaxes)
        if X.shape[1] ==self.indims:
            X=np.hstack((np.ones((X.shape[0],1)),X))
            
        if not self.weights.size:
            self.weightinit()
            
        self.normpredictions=X@self.weights
        if self.solver=="Gradient Descent":
            self.predictions=self.unnorm(self.normpredictions,self.ymins,self.ymaxes)
        else:
            self.predictions=self.normpredictions

        
    def norm(self,X,Xmin,Xmax):
        return (X-Xmin)/(Xmax-Xmin)
    
    def unnorm(self,X,Xmin,Xmax):
        return np.multiply(X,Xmax-Xmin)+Xmin
            
    def train(self,X,y,epochs,eta,X_val=np.array([]),y_val=np.array([]),lam1=0,lam2=0):
        self.trained=False
        self.valerror=[]
        self.error=[]
        
        if self.solver=="Gradient Descent":
            self.Xmaxes=X.max(0)
            self.Xmins=X.min(0)
            X=self.norm(X,self.Xmins,self.Xmaxes)
            X_val=self.norm(X_val,self.Xmins,self.Xmaxes)
            X=np.hstack((np.ones((X.shape[0],1)),X))
            X_val=np.hstack((np.ones((X_val.shape[0],1)),X_val))
            self.ymaxes=y.max(0)
            self.ymins=y.min(0)
            y=self.norm(y,self.ymins,self.ymaxes)
            if y_val.size:
                y_val=self.norm(y_val,self.ymins,self.ymaxes)
            self.predict(X, False)

             
            for i in range(epochs):
                self.weights=self.weights-eta*(X.T@(self.normpredictions-y))-lam1*self.weights-lam2*np.sign(self.weights)
                
                if self.outdims>1:
                    if X_val.size:
                        self.predict(X_val,False)
                        self.valerror.append(self.frob_cost(y_val,self.normpredictions))
                    self.predict(X,False)
                    self.error.append(self.frob_cost(y,self.normpredictions))
                else:
                    if X_val.size:
                        self.predict(X_val,False)
                        self.valerror.append(self.cost(y_val,self.normpredictions))
                    self.predict(X,False)
                    self.error.append(self.cost(y,self.normpredictions))
  
            
        elif self.solver=="Closed Form":
            X=np.hstack((np.ones((X.shape[0],1)),X))
            self.weights=np.linalg.inv(X.T@X+lam2*np.identity(X.shape[1]))@X.T@y
            if X_val.size:
                    self.predict(X_val,False)
                    if self.outdims>1:
                        self.valerror.append(self.frob_cost(y_val,self.predictions))
                    else:
                        self.valerror.append(self.cost(y_val,self.predictions))
            self.predict(X,False)
            if self.outdims>1:
                self.error.append(self.frob_cost(y,self.predictions))
            else:
                    self.error.append(self.cost(y,self.predictions))
        else:
            print("Not a valid solver. Must be 'Gradient Descent' or 'Closed Form'")





