import numpy as np

class LogisticRegression:
    def __init__(self,indims,outdims):
        
        self.indims=indims
        self.outdims=outdims
        self.weights=np.array([])
        self.Xmins=np.array([])
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def softmax(self,X):
        p=np.exp(X)
        return(p/(p.sum(1).reshape(X.shape[0],1)))
            
    def BCEC(self,y,p):
        Err=[]
        zeros=np.where(y==0)
        ones=np.where(y==1)
        Err=np.hstack((-y[ones]*np.log(p[ones]),-(1-y[zeros])*np.log(1-p[zeros])))
        return np.mean(Err)

    def GCEC(self,y,p):
        return -np.sum(np.multiply(y,np.log(p)))/y.shape[0]

    def weightinit(self):
        self.weights=np.random.randn(self.indims+1,self.outdims)
        
        
    def get_one_hot(self,targets):
        targets=np.asarray(targets).reshape(-1)
        num_classes=len(np.unique(targets))
        indices=np.searchsorted(np.unique(targets),targets)
        res = np.eye(num_classes)[indices]
        return res.reshape(len(targets),num_classes)

    def mat_ohe(self,data, cols_to_ohe):
        ohe_cols=np.apply_along_axis(self.get_one_hot,0,data[:,cols_to_ohe])[:,:,0]
        data=np.hstack((data,ohe_cols))
        data=np.delete(data,cols_to_ohe,1)
        return data.astype(float)
            
    def norm(self,X,Xmin,Xmax):
        return (X-Xmin)/(Xmax-Xmin)
            
    def predict(self, X, normalize=True):
        b = np.sort(X,axis=0)
        if not np.where((b[1:] != b[:-1]).sum(axis=0)+1<20):
            print("At least one of your features has less than 20 unique "\
                  +"values. Perhaps try one hot encoding.")
        
        if not self.Xmins.size:
            self.Xmaxes=X.max(0)
            self.Xmins=X.min(0)
            X=self.norm(X,self.Xmins,self.Xmaxes)
        elif normalize:
            X=self.norm(X,self.Xmins,self.Xmaxes)
            
        if X.shape[1] ==self.indims:
            X=np.hstack((np.ones((X.shape[0],1)),X))
        if not self.weights.size:
            self.weightinit()
        if self.outdims>1:
            self.probabilities=self.softmax(X@self.weights)
            if self.y_classes:
                self.predictions=model.y_classes[np.argmax(self.probabilities, axis=1)]
            else:
                self.predictions=np.eye(self.probabilities.shape[1])[np.argmax(self.probabilities, axis=1)]
        else:
            self.probabilities=self.sigmoid(X@self.weights)
            self.predictions=np.where(self.probabilities==1,self.positive_class,self.negative_class)
            
            
    def train(self,X,y,epochs,eta,X_val=np.array([]),y_val=np.array([]),lam1=0,lam2=0):
        self.valerror=[]
        self.error=[]
        self.Xmaxes=X.max(0)
        self.Xmins=X.min(0)
        X=self.norm(X,self.Xmins,self.Xmaxes)
        X=np.hstack((np.ones((X.shape[0],1)),X))
        if X_val.size:
            X_val=self.norm(X_val,self.Xmins,self.Xmaxes)
            X_val=np.hstack((np.ones((X_val.shape[0],1)),X_val))
      
        if (y.shape[1]>self.outdims):
            self.y_classes=np.unique(y)
            y=self.mat_ohe(y,[0])
            if y_val.size:
                y_val=self.mat_ohe(y_val,[0])
        elif (self.outdims==1):
            self.positive_class=np.unique(y)[1]
            self.negative_class=np.unique(y)[0]
            y=np.where(y==self.positive_class,1,0)
            if y_val.size:
                y_val=np.where(y_val==self.positive_class,1,0)
        self.predict(X,False)
 
        for i in range(epochs):
            self.weights=self.weights-eta*(X.T@(self.probabilities-y))-lam1*self.weights-lam2*np.sign(self.weights)

            if X_val.size:
                self.predict(X_val,False)
                if self.outdims>1:
                    self.valerror.append(self.GCEC(y_val,self.probabilities))
                else:
                    self.valerror.append(self.BCEC(y_val,self.probabilities))
            self.predict(X,False)
            if self.outdims>1:
                self.error.append(self.GCEC(y,self.probabilities))
            else:
                self.error.append(self.BCEC(y,self.probabilities))





