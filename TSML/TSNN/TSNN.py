import numpy as np
from .tanh_layer import tanh
from .relu_layer import ReLU
from .prelu_layer import pReLU
from .leakyrelu_layer import LeakyReLU
from .sigmoid_layer import Sigmoid
from .out_layer import outactivation

activ_functs={}
activ_functs['tanh']=tanh
activ_functs['relu']=ReLU
activ_functs['prelu']=pReLU
activ_functs['leakyrelu']=LeakyReLU
activ_functs['sigmoid']=Sigmoid
activ_functs['outact']=outactivation


def BCEC(y,p):
    Err=[]
    zeros=np.where(y==0)
    ones=np.where(y==1)
    Err=np.hstack((-y[ones]*np.log(p[ones]),-(1-y[zeros])*np.log(1-p[zeros])))
    return np.mean(Err)

def GCEC(y,p):
    return -1*np.sum(np.multiply(y,np.log(p)))/y.shape[0]

def SSE(y,y_hat):
    return ((y_hat-y).T@(y_hat-y))/len(y)

def FSSE(y,y_hat):
    return np.trace((y_hat-y).T@(y_hat-y))/len(y)

class NeuralNet:
    def __init__(self, nodes, indims, activations, cost=False, task='Classification', solver="Basic", scaleweights=True, seed=False):
        self.nodes=nodes
        self.layers=len(nodes)
        self.indims=indims
        activations.append('outact')
        self.activations=activations
        self.task=task.lower()
        if not cost and task.lower()=='classification':
            if nodes[-1]==1:
                self.cost=BCEC
            else:
                self.cost=GCEC
        elif not cost and task.lower()=='regression':
            if nodes[-1]==1:
                self.cost=SSE
            else:
                self.cost=FSSE
        if task.lower() !='regression' and task.lower() !='classification':
            print("Invalid task, use 'Regression' or 'Classification'")
            return None

        self.y_classes=np.array([])
        self.Xmins=np.array([])
        self.scaleweights=scaleweights
        self.seed=seed
        self.solver=solver.lower()
        self.layer_classes=[]
        nodes=[indims]+nodes
        for i in range(self.layers-1):
            self.layer_classes.append(activ_functs[activations[i].lower()](nodes[i+1],nodes[i],solver=self.solver,
                                        scaleweights=self.scaleweights,seed=self.seed))

        self.layer_classes.append(activ_functs[activations[self.layers-1]](nodes[self.layers],nodes[self.layers-1],solver=self.solver,
                                    scaleweights=self.scaleweights,seed=self.seed, task=self.task))


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

    def unnorm(self,X,Xmin,Xmax):
        return (X)*(Xmax-Xmin)+Xmin


    def predict(self, X, normalize=True):

        if not self.Xmins.size:
            self.Xmaxes=X.max(0)
            self.Xmins=X.min(0)
            X=self.norm(X,self.Xmins,self.Xmaxes)

        elif normalize:
            X=self.norm(X,self.Xmins,self.Xmaxes)

        A=X
        for i in range(self.layers):
            if self.solver=='nesterov':
                A=self.layer_classes[i].nesterov_forwardprop(A,self.mu)
            else:
                A=self.layer_classes[i].forwardprop(A)

        if self.task=='classification':
            #Convert to array in case a matrix was passed
            self.probabilities=A
            self.ohe_predictions=np.eye(self.probabilities.shape[1])[np.argmax(self.probabilities, axis=1)]

            if self.nodes[-1]>1 and self.y_classes.size:
                 self.predictions=self.y_classes[np.argmax(self.probabilities, axis=1)].reshape(-1,1)

            elif self.nodes[-1]>1:
                self.predictions=self.ohe_predictions

            else:
                self.bin_predictions=np.rint(self.probabilities)
                self.predictions=np.where(self.bin_predictions==1,self.positive_class,self.negative_class).reshape(-1,1)

        else:
            self.normpredictions=A
            self.predictions=self.unnorm(self.normpredictions,self.ymins,self.ymaxes)

    def weight_initialization(self):
        for i in range(self.layers-1,-1,-1):
            self.layer_classes[i].weightinit()

    def weight_update(self,eta,y):
        D=y
        if self.solver=='nesterov':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].nesterov_backprop(eta,self.mu,D)

        elif self.solver=='basic':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].backprop(eta,D)

        elif self.solver=='momentum':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].momentum_backprop(eta,self.mu,D)

        elif self.solver=='adagrad':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].ada_backprop(eta,D)

        elif self.solver=='rmsprop':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].RMS_prop(eta,self.gamma,D)

        elif self.solver=='adam':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].adamoptimizer(eta,self.mu,self.gamma,self.t,D)

    def train(self,X,y,epochs,eta,mu=0.1,gamma=.9,batch_size=0,error_calc=True,reinitialize=False,
                X_val=np.array([]),y_val=np.array([])):
        if reinitialize:
            self.weight_initialization()
        if batch_size==0:
            batch_size=X.shape[0]
        self.valerror=[]
        self.error=[]
        self.Xmaxes=X.max(0)
        self.Xmins=X.min(0)
        self.gamma=gamma
        self.mu=mu
        X=self.norm(X,self.Xmins,self.Xmaxes)

        if X_val.size:
            X_val=self.norm(X_val,self.Xmins,self.Xmaxes)
        if (self.task=="classification" and y.shape[1]<self.nodes[-1]):
            self.y_classes=np.unique(y)
            y=self.mat_ohe(y,[0])
            if y_val.size:
                y_val=self.mat_ohe(y_val,[0])

        elif (self.task=="classification" and y.shape[1]==self.nodes[-1]):
            self.positive_class=np.unique(y)[1]
            self.negative_class=np.unique(y)[0]
            y=np.where(y==self.positive_class,1,0)
            if y_val.size:
                y_val=np.where(y_val==self.positive_class,1,0)

        elif self.task=="regression":
            self.ymaxes=y.max(0)
            self.ymins=y.min(0)
            y=self.norm(y,self.ymins,self.ymaxes)

            if y_val.size:
                y_val=self.norm(y_val,self.ymins,self.ymaxes)

        batches=np.ceil(X.shape[0]/batch_size).astype(int)
        train=np.hstack((X,y))
        np.random.shuffle(train)
        train=np.array_split(train,batches)
        self.t=0
        for i in range(epochs):
            for a in train:
                X=np.array(a[:,:-self.nodes[-1]])
                y=np.array(a[:,-self.nodes[-1]:])
                self.predict(X,False)
                self.weight_update(eta,y)

                if error_calc and self.task=='classification':
                    self.predict(X,False)
                    self.error.append(self.cost(y,self.probabilities))

                elif error_calc:
                    self.predict(X,False)
                    self.error.append(self.cost(y,self.normpredictions))
                self.t += 1

            if self.task=='classification':
                if self.nodes[-1]>1:
                    acc=(self.ohe_predictions==y).mean()

                elif self.nodes[-1]==1:
                    acc=(self.bin_predictions==y).mean()

                if acc== 1.0:
                    break

                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.probabilities))

            else:
                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.normpredictions))
