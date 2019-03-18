import numpy as np
from .tanh_layer import tanh
from .relu_layer import ReLU
from .prelu_layer import pReLU
from .leakyrelu_layer import LeakyReLU
from .sigmoid_layer import Sigmoid
from .out_layer import outactivation
from .conv_layer import convolution
from .SPP_layer import SPP
from .max_pool_layer import maxpool
from .flatten_layer import flatten
from TSML.TSPreprocessandCost import *

activ_functs={}
activ_functs['tanh']=tanh
activ_functs['relu']=ReLU
activ_functs['prelu']=pReLU
activ_functs['leakyrelu']=LeakyReLU
activ_functs['sigmoid']=Sigmoid
activ_functs['outact']=outactivation
activ_functs['conv']=convolution
activ_functs['spp']=SPP
activ_functs['maxpool']=maxpool
activ_functs['flatten']=flatten

class NeuralNet:
    def __init__(self, nodes, indims, activations, filtersize=[], cost=False, task='Classification', solver="Basic", seed=False):
        self.nodes=nodes
        self.layers=len(nodes)
        self.indims=indims
        activations.append('outact')
        self.activations=activations
        self.task=task.lower()
        self.t=0
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
        if self.task !='regression' and self.task !='classification':
            print("Invalid task, use 'Regression' or 'Classification'")
            return None
        solvers=['momentum','nesterov','nesterovrms','nesterovada','momentumada','momentumrms','rmsprop','adagrad','basic', 'adam']

        self.y_classes=np.array([])
        self.Xmean=np.array([])
        self.seed=seed
        self.solver=solver.lower()
        if self.solver not in solvers:
            print("Possible solvers are " +', '.join(solvers))
            return None
        self.layer_classes=[]
        nodes=[indims]+nodes
        f=0
        for i in range(self.layers-1):
            if activations[i].lower()=='conv':
                self.layer_classes.append(activ_functs[activations[i].lower()](nodes[i+1],filtersize[f],nodes[i],solver=self.solver,seed=self.seed))
                f+=1
            else:
                self.layer_classes.append(activ_functs[activations[i].lower()](nodes[i+1],nodes[i],solver=self.solver,seed=self.seed))
                
        self.layer_classes.append(activ_functs[activations[self.layers-1]](nodes[self.layers],nodes[self.layers-1],solver=self.solver,
                                    seed=self.seed, task=self.task))

    def forward(self, X, p, normalize=True):

        if not self.dropout:
            self.predict(X,False)
        else:
            if not self.Xmean.size:
                return("Please train before predicting." )

            elif normalize:
                X=norm(X,self.Xmean,self.Xstd)
            A=X
            if 'nesterov' in self.solver:
                for i in range(self.layers):
                    A=self.layer_classes[i].nesterov_forwardprop_dropout(A,self.mu,p[i])
            else:
                for i in range(self.layers):
                    A=self.layer_classes[i].forwardprop_dropout(A,p[i])

            if self.task=='classification':
                self.probabilities=A
            else:
                self.normpredictions=A

    def predict(self, X, normalize=True):
        if not self.Xmean.size:
            return("Please train before predicting." )

        elif normalize:
            X=norm(X,self.Xmean,self.Xstd)
        A=X
        if 'nesterov' in self.solver:
            for i in range(self.layers):
                A=self.layer_classes[i].nesterov_forwardprop(A,self.mu)
        else:
            for i in range(self.layers):
                A=self.layer_classes[i].forwardprop(A)

        if self.task=='classification':
            self.probabilities=A
            if self.convolutional==False:
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
            if convolutional==False:   
                self.predictions=unnorm(self.normpredictions,self.ymean,self.ystd)


    def weight_initialization(self):
        self.t=0
        for i in range(self.layers-1,-1,-1):
            self.layer_classes[i].weightinit(self.seed)

    def weight_update(self,eta,y):
        D=y
        if self.solver=='nesterov':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].nesterov_backprop(eta,self.mu,D,self.lam1,self.lam2)

        elif self.solver=='basic':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].backprop(eta,D,self.lam1,self.lam2)

        elif self.solver=='momentum':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].momentum_backprop(eta,self.mu,D,self.lam1,self.lam2)

        elif self.solver=='adagrad':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].ada_backprop(eta,D,self.nesterov,self.lam1,self.lam2)

        elif self.solver=='rmsprop':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].RMS_prop(eta,self.gamma,D,self.nesterov,self.lam1,self.lam2)

        elif self.solver=='adam':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].adamoptimizer(eta,self.mu,self.gamma,self.t,D,self.lam1,self.lam2)

        elif self.solver=='momentumrms':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].momentum_RMS_prop(eta,self.mu,self.gamma,D,self.lam1,self.lam2)

        elif self.solver=='momentumada':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].momentum_ada_backprop(eta,self.mu,D,self.lam1,self.lam2)

        elif self.solver=='nesterovrms':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].nesterov_RMS_prop(eta,self.mu,D,self.lam1,self.lam2)

        elif self.solver=='nesterovada':
            for i in range(self.layers-1,-1,-1):
                D=self.layer_classes[i].nesterov_ada_backprop(eta,self.mu,D,self.lam1,self.lam2)

    def train(self,X,y,epochs,eta,mu=0.1,gamma=.9,lam1=0,lam2=0,decay=False,k=0,T=1,batch_size=0,error_calc=True,reinitialize=False,dropout=False,
                p=[],X_val=np.array([]),y_val=np.array([]), convolutional=False):
        if reinitialize:
            self.weight_initialization()
        if batch_size==0:
            batch_size=X.shape[0]
        self.valerror=[]
        self.error=[]
        self.Xmean=X.mean(0)
        self.Xstd=X.std(0)
        self.gamma=gamma
        self.mu=mu
        self.nesterov=False
        self.lam1=lam1
        self.lam2=lam2
        self.convolutional=convolutional
        eta0=eta
        if len(p)==1:
            p=p*(self.layers)
        self.dropout=dropout
        if convolutional==False:
            X=norm(X,self.Xmean,self.Xstd)
    
            if X_val.size:
                X_val=norm(X_val,self.Xmean,self.Xstd)
            if (self.task=="classification" and y.shape[1]<self.nodes[-1]):
                self.y_classes=np.unique(y)
                y=mat_ohe(y,[0])
                if y_val.size:
                    y_val=mat_ohe(y_val,[0])
    
            elif (self.task=="classification" and y.shape[1]==self.nodes[-1]):
                self.positive_class=np.unique(y)[1]
                self.negative_class=np.unique(y)[0]
                y=np.where(y==self.positive_class,1,0)
                if y_val.size:
                    y_val=np.where(y_val==self.positive_class,1,0)
    
            elif self.task=="regression":
                self.ymean=y.mean(0)
                self.ystd=y.std(0)
                y=norm(y,self.ymean,self.ystd)
    
                if y_val.size:
                    y_val=norm(y_val,self.ymean,self.ystd)

        batches=np.ceil(X.shape[0]/batch_size).astype(int)
        for i in range(epochs):
            if convolutional==False:
                train=np.hstack((X,y))
                np.random.shuffle(train)
                train=np.array_split(train,batches)
            else:
                train=[0]
            for a in train:
                if convolutional==False:
                    X_temp=np.array(a[:,:-self.nodes[-1]])
                    y_temp=np.array(a[:,-self.nodes[-1]:])
                else:
                    X_temp=X
                    y_temp=y

                if self.solver == 'nesterovada':
                    self.solver ='adagrad'
                    self.nesterov=True
                    self.forward(X_temp,p,False)
                    self.weight_update(eta,y_temp)
                    self.solver='nesterovada'
                    self.forward(X_temp,p,False)
                    self.weight_update(eta,y_temp)
                elif self.solver == 'nesterovrms':
                    self.solver ='rmsprop'
                    self.nesterov=True
                    self.forward(X_temp,p,False)
                    self.weight_update(eta,y_temp)
                    self.solver='nesterovrms'
                    self.forward(X_temp,p,False)
                    self.weight_update(eta,y_temp)
                else:
                    self.forward(X_temp,p,False)
                    self.weight_update(eta,y_temp)

                if error_calc and self.task=='classification':
                    self.forward(X_temp,p,False)
                    self.error.append(self.cost(y_temp,self.probabilities))

                elif error_calc:
                    self.forward(X_temp,p,False)
                    self.error.append(self.cost(y_temp,self.normpredictions))
                self.t += 1

                if decay=="Scheduled":
                    eta=eta0*k**(self.t/T)
                elif decay=="Inverse":
                    eta=eta0/(k*self.t+1)
                elif decay=="Exponential":
                    eta=eta0*np.exp(-k*self.t)

                if convolutional==False:
                    if self.task=='classification':
                        self.predict(X,False)
                        if self.nodes[-1]>1:
                            acc=(self.ohe_predictions==y).mean()
    
                        elif self.nodes[-1]==1:
                            acc=(self.bin_predictions==y).mean()
                        if acc==1.0:
                            return acc
            if self.task=='classification':
                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.probabilities))

            else:
                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.normpredictions))
