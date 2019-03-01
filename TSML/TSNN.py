import numpy as np


class NeuralNet:
    def __init__(self,nodes,indims,activations,
                  actders, cost, task='Classifcation',scaleweights=True, seed=False):
        self.nodes=nodes
        self.layers=len(nodes)
        self.indims=indims
        self.activations=activations
        self.actders=actders
        self.task=task
        self.cost=cost
        self.weights={}
        self.biases={}
        self.y_classes=[]
        self.Xmins=np.array([])
        self.scaleweights=scaleweights
        if seed !=False:
            np.random.seed(seed)

    def weightinit(self):
        self.weights['W0']=np.random.randn(self.indims,self.nodes[0])
        self.biases['B0']=np.random.randn(1,self.nodes[0])
        for i in range(1,self.layers):
            if self.scaleweights:
                self.weights['W'+str(i)]=np.random.randn(self.nodes[i-1],self.nodes[i])*np.sqrt(1/(self.nodes[i-1]+self.nodes[i]))
                self.biases['B'+str(i)]=np.random.randn(1,self.nodes[i])*np.sqrt(1/(self.nodes[i-1]+self.nodes[i]))
            else:
                self.weights['W'+str(i)]=np.random.randn(self.nodes[i-1],self.nodes[i])
                self.biases['B'+str(i)]=np.random.randn(1,self.nodes[i])

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
        self.Z={}
        self.Zprime={}

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

        if not self.weights:
            self.weightinit()

        self.Z['Z0']=X

        for i in range(self.layers-1):
            self.Z['Z'+str(i+1)]=self.activations[i](self.Z['Z'+str(i)]@self.weights['W'+str(i)]+\
                  self.biases["B"+str(i)])
            self.Zprime['Zprime'+str(i)]=self.actders[i](self.Z['Z'+str(i)]@self.weights['W'+str(i)]+\
                       self.biases["B"+str(i)])
        if self.task=='Classification':
            #Convert to array in case a matrix was passed
            self.probabilities=np.array(self.activations[self.layers-1](self.Z['Z'+str(self.layers-1)]\
                                        @self.weights['W'+str(self.layers-1)]+\
                                        self.biases["B"+str(self.layers-1)]))
            if self.nodes[-1]>1 and self.y_classes:
                 self.predictions=model.y_classes[np.argmax(self.probabilities, axis=1)]
            elif self.nodes[-1]>1:
                self.predictions=np.eye(self.probabilities.shape[1])[np.argmax(self.probabilities, axis=1)]
            else:
                self.predictions=np.where(np.rint(self.probabilities)==1,self.positive_class,self.negative_class)
        elif self.task=='Regression':
            self.normpredictions=self.Z['Z'+str(self.layers-1)]@self.weights['W'+str(self.layers-1)]+\
            self.biases["B"+str(self.layers-1)]

            self.predictions=self.unnorm(self.normpredictions,self.ymins,self.ymaxes)
        else:
            self.predictions="Invalid task. Must use regression or classification.Default is classification"

    def train(self,X,y,epochs,eta,X_val=np.array([]),y_val=np.array([])):
        self.valerror=[]
        self.error=[]
        d={}
        self.Xmaxes=X.max(0)
        self.Xmins=X.min(0)
        X=self.norm(X,self.Xmins,self.Xmaxes)
        if X_val.size:
            X_val=self.norm(X_val,self.Xmins,self.Xmaxes)

        if (self.task=="Classification" and y.shape[1]<self.nodes[-1]):
            self.y_classes=np.unique(y)
            y=self.mat_ohe(y,[0])
            if y_val.size:
                y_val=self.mat_ohe(y_val,[0])
        elif (self.task=="Classification" and y.shape[1]==self.nodes[-1]):
            self.positive_class=np.unique(y)[1]
            self.negative_class=np.unique(y[0])
            y=np.where(y==self.positive_class,1,0)
            if y_val.size:
                y_val=np.where(y_val==self.positive_class,1,0)
        elif self.task=="Regression":
            self.ymaxes=y.max(0)
            self.ymins=y.min(0)
            y=self.norm(y,self.ymins,self.ymaxes)
            if y_val.size:
                y_val=self.norm(y_val,self.ymins,self.ymaxes)
        self.predict(X,False)
        for i in range(epochs):
            if self.task=='Classification':
                d['d'+str(self.layers-1)]=self.probabilities-y
            elif self.task=='Regression':
                d['d'+str(self.layers-1)]=self.normpredictions-y
            else:
                return self.predictions

            for j in range(self.layers-1):
                d['d'+str(self.layers-2-j)]=np.multiply(d['d'+str(self.layers-1-j)]                                                        @self.weights['W'+str(self.layers-1-j)].T,                                                        self.Zprime['Zprime'+str(self.layers-2-j)])

            for j in range(self.layers):
                self.weights['W'+str(j)]=self.weights['W'+str(j)]-eta*self.Z['Z'+str(j)].T@d['d'+str(j)]
                self.biases['B'+str(j)]=self.biases['B'+str(j)]-eta*np.sum(d['d'+str(j)], axis=0)

            if self.task=='Classification':
                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.probabilities))
                self.predict(X,False)
                self.error.append(self.cost(y,self.probabilities))
            else:
                if y_val.size:
                    self.predict(X_val,False)
                    self.valerror.append(self.cost(y_val,self.normpredictions))
                self.predict(X,False)
                self.error.append(self.cost(y,self.normpredictions))
