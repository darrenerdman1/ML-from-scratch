import numpy as np

def get_one_hot(targets, uniques):
    if not uniques.size:
        uniques=np.unique(targets)
    targets=np.asarray(targets).reshape(-1)
    num_classes=uniques.shape[0]
    indices=np.searchsorted(uniques,targets)
    res = np.eye(num_classes)[indices]
    return res.reshape(len(targets),num_classes)

def mat_ohe(data, cols_to_ohe, uniques=[]):
    if not uniques:
        uniques=[np.array()]*len(cols_to_ohe)
    for i in range(len(cols_to_ohe)):
        ohe_cols=get_one_hot(data[:,cols_to_ohe[i]], uniques[i])
        data=np.hstack((data,ohe_cols))
    data=np.delete(data,cols_to_ohe,1)
    return data.astype(float)

def norm(X,Xmean,Xstd):
    return (X-Xmean)/Xstd

def unnorm(X,Xmean,Xstd):
    return (X*Xstd)+Xmean

def GCEC(y,p):
    return -1*np.sum(np.multiply(y,np.log(p)))/y.shape[0]

def SSE(y,y_hat):
    return np.asscalar((y_hat-y).T@(y_hat-y))/y.shape[0]

def FSSE(y,y_hat):
    return np.trace((y_hat-y).T@(y_hat-y))/y.shape[0]

def BCEC(y,p):
    Err=[]
    zeros=np.where(y==0)
    ones=np.where(y==1)
    Err=np.hstack((-y[ones]*np.log(p[ones]).reshape(-1),-(1-y[zeros])*np.log(1-p[zeros]).reshape(-1)))
    return np.mean(Err)
