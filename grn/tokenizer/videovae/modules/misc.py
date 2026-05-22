import torch

def swish(x):
    if type(x) == list:
        for i in range(len(x)):
            x[i] = swish(x[i])
        return x
    try:
        return x*torch.sigmoid(x)
    except:
        for _i in range(x.shape[2]):
            x[:,:,_i:_i+1,:,:] = x[:,:,_i:_i+1,:,:]*torch.sigmoid(x[:,:,_i:_i+1,:,:])
        return x