"""Unlabeled target discovery and VICReg loss; no target class supervision."""
from pathlib import Path


def discover_unlabeled_train(root):
    paths=sorted((Path(root)/'train').glob('*/*.jpg'))
    if not paths:raise FileNotFoundError('No target images under train/*/*.jpg')
    # Directory names are not parsed as labels, counted by class, or returned.
    return [str(p) for p in paths]


def vicreg_loss(z1,z2,sim_coeff=25.,std_coeff=25.,cov_coeff=1.):
    import torch
    import torch.nn.functional as F
    if z1.ndim!=2 or z1.shape!=z2.shape or z1.shape[0]<2 or z1.shape[1]<2:
        raise ValueError('VICReg requires matching [B,D] tensors, B,D >=2')
    if min(sim_coeff,std_coeff,cov_coeff)<0:raise ValueError('negative coefficients')
    # No L2 normalization or final BN: these would defeat the std objective.
    invariance=F.mse_loss(z1,z2)
    std1=torch.sqrt(z1.var(dim=0,unbiased=False)+1e-4)
    std2=torch.sqrt(z2.var(dim=0,unbiased=False)+1e-4)
    variance=(F.relu(1-std1).mean()+F.relu(1-std2).mean())/2
    n,d=z1.shape
    x=z1-z1.mean(0);y=z2-z2.mean(0)
    cov1=x.T@x/(n-1);cov2=y.T@y/(n-1)
    offdiag=~torch.eye(d,device=z1.device,dtype=torch.bool)
    covariance=(cov1[offdiag].square().sum()+cov2[offdiag].square().sum())/d
    total=sim_coeff*invariance+std_coeff*variance+cov_coeff*covariance
    return total,dict(invariance=invariance,variance=variance,covariance=covariance,
        projected_std_mean=(std1.mean()+std2.mean())/2,
        projected_fraction_std_below1=((std1<1).float().mean()+(std2<1).float().mean())/2)
