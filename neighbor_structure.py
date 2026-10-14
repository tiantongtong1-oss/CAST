"""Label-free, dual-view-stable reciprocal neighbor relation targets."""
import numpy as np


def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def probabilities(x, temperature):
    a=np.asarray(x,dtype=np.float64)/temperature
    a-=a.max();p=np.exp(a);return p/p.sum()


class StructureBank:
    def __init__(self, k=10, min_neighbors=3, similarity=.5, temperature=.2,
                 max_js=.05, capacity=3000):
        if k<min_neighbors or min_neighbors<2 or capacity<2 or temperature<=0:
            raise ValueError('invalid structure sizes/temperature')
        if not all(np.isfinite(v) for v in (similarity,temperature,max_js)) or not -1<=similarity<=1 or max_js<0:
            raise ValueError('invalid structure thresholds')
        self.k=k;self.min_neighbors=min_neighbors;self.similarity=similarity
        self.temperature=temperature;self.max_js=max_js;self.capacity=capacity
        self.ids=np.empty(0,dtype=np.int64);self.features=None;self.cutoffs=None

    def refresh(self, ids, features):
        ids=np.asarray(ids,dtype=np.int64);features=np.asarray(features,dtype=np.float32)
        if features.ndim!=2 or len(ids)!=len(features) or len(np.unique(ids))!=len(ids):
            raise ValueError('invalid/duplicate bank samples')
        valid=np.isfinite(features).all(1)&(np.linalg.norm(features,axis=1)>1e-12)
        ids,z=ids[valid],normalize(features[valid])
        if len(ids)>self.capacity:
            h=(ids.astype(np.uint64)*np.uint64(2654435761))%np.uint64(4294967296)
            ix=np.argsort(h,kind='stable')[:self.capacity];ids,z=ids[ix],z[ix]
        self.ids,self.features=ids,z
        n=len(ids);self.cutoffs=np.full(n,np.inf,np.float32)
        if n<self.min_neighbors+1:return
        count=min(self.k,n-1)
        for start in range(0,n,256):
            sims=z[start:start+256]@z.T
            sims[np.arange(len(sims)),np.arange(start,start+len(sims))]=-np.inf
            self.cutoffs[start:start+len(sims)]=np.partition(sims,n-count,axis=1)[:,n-count]

    def query(self, ids, view1, view2, accepted):
        ids=np.asarray(ids,dtype=np.int64);v1=np.asarray(view1,dtype=np.float32);v2=np.asarray(view2,dtype=np.float32)
        accepted=np.asarray(accepted,dtype=bool)
        if v1.shape!=v2.shape or v1.ndim!=2 or len(ids)!=len(v1) or accepted.shape!=ids.shape:
            raise ValueError('invalid query shapes')
        rows=[];refs=[];targets=[];js_values=[]
        if self.features is None or len(self.ids)<self.min_neighbors+1:
            return rows,refs,targets,js_values
        if v1.shape[1]!=self.features.shape[1]:raise ValueError('feature dimension mismatch')
        z1,z2=normalize(v1),normalize(v2);mean=normalize(z1+z2)
        for row in np.flatnonzero(accepted):
            if not np.isfinite(v1[row]).all() or not np.isfinite(v2[row]).all() or min(np.linalg.norm(v1[row]),np.linalg.norm(v2[row]),np.linalg.norm(mean[row]))<1e-12:
                continue
            sim=self.features@mean[row];sim[self.ids==ids[row]]=-np.inf
            order=np.argsort(-sim,kind='stable')[:self.k]
            # Candidate would include query in its k-neighbor set. For an existing
            # bank query use its frozen bank feature to test reciprocal membership.
            bank_self=np.flatnonzero(self.ids==ids[row])
            reverse=self.features@self.features[bank_self[0]] if len(bank_self) else sim
            selected=order[(sim[order]>=self.similarity)&(reverse[order]>=self.cutoffs[order])]
            if len(selected)<self.min_neighbors:continue
            r=self.features[selected];p1=probabilities(r@z1[row],self.temperature);p2=probabilities(r@z2[row],self.temperature)
            p=(p1+p2)/2
            js=.5*np.sum(p1*np.log(np.maximum(p1,1e-12)/np.maximum(p,1e-12)))+.5*np.sum(p2*np.log(np.maximum(p2,1e-12)/np.maximum(p,1e-12)))
            if js>self.max_js:continue
            rows.append(int(row));refs.append(r.copy());targets.append(p.astype(np.float32));js_values.append(float(js))
        return rows,refs,targets,js_values


def structure_loss(student_features, query_result, temperature=.2):
    import torch
    import torch.nn.functional as F
    rows,refs,targets,_=query_result
    if not rows:return student_features.sum()*0.0
    losses=[]
    for row,r,p in zip(rows,refs,targets):
        reference=torch.as_tensor(r,device=student_features.device,dtype=student_features.dtype).detach()
        target=torch.as_tensor(p,device=student_features.device,dtype=student_features.dtype).detach()
        logits=F.normalize(student_features[row],dim=0)@reference.T/temperature
        losses.append(F.kl_div(F.log_softmax(logits,dim=0),target,reduction='sum'))
    return torch.stack(losses).mean()
