"""Unweighted majority audit using exact Euclidean KNN on unit features."""
import numpy as np
from dual_domain_ds import EvidenceBank

class NeighborVoteBank(EvidenceBank):
    def evaluate(self, queries, query_ids, config, exclude_self=False):
        z = np.asarray(queries, dtype=np.float32)
        ids = np.asarray(query_ids, dtype=np.int64)
        if z.ndim != 2 or z.shape[1] != self.x.shape[1] or ids.shape != (len(z),):
            raise ValueError('invalid query shapes')
        n = len(z)
        result = dict(mass=np.zeros((n,self.classes)), uncertainty=np.ones(n),
            conflict=np.zeros(n), count=np.zeros(n,dtype=np.int64),
            available=np.zeros(n,dtype=bool), votes=np.zeros((n,self.classes),dtype=np.int64))
        norms=np.linalg.norm(z,axis=1)
        valid=np.isfinite(z).all(1) & (norms>1e-8)
        z=np.nan_to_num(z)/np.maximum(np.nan_to_num(norms[:,None]),1e-8)
        for start in range(0,n,32):
            distances=np.maximum(2-2*(z[start:start+32]@self.x.T),0)
            for offset,distance in enumerate(distances):
                i=start+offset
                if not valid[i]:continue
                allowed=np.ones(len(self.x),dtype=bool)
                if exclude_self:allowed &= self.ids != ids[i]
                candidates=np.flatnonzero(allowed)
                chosen=candidates[np.argsort(distance[candidates],kind='stable')[:config.k]]
                result['count'][i]=len(chosen)
                # Require the FULL K-neighborhood: never turn 3 of 3 into 6 of 11.
                if len(chosen) != config.k:continue
                votes=np.bincount(self.labels[chosen],minlength=self.classes)
                result['votes'][i]=votes
                result['mass'][i]=votes/config.k
                result['uncertainty'][i]=0
                result['available'][i]=True
        return result


def audit_vote(source, pseudo_labels, minimum_votes=6):
    labels=np.asarray(pseudo_labels,dtype=np.int64)
    mass=source['mass']
    if labels.shape != (len(mass),) or ((labels<0)|(labels>=mass.shape[1])).any():
        raise ValueError('invalid pseudo-labels')
    votes=source['votes'][np.arange(len(labels)),labels]
    support=mass[np.arange(len(labels)),labels]
    other=mass.copy();other[np.arange(len(labels)),labels]=-np.inf
    return dict(pass_mask=source['available'] & (votes>=minimum_votes),
        available=source['available'], support=support, vote_count=votes,
        margin=support-other.max(1), uncertainty=source['uncertainty'], mass=mass)


def rescue_union(radius_pass, vote_pass):
    """Independent rescue branches; callers enforce agreement, low confidence, warmup."""
    return radius_pass | vote_pass
