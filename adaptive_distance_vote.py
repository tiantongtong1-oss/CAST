"""Source-only distance calibration and variable-size neighborhood audit."""
import numpy as np
from dual_domain_ds import EvidenceBank


class AdaptiveDistanceVoteBank(EvidenceBank):
    def calibrate(self, quantile=0.90, k=11):
        if not np.isfinite(quantile) or not 0 < quantile <= 1 or k != 11:
            raise ValueError('require finite quantile in (0,1] and k=11')
        self.distance_thresholds = np.full(self.classes, np.nan)
        for c in range(self.classes):
            x = self.x[self.labels == c]
            if len(x) < 2:
                continue
            rank = min(k, len(x) - 1)
            kth = []
            for start in range(0, len(x), 64):
                d2 = np.maximum(2 - 2 * (x[start:start+64] @ x.T), 0)
                d2[np.arange(len(d2)), np.arange(start, start + len(d2))] = np.inf
                kth.extend(np.sqrt(np.partition(d2, rank-1, axis=1)[:, rank-1]))
            self.distance_thresholds[c] = np.quantile(kth, quantile)
        return self.distance_thresholds.copy()

    def evaluate(self, queries, query_ids, config, candidate_labels, exclude_self=False):
        z = np.asarray(queries, dtype=np.float32)
        labels = np.asarray(candidate_labels, dtype=np.int64)
        ids = np.asarray(query_ids, dtype=np.int64)
        if z.ndim != 2 or z.shape[1] != self.x.shape[1] or labels.shape != (len(z),) or ids.shape != labels.shape:
            raise ValueError('invalid query shapes')
        if config.k != 11 or ((labels < 0) | (labels >= self.classes)).any():
            raise ValueError('require K=11 and valid candidate labels')
        n = len(z)
        result = dict(mass=np.zeros((n,self.classes)), uncertainty=np.ones(n),
            conflict=np.zeros(n), count=np.zeros(n,dtype=np.int64),
            available=np.zeros(n,dtype=bool), votes=np.zeros((n,self.classes),dtype=np.int64),
            distance_threshold=self.distance_thresholds[labels].copy(),
            mean_distance=np.full(n,np.nan), max_distance=np.full(n,np.nan),
            raw_count=np.zeros(n,dtype=np.int64))
        norms=np.linalg.norm(z,axis=1)
        valid=np.isfinite(z).all(1) & (norms>1e-8)
        z=np.nan_to_num(z)/np.maximum(np.nan_to_num(norms[:,None]),1e-8)
        for start in range(0,n,32):
            distances=np.maximum(2-2*(z[start:start+32]@self.x.T),0)
            for offset,d2 in enumerate(distances):
                i=start+offset
                if not valid[i]: continue
                candidates=np.arange(len(self.x))
                if exclude_self: candidates=candidates[self.ids[candidates]!=ids[i]]
                chosen=candidates[np.argsort(d2[candidates],kind='stable')[:11]]
                result['raw_count'][i]=len(chosen)
                threshold=result['distance_threshold'][i]
                if not np.isfinite(threshold): continue
                distance=np.sqrt(d2[chosen])
                keep=distance <= threshold
                chosen,distance=chosen[keep],distance[keep]
                count=len(chosen); result['count'][i]=count
                if not count: continue
                votes=np.bincount(self.labels[chosen],minlength=self.classes)
                result['votes'][i]=votes
                result['mass'][i]=votes/count
                result['mean_distance'][i]=distance.mean()
                result['max_distance'][i]=distance.max()
                result['available'][i]=count>=3
                result['uncertainty'][i]=0 if count>=3 else 1
        return result


def audit_adaptive_vote(source, pseudo_labels):
    labels=np.asarray(pseudo_labels,dtype=np.int64)
    rows=np.arange(len(labels)); mass=source['mass']; counts=source['count']
    votes=source['votes'][rows,labels]
    other_votes=source['votes'].copy(); other_votes[rows,labels]=-1
    vote_gap=votes-other_votes.max(1)
    other=mass.copy(); other[rows,labels]=-np.inf
    # Integer comparison avoids floating rounding at 75%.
    passed=((counts>=3)&(counts<=5)&(votes==counts))
    passed |= (counts>=6)&(counts<=11)&(4*votes>=3*counts)&(vote_gap>=3)
    passed &= source['available']
    return dict(pass_mask=passed, available=source['available'],
        support=mass[rows,labels], vote_count=votes, vote_gap=vote_gap,
        margin=mass[rows,labels]-other.max(1), uncertainty=source['uncertainty'],mass=mass)
