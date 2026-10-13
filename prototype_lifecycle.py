"""Mutual-neighbor communities with fixed-ID external audit and revocation.
Target labels are pseudo labels only. Audit is not a correctness guarantee.
"""
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mutual_components(features, k=10, similarity=.8, min_size=20):
    z = unit(features); n = len(z)
    if n < 2:
        return []
    k = min(k, n-1)
    neighbors = []
    for start in range(0, n, 256):
        sims = z[start:start+256] @ z.T
        sims[np.arange(len(sims)), np.arange(start, start+len(sims))] = -np.inf
        order = np.argsort(-sims, axis=1, kind='stable')[:, :k]
        for row, indices in enumerate(order):
            neighbors.append(set(int(j) for j in indices if sims[row,j] >= similarity))
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i=parent[i]
        return i
    for i, ns in enumerate(neighbors):
        for j in ns:
            if i in neighbors[j]:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [np.array(v, dtype=np.int64) for v in groups.values() if len(v) >= min_size]


def margins(features, labels, base, children):
    scores = unit(features) @ unit(base).T
    for c, p in children:
        scores[:,c] = np.maximum(scores[:,c], unit(features) @ unit(p))
    assigned = scores[np.arange(len(labels)), labels].copy()
    scores[np.arange(len(labels)), labels] = -np.inf
    return assigned - scores.max(1)


class PrototypeLifecycle:
    def __init__(self, classes=7, k=10, similarity=.8, min_size=20,
                 persistence=3, purity=.9, semantic_margin=.02,
                 overlap=.5, audit_min=10, gain=.005, tolerance=.01,
                 merge_similarity=.98, max_children=3, graph_limit=3000):
        if min(classes-1,k,min_size,persistence,audit_min,max_children,graph_limit) < 1:
            raise ValueError('invalid lifecycle size')
        if not 0 <= purity <= 1 or not 0 <= overlap <= 1 or not -1 <= similarity <= 1 or not -1 <= merge_similarity <= 1:
            raise ValueError('invalid lifecycle threshold')
        if not all(np.isfinite(v) for v in (similarity,purity,semantic_margin,overlap,gain,tolerance,merge_similarity)):
            raise ValueError('finite thresholds required')
        if min(semantic_margin,gain,tolerance) < 0:
            raise ValueError('nonnegative margins required')
        self.classes=classes; self.k=k; self.similarity=similarity; self.min_size=min_size
        self.persistence=persistence; self.purity=purity; self.semantic_margin=semantic_margin
        self.overlap=overlap; self.audit_min=audit_min; self.gain=gain; self.tolerance=tolerance
        self.merge_similarity=merge_similarity; self.max_children=max_children; self.graph_limit=graph_limit
        self.tracks=[]; self.active=[]; self.epoch=-1

    def update(self, ids, features, labels, source, base, epoch):
        if epoch != self.epoch+1:
            raise ValueError('consecutive epochs required')
        self.epoch=epoch
        ids=np.asarray(ids,dtype=np.int64); z=np.asarray(features,dtype=np.float32); y=np.asarray(labels,dtype=np.int64)
        if z.ndim != 2 or len(ids)!=len(z) or y.shape!=ids.shape or len(np.unique(ids))!=len(ids):
            raise ValueError('invalid or duplicate samples')
        valid=np.isfinite(z).all(1)&(np.linalg.norm(z,axis=1)>1e-12)&(y>=0)&(y<self.classes)
        ids,z,y=ids[valid],unit(z[valid]),y[valid]
        # Fixed disjoint roles: audit IDs never form graph communities or centroids.
        audit=ids%5==0; build=~audit
        bi=np.flatnonzero(build)
        if len(bi)>self.graph_limit:
            hashes=(ids[bi].astype(np.uint64)*np.uint64(2654435761))%np.uint64(4294967296)
            bi=bi[np.argsort(hashes,kind='stable')[:self.graph_limit]]
        groups=mutual_components(z[bi],self.k,self.similarity,self.min_size)
        new=[]; used=set(); old=self.tracks
        for g in groups:
            members=bi[g]; counts=np.bincount(y[members],minlength=self.classes); c=int(counts.argmax())
            if counts[c]/len(members)<self.purity:
                continue
            # Graph discovered without labels; only now assign semantics and form centroid.
            class_members=members[y[members]==c]
            if len(class_members)<self.min_size:
                continue
            p=unit(z[class_members].mean(0)); ss=unit(source)@p
            if ss[c]-np.max(np.delete(ss,c))<self.semantic_margin:
                continue
            member_ids=set(ids[class_members].tolist()); best=None; best_overlap=0
            for j,t in enumerate(old):
                if j in used or t['class']!=c:continue
                jac=len(member_ids & t['members'])/max(1,len(member_ids | t['members']))
                if jac>best_overlap:best=j;best_overlap=jac
            streak=1
            if best is not None and best_overlap>=self.overlap:
                used.add(best);streak=old[best]['streak']+1
            new.append({'class':c,'prototype':p,'members':member_ids,'streak':streak})
        self.tracks=new
        # Rebuild active set each epoch: disappearing/unstable/failed groups revoked.
        proposals=sorted((t for t in new if t['streak']>=self.persistence),
                         key=lambda t:(-t['streak'],-len(t['members']),t['class']))
        active=[]; merged=failed=0
        az,ay=z[audit],y[audit]
        for t in proposals:
            c,p=t['class'],t['prototype']
            if any(cc==c and float(unit(pp)@p)>=self.merge_similarity for cc,pp in active):
                merged+=1;continue
            if sum(cc==c for cc,_ in active)>=self.max_children:continue
            same=ay==c
            # Require multiple discovered communities: one new center is not a split.
            if sum(u['class']==c for u in proposals)<2:continue
            if same.sum()<self.audit_min:failed+=1;continue
            before=margins(az,ay,base,active)
            proposed=active+[(c,p)]
            after=margins(az,ay,base,proposed)
            affected=same & ((az@p) > (az@unit(base[c])))
            if affected.sum()<self.audit_min or np.mean(after[affected]-before[affected])<self.gain:
                failed+=1;continue
            # Protect each held-out pseudo-class, rather than only majority aggregate.
            if any(np.mean(after[ay==d]-before[ay==d]) < -self.tolerance for d in np.unique(ay)):
                failed+=1;continue
            active=proposed
        self.active=active
        return dict(build=len(bi),audit=int(audit.sum()),communities=len(groups),
                    tracked=len(new),persistent=len(proposals),active=len(active),merged=merged,audit_failed=failed)

    def state_dict(self):
        return dict(epoch=self.epoch,tracks=self.tracks,active=self.active)
