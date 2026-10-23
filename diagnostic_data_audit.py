"""Dataset split/label alignment checks for independent RAF evaluation."""
from pathlib import Path
import numpy as np

EXPECTED_FER_TO_CAST={0:5,1:2,2:1,3:3,4:4,5:0,6:6}


def assert_disjoint_paths(reference_paths,query_paths):
    refs=[str(Path(p).resolve()) for p in reference_paths]
    queries=[str(Path(p).resolve()) for p in query_paths]
    if len(set(refs))!=len(refs) or len(set(queries))!=len(queries):
        raise ValueError('duplicate image paths within reference/query split')
    overlap=set(refs)&set(queries)
    if overlap:raise ValueError('reference/query image path overlap: '+next(iter(overlap)))
    return dict(reference_count=len(refs),query_count=len(queries),resolved_path_overlap=0)


def audit_raf_labels(root,paths,labels,split):
    if split not in ['train','test']:raise ValueError('invalid RAF split')
    official={}
    with (Path(root)/'EmoLabel/list_patition_label.txt').open() as f:
        for line in f:
            parts=line.split()
            if not parts:continue
            name,raw=parts;raw=int(raw)
            if not 1<=raw<=7:raise ValueError('RAF label must be 1..7')
            aligned=Path(name).stem+'_aligned.jpg'
            if aligned in official:raise ValueError('duplicate RAF official image')
            official[aligned]=(raw-1,name.startswith(split+'_'))
    expected={name for name,(_,matches) in official.items() if matches}
    if len(paths)!=len(labels):raise ValueError('RAF path/label count mismatch')
    if {Path(p).name for p in paths}!=expected:raise ValueError('RAF loader does not match complete official '+split+' split')
    for path,label in zip(paths,labels):
        if official[Path(path).name]!=(int(label),True):raise ValueError('RAF shuffled path/label mismatch: '+str(path))
    return dict(split=split,official_samples=len(expected),path_label_alignment='verified',
                class_counts=np.bincount(np.asarray(labels,dtype=int),minlength=7).tolist())
