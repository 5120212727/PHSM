#!/usr/bin/env python3
"""Match the held-out validation positions in macon_reproduction/train.py exactly."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--val-seed', type=int, default=2026)
    p.add_argument('--val-fraction', type=float, default=0.05)
    a=p.parse_args()
    if not 0 < a.val_fraction < 0.5:
        p.error('--val-fraction must be between 0 and 0.5')
    out=Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path=out/'train_95.pkl'
    val_path=out/'val_5.pkl'
    meta_path=out/'split_manifest.json'
    if any(p.exists() for p in (train_path,val_path,meta_path)):
        raise FileExistsError(f'Split files already exist under {out}; do not overwrite silently.')
    with open(a.source,'rb') as f:
        parts=pickle.load(f)
    if len(parts) != 4:
        raise ValueError('Expected (texts, tags, post_ids, users)')
    n=len(parts[0]); assert all(len(x)==n for x in parts)
    shuffled=np.random.default_rng(a.val_seed).permutation(n)
    val_n=max(1,round(n*a.val_fraction))
    val_ids,train_ids=shuffled[:val_n],shuffled[val_n:]
    # Order of indices matters. Both splits preserve shuffled relative order.
    def subset(indices):
        return tuple([[part[int(i)] for i in indices] for part in parts])
    with train_path.open('wb') as f:
        pickle.dump(subset(train_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
    with val_path.open('wb') as f:
        pickle.dump(subset(val_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest={
        'source':str(Path(a.source).resolve()),'n_full':n,
        'n_train':len(train_ids),'n_val':len(val_ids),
        'val_seed':a.val_seed,'val_fraction':a.val_fraction,
        'val_indices_sha256':hashlib.sha256(val_ids.tobytes()).hexdigest(),
        'train_indices_sha256':hashlib.sha256(train_ids.tobytes()).hexdigest(),
        'protocol':'np.random.default_rng(val_seed).permutation(n); val_ids = idx[:round(n*val_fraction)]',
        'test_split':'never loaded'
    }
    meta_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[SPLIT]',json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    if n==545262:
        assert (len(train_ids),len(val_ids))==(517999,27263)
    print('[OK] saved',train_path,val_path,flush=True)

if __name__=='__main__': main()
