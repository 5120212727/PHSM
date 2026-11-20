#!/usr/bin/env python3
"""Stages: baseline, v1, v2, v3. Parallel Co-Attention PHSM with held-out validation.

All models, losses, hashtag branches, and prototype construction retain the
PHSM original source; only SimpleCoAttention is replaced by the MaCon PCA.
This script adds shared fixed train/val splits and gradient accumulation.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import build_loader
from model import MaConLikeBaseline, move_batch_to_device
from model_matching_v1 import SemanticMatchingV1
from model_matching_v2 import SemanticMatchingV2
from model_matching_v3 import SemanticMatchingV3
from train_helpers import compute_pos_weight, build_train_mask_and_pos_weight, masked_bce_loss

STAGE_CKPT = {
    'baseline': 'baseline_best.pt',
    'v1': 'matching_v1_best.pt',
    'v2': 'matching_v2_best.pt',
    'v3': 'matching_v3_best.pt',
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device, k):
    model.eval()
    totals = dict(p=0., r=0., f1=0., acc=0.)
    n=0
    for batch in loader:
        batch=move_batch_to_device(batch,device)
        logits=model(batch)['logits']
        y=batch['y']
        ids=logits.topk(k,dim=1).indices
        hits=y.gather(1,ids).sum(dim=1)
        count=y.sum(dim=1).clamp(min=1)
        p=hits/k
        r=hits/count
        f1=2*hits/(k+count)
        totals['p']+=float(p.sum().item());totals['r']+=float(r.sum().item())
        totals['f1']+=float(f1.sum().item())
        totals['acc']+=float((hits>0).sum().item())
        n+=y.shape[0]
    if not n: raise ValueError('Empty validation dataset')
    return {'n':n, **{key:value/n for key,value in totals.items()}}

def make_model(a):
    common=dict(num_tags=a.num_tags,vocab_size=a.vocab_size,emb_dim=a.emb_dim,
                text_hidden_dim=a.text_hidden_dim,num_hist=2,dropout=a.dropout)
    if a.stage=='baseline':
        return MaConLikeBaseline(**common)
    common['tag_text_inputs_pkl']=a.tag_text_inputs
    if a.stage=='v1':
        return SemanticMatchingV1(**common)
    common['tag_prototypes_pkl']=a.tag_prototypes
    common['post_dim']=a.post_dim
    if a.stage=='v2':
        return SemanticMatchingV2(**common)
    return SemanticMatchingV3(**common)

def strict_load_post(model, path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    state=ck['model_state']
    # PHSM baseline includes the classification head, while the V1/V2/V3
    # encoder contains the same head. Require identical shapes and keys.
    model.post_encoder.load_state_dict(state,strict=True)

def init_model(a, model):
    if a.stage == 'baseline':
        if a.init_ckpt:
            ck = torch.load(
                a.init_ckpt,
                map_location='cpu',
                weights_only=False
            )

            model.load_state_dict(
                ck['model_state'],
                strict=True
            )

            print(
                f"[INIT] loaded baseline checkpoint: "
                f"{a.init_ckpt}, source_epoch={ck.get('epoch')}",
                flush=True
            )
        return

    if not a.init_ckpt:
        raise ValueError(f'--init-ckpt is required for stage {a.stage}')

    if not Path(a.init_ckpt).exists():
        raise FileNotFoundError(a.init_ckpt)

    if a.stage == 'v1':
        strict_load_post(model, a.init_ckpt)
    elif a.stage == 'v2':
        model.load_from_v1(a.init_ckpt)
    elif a.stage == 'v3':
        model.load_from_v2(a.init_ckpt)

    print('[INIT]', a.init_ckpt, flush=True)

def initialize_v3_id_from_prototype(a, model):
    """Initialize V3 trainable ID vectors using V2 projected prototypes.

    All modifications occur in model RAM, before training and before the model
    moves to GPU. The source prototype file and every checkpoint remain intact.
    """
    if a.stage != 'v3':
        if a.v3_id_init != 'random' or a.v3_id_init_noise != 0.0:
            raise ValueError('--v3-id-init is only meaningful for --stage v3')
        return
    if a.v3_id_init == 'random':
        if a.v3_id_init_noise != 0.0:
            raise ValueError('--v3-id-init-noise requires --v3-id-init prototype')
        print('[ID-INIT] mode=random (existing behavior); id_gate starts at sigmoid(-3)', flush=True)
        return
    if not 0.0 <= a.v3_id_init_noise <= 1.0:
        raise ValueError('--v3-id-init-noise must be between 0 and 1')
    with torch.no_grad():
        # The prototype buffer is built using the chosen PCA baseline.
        # Crucially, proto_proj has already been restored from the V2 checkpoint.
        projected = model.proto_proj(model.prototype).detach()
        valid = model.id_mask.reshape(-1).bool()
        weight = model.tag_id_embedding.weight
        if projected.shape != weight.shape or valid.shape[0] != weight.shape[0]:
            raise RuntimeError(f'ID/prototype shape mismatch: {projected.shape} vs {weight.shape}')
        if int(valid.sum().item()) == 0:
            raise RuntimeError('No seen-label prototypes available for ID initialization')
        norms = projected[valid].norm(dim=-1)
        if torch.any(norms < 1e-8):
            raise RuntimeError('Some seen-tag projected prototypes have zero norm')
        base = F.normalize(projected, dim=-1)
        if a.v3_id_init_noise == 0.0:
            init = base
        else:
            noise = F.normalize(torch.randn_like(base), dim=-1)
            init = F.normalize(base + a.v3_id_init_noise * noise, dim=-1)
        # Never change unseen-tag ID weights: their ID mask is zero anyway.
        weight[valid] = init[valid].to(dtype=weight.dtype)
        cos = F.cosine_similarity(F.normalize(weight[valid], dim=-1), base[valid], dim=-1)
        print(f'[ID-INIT] mode=prototype; valid={int(valid.sum())}/{valid.numel()}; '
              f'noise={a.v3_id_init_noise:g}; '
              f'mean_cosine_to_V2_projected_prototype={float(cos.mean()):.6f}; '
              f'id_gate={float(torch.sigmoid(model.id_gate_logit)):.6f}', flush=True)

def train(a):
    seed_everything(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] stage={a.stage}, device={device}, seed={a.seed}, effective_batch={a.batch_size*a.grad_accum}',flush=True)
    path=Path(a.out_dir)
    best_path=path/STAGE_CKPT[a.stage]
    if best_path.exists(): raise FileExistsError(f'{best_path} exists; refusing overwrite')
    path.mkdir(parents=True,exist_ok=True)
    # Ensure original test split is never used for either of the two loaders.
    if Path(a.train_pkl).resolve()==Path(a.val_pkl).resolve(): raise ValueError('train and val must differ')
    if 'testset' in Path(a.val_pkl).name.lower(): raise ValueError('Test split cannot be used as validation')
    if a.grad_accum<1 or a.batch_size<1: raise ValueError('Invalid batch sizes')
    (path/'args.json').write_text(json.dumps(vars(a),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    # Dedicated data loader generators avoid consuming model/global RNG.
    train_generator = torch.Generator().manual_seed(a.seed)
    val_generator = torch.Generator().manual_seed(a.seed + 1)
    print(f'[RNG] train_loader_seed={a.seed}, val_loader_seed={a.seed + 1}', flush=True)
    print('[INFO] loading train / validation datasets',flush=True)
    train_ds, train_loader=build_loader(a.train_pkl,a.user_history_pkl,a.img_feat_h5,
                                       batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,generator=train_generator)
    val_ds,val_loader=build_loader(a.val_pkl,a.user_history_pkl,a.img_feat_h5,
                                   batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,generator=val_generator)
    print('[INFO] train samples',len(train_ds),'val samples',len(val_ds),flush=True)
    model=make_model(a)
    init_model(a,model)
    initialize_v3_id_from_prototype(a,model)
    model=model.to(device)
    if a.stage=='baseline':
        pos_weight=compute_pos_weight(a.train_pkl,num_tags=a.num_tags,max_weight=a.max_pos_weight).to(device)
        mask=None
    else:
        mask,pos_weight=build_train_mask_and_pos_weight(a.train_pkl,'',num_tags=a.num_tags,
                                                        max_weight=a.max_pos_weight,original_split=True)
        mask=mask.to(device);pos_weight=pos_weight.to(device)
    # Optional V3 learning rates: inherited weights, ID Embedding, and ID Gate.
    use_v3_group_lr = any(v is not None for v in (
        a.v3_inherited_lr, a.v3_id_lr, a.v3_id_gate_lr,
    ))
    if use_v3_group_lr:
        if a.stage != 'v3':
            raise ValueError('Grouped learning rates are only supported for --stage v3')
        if a.v3_inherited_lr is None or a.v3_id_lr is None:
            raise ValueError('Pass both --v3-inherited-lr and --v3-id-lr')
        gate_lr = a.v3_id_lr if a.v3_id_gate_lr is None else a.v3_id_gate_lr
        if min(a.v3_inherited_lr, a.v3_id_lr, gate_lr) <= 0:
            raise ValueError('Learning rates must be positive')

    if a.stage == 'v3' and use_v3_group_lr:
        inherited_params, embedding_params, gate_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'tag_id_embedding.weight':
                embedding_params.append(param)
            elif name == 'id_gate_logit':
                gate_params.append(param)
            else:
                inherited_params.append(param)
        if not inherited_params or len(embedding_params) != 1 or len(gate_params) != 1:
            raise RuntimeError('Unexpected trainable V3 parameter groups')
        optimizer = AdamW(
            [
                {'params': inherited_params, 'lr': a.v3_inherited_lr},
                {'params': embedding_params, 'lr': a.v3_id_lr},
                {'params': gate_params, 'lr': gate_lr},
            ],
            weight_decay=a.weight_decay,
        )
        print(f'[GROUP-LR] inherited={a.v3_inherited_lr:g}, '
              f'id_embedding={a.v3_id_lr:g}, id_gate={gate_lr:g}, '
              f'weight_decay={a.weight_decay:g}', flush=True)
        print(f'[GROUP-LR] inherited trainable tensors={len(inherited_params)}, '
              f'ID Embedding tensors={len(embedding_params)}, '
              f'ID Gate tensors={len(gate_params)}', flush=True)
    else:
        optimizer = AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        print(f'[GROUP-LR] disabled; all trainable parameters lr={a.lr:g}', flush=True)
    best_f1=-1
    for epoch in range(1,a.epochs+1):
        # V3 MaCon Softmax: piecewise learning-rate decay
        if a.stage == 'v3' and a.v3_loss == 'macon_softmax':
            if epoch <= 20:
                lr_now = 1e-4
            elif epoch <= 30:
                lr_now = 5e-5
            else:
                lr_now = 1e-5

            for group in optimizer.param_groups:
                group['lr'] = lr_now

            print(
                f'[LR-SCHEDULE] epoch={epoch}/{a.epochs} '
                f'lr={lr_now:g}',
                flush=True
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window=0
        total_loss=0.;n=0
        steps=min(len(train_loader),a.debug_train_batches) if a.debug_train_batches else len(train_loader)
        bar=tqdm(train_loader,total=steps,desc=f'{a.stage} epoch {epoch}',dynamic_ncols=True,leave=True,file=sys.stdout)
        for batch_idx,cpu_batch in enumerate(bar):
            if batch_idx>=steps: break
            batch=move_batch_to_device(cpu_batch,device)
            logits=model(batch)['logits']
            y=batch['y']
            if a.stage=='baseline':
                loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)
            elif a.stage in ('v1', 'v2', 'v3') and a.v3_loss == 'macon_softmax':
                loss = -(y * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                loss=masked_bce_loss(logits,y,mask,pos_weight)
            (loss/a.grad_accum).backward()
            n+=y.shape[0];total_loss+=float(loss.item())*y.shape[0]
            window+=1
            if window==a.grad_accum or batch_idx==steps-1:
                if window!=a.grad_accum:
                    for param in model.parameters():
                        if param.grad is not None: param.grad.mul_(a.grad_accum/window)
                if a.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
                optimizer.step();optimizer.zero_grad(set_to_none=True);window=0
            if (batch_idx+1)%25==0 or batch_idx+1==steps:
                if a.stage == 'v3':
                    bar.set_postfix(
                        id_gate=f'{torch.sigmoid(model.id_gate_logit.detach()).item():.4f}',
                        loss=f'{total_loss/max(n,1):.5f}',
                        proto_gate=f'{torch.sigmoid(model.proto_gate_logit.detach()).item():.4f}',
                    )
                else:
                    bar.set_postfix(loss=f'{total_loss/max(n,1):.5f}')
        bar.close()
        metrics=evaluate(model,val_loader,device,7)
        print(f'[Val@7] epoch={epoch}, acc={metrics["acc"]:.5f}, P={metrics["p"]:.5f}, R={metrics["r"]:.5f}, F1={metrics["f1"]:.6f}',flush=True)
        record={'epoch':epoch,'train_loss':total_loss/max(n,1),'val':metrics,'debug':bool(a.debug_train_batches)}
        if a.stage == 'v3':
            record['id_gate'] = float(torch.sigmoid(model.id_gate_logit.detach()).item())
            record['proto_gate'] = float(torch.sigmoid(model.proto_gate_logit.detach()).item())
            print(f'[GATE] epoch={epoch}, id_gate={record["id_gate"]:.6f}, '
                  f'proto_gate={record["proto_gate"]:.6f}', flush=True)
        with (path/'history.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if metrics['f1']>best_f1:
            best_f1=metrics['f1']
            payload=dict(epoch=epoch, model_state=model.state_dict(),
                         warm_metrics={'f1':metrics['f1'],'precision':metrics['p'],'recall':metrics['r'],'acc':metrics['acc']},
                         args=vars(a),architecture='PHSM-PCA; MaCon ParallelCoAttention dim_k=100')
            torch.save(payload,best_path)
            print(f'[OK] best checkpoint: {best_path} F1={best_f1:.6f}',flush=True)
    last_path=path/STAGE_CKPT[a.stage].replace('_best.pt','_last.pt')
    torch.save(dict(epoch=epoch,model_state=model.state_dict(),args=vars(a)),last_path)
    print(f'[DONE] {a.stage} best validation F1@7={best_f1:.6f}',flush=True)
    train_ds.close();val_ds.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=STAGE_CKPT,required=True)
    p.add_argument('--train-pkl',required=True)
    p.add_argument('--val-pkl',required=True)
    p.add_argument('--user-history-pkl',required=True)
    p.add_argument('--img-feat-h5',required=True)
    p.add_argument('--tag-text-inputs',default='')
    p.add_argument('--tag-prototypes',default='')
    p.add_argument('--init-ckpt',default='')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=64)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--max-pos-weight',type=float,default=25.0)
    p.add_argument('--v3-loss', choices=['bce', 'macon_softmax'], default='bce')
    p.add_argument('--lr',type=float,default=None)
    p.add_argument('--v3-inherited-lr',type=float,default=None)
    p.add_argument('--v3-id-lr',type=float,default=None)
    p.add_argument('--v3-id-gate-lr',type=float,default=None)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--grad-clip',type=float,default=5.0)
    p.add_argument('--num-tags',type=int,default=3896)
    p.add_argument('--vocab-size',type=int,default=212003)
    p.add_argument('--emb-dim',type=int,default=300)
    p.add_argument('--text-hidden-dim',type=int,default=150)
    p.add_argument('--post-dim',type=int,default=600)
    p.add_argument('--dropout',type=float,default=0.5)
    p.add_argument('--v3-id-init',choices=['random','prototype'],default='random')
    p.add_argument('--v3-id-init-noise',type=float,default=0.0)
    p.add_argument('--debug-train-batches',type=int,default=0)
    a=p.parse_args()
    if a.lr is None:a.lr=(1e-3 if a.stage=='baseline' else (5e-4 if a.stage=='v1' else 1e-4))
    if a.stage in {'v1','v2','v3'} and not a.tag_text_inputs: p.error('--tag-text-inputs required')
    if a.stage in {'v2','v3'} and not a.tag_prototypes: p.error('--tag-prototypes required')
    train(a)
if __name__=='__main__':main()
