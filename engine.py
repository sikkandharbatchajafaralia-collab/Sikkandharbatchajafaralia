import os, json, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error
from .data import TrafficClipDataset
from .model import UTriSNet
from .utils import collate_video_batch, ensure_dir, seed_everything

def make_loaders(cfg, manifest, external=False, class_to_id=None):
    d=cfg['data']; common=dict(clip_length=d['clip_length'],stride=d['stride'],image_size=d['image_size'],class_to_id=class_to_id)
    train=TrafficClipDataset(manifest,'train',**common) if not external else None
    val=TrafficClipDataset(manifest,'val',**common) if not external else None
    test=TrafficClipDataset(manifest,'test',**common)
    kw=dict(batch_size=d['batch_size'],num_workers=d['num_workers'],collate_fn=collate_video_batch,pin_memory=torch.cuda.is_available())
    return (DataLoader(train,shuffle=True,**kw) if train else None, DataLoader(val,shuffle=False,**kw) if val else None, DataLoader(test,shuffle=False,**kw),train.class_to_id if train else class_to_id)

def build_model(cfg,nclasses):
    m=cfg['model']; return UTriSNet(nclasses,m['hidden_dim'],m['attention_heads'],m['graph_layers'],m['dropout'],m.get('pretrained',False))

def losses(out,b,cfg):
    y=b['event'].to(out['event_prob'].device); ce=nn.functional.binary_cross_entropy(out['event_prob'].clamp(1e-6,1-1e-6),y)
    yc=b['event_class'].to(y.device); lc=nn.functional.cross_entropy(out['class_logits'],yc)
    sev=b['severity'].to(y.device); valid=sev>=0
    ls=nn.functional.cross_entropy(out['severity_logits'][valid],sev[valid]) if valid.any() else out['severity_logits'].sum()*0
    # Brier score is used as a differentiable calibration regularizer, not as proof of calibration.
    cal=((out['event_prob']-y)**2).mean()
    tr=out['trend_logits'].sum()*0 # trend labels require a separately curated target column
    t=cfg['train']; total=t['lambda_event']*ce+t['lambda_class']*lc+t['lambda_severity']*ls+t['lambda_calibration']*cal+t['lambda_trend']*tr
    return total,{'event_bce':ce.item(),'class_ce':lc.item(),'severity_ce':ls.item(),'brier':cal.item()}

@torch.no_grad()
def run_eval(model,loader,device,threshold=.5):
    model.eval(); ys=[]; ps=[]; cls=[]; ctrue=[]; sevp=[]; sevt=[]; vids=[]; starts=[]; onsets=[]
    for b in loader:
        x=b['video'].to(device); nodes=b['nodes'].to(device); mask=b['node_mask'].to(device)
        o=model(x,nodes,mask); ys.extend(b['event'].numpy().tolist()); ps.extend(o['event_prob'].cpu().numpy().tolist())
        cls.extend(o['class_logits'].argmax(-1).cpu().numpy().tolist()); ctrue.extend(b['event_class'].numpy().tolist())
        sevp.extend(o['severity_prob'].argmax(-1).cpu().numpy().tolist()); sevt.extend(b['severity'].numpy().tolist())
        vids.extend(b['video_id']); starts.extend(b['start_frame'].numpy().tolist()); onsets.extend(b['onset_frame'].numpy().tolist())
    pred=(np.array(ps)>=threshold).astype(int); y=np.array(ys).astype(int)
    metrics={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
             'precision':float(precision_recall_fscore_support(y,pred,average='binary',zero_division=0)[0]),
             'recall':float(precision_recall_fscore_support(y,pred,average='binary',zero_division=0)[1]),
             'f1':float(precision_recall_fscore_support(y,pred,average='binary',zero_division=0)[2]),
             'mcc':matthews_corrcoef(y,pred),'brier':float(np.mean((np.array(ps)-y)**2)),
             'auc':float(roc_auc_score(y,ps)) if len(set(y))>1 else None,
             'confusion_matrix':confusion_matrix(y,pred).tolist()}
    sv=np.array(sevt); sp=np.array(sevp); valid=sv>=0
    metrics['severity']={'count':int(valid.sum()),'macro_f1':float(precision_recall_fscore_support(sv[valid],sp[valid],average='macro',zero_division=0)[2]) if valid.any() else None,
                         'mae':float(mean_absolute_error(sv[valid],sp[valid])) if valid.any() else None,
                         'rmse':float(math.sqrt(mean_squared_error(sv[valid],sp[valid]))) if valid.any() else None}
    metrics['samples']=len(y)
    return metrics

def train_one(cfg,manifest,seed,output_dir):
    seed_everything(seed); device='cuda' if torch.cuda.is_available() else 'cpu'; tr,va,te,class_map=make_loaders(cfg,manifest)
    model=build_model(cfg,len(class_map)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay'])
    best=float('inf'); patience=0; ensure_dir(output_dir); history=[]
    for epoch in range(cfg['train']['epochs']):
        model.train(); losses_epoch=[]
        for b in tr:
            opt.zero_grad(set_to_none=True); o=model(b['video'].to(device),b['nodes'].to(device),b['node_mask'].to(device),cfg['model']['edge_threshold'])
            loss,parts=losses(o,b,cfg); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses_epoch.append(loss.item())
        model.eval(); vloss=[]
        with torch.no_grad():
            for b in va:
                o=model(b['video'].to(device),b['nodes'].to(device),b['node_mask'].to(device),cfg['model']['edge_threshold']); l,_=losses(o,b,cfg); vloss.append(l.item())
        train_loss=float(np.mean(losses_epoch)); val_loss=float(np.mean(vloss)); history.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':val_loss})
        print(f'seed={seed} epoch={epoch+1} train={train_loss:.4f} val={val_loss:.4f}')
        if val_loss<best:
            best=val_loss; patience=0; torch.save({'model':model.state_dict(),'class_to_id':class_map,'config':cfg,'seed':seed},os.path.join(output_dir,'best.pt'))
        else:
            patience+=1
            if patience>=cfg['train']['patience']: break
    with open(os.path.join(output_dir,'history.json'),'w') as f: json.dump(history,f,indent=2)
    ck=torch.load(os.path.join(output_dir,'best.pt'),map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    metrics=run_eval(model,te,device); json.dump(metrics,open(os.path.join(output_dir,'test_metrics.json'),'w'),indent=2)
    return metrics
