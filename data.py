"""Manifest-driven clip dataset. See README for the required CSV schema and label policy."""
import os, json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TrafficClipDataset(Dataset):
    def __init__(self, csv_path, split, clip_length=16, stride=8, image_size=224, class_to_id=None):
        self.df=pd.read_csv(csv_path).fillna(''); self.df=self.df[self.df.split.astype(str)==split].reset_index(drop=True)
        self.L=clip_length; self.stride=stride
        classes=sorted(self.df.event_class.astype(str).unique()); self.class_to_id=class_to_id or {c:i for i,c in enumerate(classes)}
        self.tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.items=[]
        for _,r in self.df.iterrows():
            if not os.path.isdir(str(r.frames_dir)): continue
            fs=sorted(os.path.join(str(r.frames_dir),f) for f in os.listdir(str(r.frames_dir)) if f.lower().endswith(('.jpg','.jpeg','.png')))
            if not fs: continue
            onset=int(r.onset_frame) if str(r.onset_frame).strip() else len(fs)
            for st in range(0,max(1,len(fs)-self.L+1),self.stride): self.items.append((r,fs,st,onset))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        r,fs,st,onset=self.items[i]; ids=[min(st+j,len(fs)-1) for j in range(self.L)]
        video=torch.stack([self.tf(Image.open(fs[j]).convert('RGB')) for j in ids])
        # 2-sec anticipation horizon at the configured 10 fps. Recompute if source fps differs.
        rel=onset-st; event=float(rel <= 20)
        nodes=torch.zeros(32,8); mask=torch.zeros(32,dtype=torch.bool)
        bp=str(r.get('boxes_json',''))
        if bp and os.path.isfile(bp):
            with open(bp,encoding='utf-8') as f: ann=json.load(f)
            amap={int(a['frame']):a.get('boxes',[]) for a in ann}; bb=[]
            for j in ids: bb.extend(amap.get(j,[]))
            bb=sorted(bb,key=lambda b:float(b[4]) if len(b)>4 else 0,reverse=True)[:32]
            for k,b in enumerate(bb):
                x1,y1,x2,y2=map(float,b[:4]); conf=float(b[4]) if len(b)>4 else 1.; typ=float(b[5]) if len(b)>5 else 0.
                nodes[k]=torch.tensor([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,conf,typ,0.,0.]); mask[k]=True
        sev=int(r.severity) if 'severity' in r and str(r.severity).strip() else -1
        return {'video':video,'event':torch.tensor(event,dtype=torch.float32),'event_class':torch.tensor(self.class_to_id.get(str(r.event_class),0)),
                'severity':torch.tensor(sev),'nodes':nodes,'node_mask':mask,'video_id':str(r.video_id),'start_frame':torch.tensor(st),'onset_frame':torch.tensor(onset)}
