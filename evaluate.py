import argparse,yaml,torch,json,os
from torch.utils.data import DataLoader
from src.data import TrafficClipDataset
from src.engine import build_model,run_eval
from src.utils import collate_video_batch
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--split',default='test'); p.add_argument('--out',default='evaluation.json'); a=p.parse_args()
cfg=yaml.safe_load(open(a.config)); dev='cuda' if torch.cuda.is_available() else 'cpu'; ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); cm=ck['class_to_id']
ds=TrafficClipDataset(a.manifest,a.split,clip_length=cfg['data']['clip_length'],stride=cfg['data']['stride'],image_size=cfg['data']['image_size'],class_to_id=cm)
loader=DataLoader(ds,batch_size=cfg['data']['batch_size'],shuffle=False,num_workers=cfg['data']['num_workers'],collate_fn=collate_video_batch)
m=build_model(cfg,len(cm)).to(dev); m.load_state_dict(ck['model']); res=run_eval(m,loader,dev); json.dump(res,open(a.out,'w'),indent=2); print(json.dumps(res,indent=2))
