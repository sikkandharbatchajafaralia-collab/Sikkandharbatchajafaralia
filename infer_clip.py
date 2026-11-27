import argparse,os,glob,json,yaml,torch
from PIL import Image
from torchvision import transforms
from src.engine import build_model
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--checkpoint',required=True); p.add_argument('--frames',required=True); p.add_argument('--out',default='prediction.json'); a=p.parse_args()
cfg=yaml.safe_load(open(a.config)); dev='cuda' if torch.cuda.is_available() else 'cpu'; ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); cm=ck['class_to_id']; m=build_model(cfg,len(cm)).to(dev); m.load_state_dict(ck['model']); m.eval()
files=sorted(glob.glob(os.path.join(a.frames,'*.jpg'))+glob.glob(os.path.join(a.frames,'*.png'))); L=cfg['data']['clip_length']; files=files[-L:]
if not files: raise SystemExit('No jpg/png frames found')
tf=transforms.Compose([transforms.Resize((cfg['data']['image_size'],cfg['data']['image_size'])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
x=torch.stack([tf(Image.open(f).convert('RGB')) for f in files]);
while len(x)<L: x=torch.cat([x[:1],x],0)
x=x[-L:].unsqueeze(0).to(dev); nodes=torch.zeros(1,32,8,device=dev); mask=torch.zeros(1,32,dtype=torch.bool,device=dev)
with torch.no_grad(): o=m(x,nodes,mask,cfg['model']['edge_threshold'])
p_evt=float(o['event_prob'].item()); unc=float(o['uncertainty'].item()); sev=int(o['severity_prob'].argmax(-1).item()); risk=p_evt*(sev/2+.25)*(1-unc)
t1,t2,t3=cfg['alerts']['thresholds'];
if risk<t1: alert='Normal'
elif risk<t2: alert='Monitoring'
elif risk<t3: alert='Operator Warning'
elif sev==2 and unc<=cfg['alerts']['max_uncertainty_for_emergency']: alert='Emergency Alert'
else: alert='Defer / Operator Verification'
res={'event_probability':p_evt,'uncertainty':unc,'severity_level_0_to_2':sev,'risk_score':risk,'alert':alert,'event_class_id':int(o['class_logits'].argmax(-1).item()),'class_id_to_name':{v:k for k,v in cm.items()},'note':'Single-clip inference; temporal risk smoothing and object responsibility localisation require sequential clips and detector tracks.'}
json.dump(res,open(a.out,'w'),indent=2); print(json.dumps(res,indent=2))
