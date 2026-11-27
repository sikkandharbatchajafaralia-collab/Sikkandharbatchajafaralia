import argparse, yaml, os, json
from src.engine import train_one
from src.utils import ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--manifest'); p.add_argument('--seed',type=int); a=p.parse_args()
cfg=yaml.safe_load(open(a.config)); manifest=a.manifest or cfg['paths']['train_manifest']; root=cfg['paths']['output_dir']; ensure_dir(root)
seeds=[a.seed] if a.seed is not None else cfg['train']['seeds']; results=[]
for s in seeds:
    out=os.path.join(root,f'seed_{s}'); results.append({'seed':s,'metrics':train_one(cfg,manifest,s,out)})
json.dump(results,open(os.path.join(root,'all_seed_results.json'),'w'),indent=2)
