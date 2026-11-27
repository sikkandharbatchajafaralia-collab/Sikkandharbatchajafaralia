"""Build a metadata-level DoTA CSV from metadata_train.json/metadata_val.json and extracted frame folders.
Usage: python scripts/make_dota_manifest.py --metadata data/DoTA/metadata_train.json data/DoTA/metadata_val.json --frames data/DoTA/frames --out manifests/dota_raw.csv
Then create a video-level split with scripts/split_manifest.py.
"""
import argparse,json,os,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--metadata',nargs='+',required=True); p.add_argument('--frames',required=True); p.add_argument('--out',required=True); a=p.parse_args(); rows=[]
for mf in a.metadata:
 d=json.load(open(mf,encoding='utf-8'))
 for vid,v in d.items():
  folder=os.path.join(a.frames,str(vid));
  if not os.path.isdir(folder): continue
  rows.append({'video_id':str(vid),'frames_dir':folder,'onset_frame':int(v.get('anomaly_start',0)), 'event_class':str(v.get('anomaly_class','unknown')),'source_fps':10,'native_subset':v.get('subset','')})
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False); print(f'wrote {len(rows)} videos to {a.out}')
