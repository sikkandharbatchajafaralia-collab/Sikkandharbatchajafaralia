"""Optional YOLO road-user detections -> boxes JSON consumable by the manifest loader.
This exports per-frame detections, not identity-linked tracks. Coordinates are normalized xyxy.
"""
import argparse,os,json,glob
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument('--frames',required=True); p.add_argument('--weights',default='yolov8n.pt'); p.add_argument('--out',required=True); p.add_argument('--conf',type=float,default=.25); a=p.parse_args()
model=YOLO(a.weights); files=sorted(glob.glob(os.path.join(a.frames,'*'))); out=[]
for i,path in enumerate(files):
 r=model(path,conf=a.conf,verbose=False)[0]; h,w=r.orig_shape; boxes=[]
 if r.boxes is not None:
  for b in r.boxes:
   cid=int(b.cls.item());
   if cid not in (0,1,2,3,5,7): continue # person, bicycle, car, motorcycle, bus, truck
   x1,y1,x2,y2=b.xyxy[0].tolist(); boxes.append([x1/w,y1/h,x2/w,y2/h,float(b.conf.item()),float(cid)])
 out.append({'frame':i,'boxes':boxes})
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True); json.dump(out,open(a.out,'w'),indent=2); print(f'Wrote {len(out)} frames: {a.out}')
