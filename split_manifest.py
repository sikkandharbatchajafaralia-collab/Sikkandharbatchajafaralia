import argparse,pandas as pd
from sklearn.model_selection import train_test_split
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--train',type=float,default=.70); p.add_argument('--val',type=float,default=.15); a=p.parse_args()
df=pd.read_csv(a.input).fillna(''); ids=df.video_id.astype(str).unique(); tr,rest=train_test_split(ids,test_size=1-a.train,random_state=a.seed); va,te=train_test_split(rest,test_size=1-a.val/(1-a.train),random_state=a.seed)
s={**{x:'train' for x in tr},**{x:'val' for x in va},**{x:'test' for x in te}}; df['split']=df.video_id.astype(str).map(s); df.to_csv(a.output,index=False)
assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te)); print(df.groupby('split').video_id.nunique())
