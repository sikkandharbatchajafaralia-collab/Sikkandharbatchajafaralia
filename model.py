import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class GraphBlock(nn.Module):
    def __init__(self,d):
        super().__init__(); self.msg=nn.Sequential(nn.Linear(2*d+4,d),nn.ReLU(),nn.Linear(d,d)); self.norm=nn.LayerNorm(d)
    def forward(self,x,geom,mask,thr):
        xy=geom[...,:2]; dist=torch.cdist(xy,xy); valid=mask[:,:,None]&mask[:,None,:]
        adj=(torch.exp(-5*dist)>=thr)&valid
        eye=torch.eye(x.size(1),device=x.device,dtype=torch.bool)[None]; adj|=eye&valid
        xi=x[:,:,None,:].expand(-1,-1,x.size(1),-1); xj=x[:,None,:,:].expand(-1,x.size(1),-1,-1)
        delta=geom[:,:,None,:]-geom[:,None,:,:]; m=self.msg(torch.cat([xi,xj,delta],-1))*adj[:,:,:,None]
        agg=m.sum(2)/adj.sum(2,keepdim=True).clamp_min(1)
        return self.norm(x+agg)*mask[:,:,None]

class UTriSNet(nn.Module):
    def __init__(self,num_classes=13,hidden=256,heads=8,graph_layers=3,dropout=.2,pretrained=False):
        super().__init__(); net=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone=nn.Sequential(*list(net.children())[:-2]); self.pool=nn.AdaptiveAvgPool2d(1); self.proj=nn.Linear(512,hidden)
        self.motion=nn.Linear(hidden,hidden); self.gate=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Sigmoid())
        enc=nn.TransformerEncoderLayer(hidden,heads,hidden*4,dropout,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(enc,2)
        self.node=nn.Sequential(nn.Linear(8,hidden),nn.ReLU(),nn.Linear(hidden,hidden)); self.graph=nn.ModuleList([GraphBlock(hidden) for _ in range(graph_layers)])
        self.pool_score=nn.Linear(hidden,1); self.fuse=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.evidence=nn.Linear(hidden,2); self.classifier=nn.Linear(hidden,num_classes); self.severity=nn.Linear(hidden,3); self.trend=nn.Linear(hidden,4)
        self.fuzzy_centres=nn.Parameter(torch.tensor([.2,.5,.8])); self.fuzzy_widths=nn.Parameter(torch.tensor([.2,.2,.2])); self.rule_weights=nn.Parameter(torch.zeros(3))
    def forward(self,video,nodes,node_mask,edge_threshold=.5):
        b,t,c,h,w=video.shape; f=self.backbone(video.reshape(b*t,c,h,w)); f=self.pool(f).flatten(1); app=self.proj(f).reshape(b,t,-1)
        mot=torch.zeros_like(app); mot[:,1:]=app[:,1:]-app[:,:-1]; g=self.gate(torch.cat([app,mot],-1)); temporal=self.temporal(g*app+(1-g)*self.motion(mot)).mean(1)
        n=self.node(nodes); geom=nodes[...,:4]
        for layer in self.graph: n=layer(n,geom,node_mask,edge_threshold)
        score=self.pool_score(n).squeeze(-1).masked_fill(~node_mask,-1e4); w=score.softmax(1); pooled=(w[:,:,None]*n).sum(1)
        pooled=torch.where(node_mask.any(1)[:,None],pooled,torch.zeros_like(pooled)); z=self.fuse(torch.cat([temporal,pooled],-1))
        ev=torch.nn.functional.softplus(self.evidence(z))+1e-5; alpha=ev+1; p=alpha/alpha.sum(-1,keepdim=True); u=(2/alpha.sum(-1)).clamp(0,1)
        sev_logits=self.severity(z); sev_neural=sev_logits.softmax(-1)
        # Interval type-2-inspired Gaussian footprint; label as approximation unless full type-reduction is implemented.
        x=torch.sigmoid(sev_logits.mean(-1,keepdim=True)); width=self.fuzzy_widths.abs().clamp(.05,.5)
        mu=torch.exp(-.5*((x-self.fuzzy_centres[None,:])/width[None,:])**2); lo=.85*mu; hi=(mu+.15).clamp(max=1); mu2=(lo+hi)/2
        fs=(mu2*self.rule_weights.softmax(0)).sum(-1); fd=torch.stack([1-fs,(1-(fs-.5).abs()*2).clamp(0,1),fs],-1); fd=fd/fd.sum(-1,keepdim=True).clamp_min(1e-6)
        sev=(1-u[:,None])*sev_neural+u[:,None]*fd
        return {'event_prob':p[:,1],'uncertainty':u,'evidence':ev,'class_logits':self.classifier(z),'severity_logits':sev_logits,'severity_prob':sev,'trend_logits':self.trend(z),'embedding':z,'node_attention':w}
