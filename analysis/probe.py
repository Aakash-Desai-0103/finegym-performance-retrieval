"""Day 3 checks on frozen VideoMAE features: vault identity, scene bias, stage separability,
and a first ridge-regression score prediction (E / Total / D) with overall and within-vault SRCC.

Usage (files from Drive: capstone/features/):
    python analysis/probe.py --feats videomae_k400_all.npz --index videomae_k400_index.csv
Writes probe_results.csv next to --out (default: current folder).
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="videomae_k400_all.npz")
ap.add_argument("--index", default="videomae_k400_index.csv")
ap.add_argument("--out", default=".")
args = ap.parse_args()
import numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
F=np.load(args.feats,allow_pickle=True); m=pd.read_csv(args.index,dtype={'vault_code':str})
W=F['whole'].astype(np.float32); S=F['stages'].astype(np.float32); hs=F['has_stages']
tr=(m.split=='train').values; te=(m.split=='test').values
def norm(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
code=m.vault_code.fillna('NA').values; vid=m.video_id.values

print('== 1. vault identity ==')
base=(code[te]==pd.Series(code[tr]).mode()[0]).mean()
for name,X in [('raw',W),('centered',W-W[tr].mean(0)),('standardized',StandardScaler().fit(W[tr]).transform(W))]:
    sim=norm(X[te])@norm(X[tr]).T
    acc1=(code[tr][sim.argmax(1)]==code[te]).mean()
    top5=np.argsort(-sim,1)[:,:5]
    acc5=np.mean([pd.Series(code[tr][r]).mode()[0]==c for r,c in zip(top5,code[te])])
    print(f'  {name:12s} 1-NN {acc1:.3f}  5-NN vote {acc5:.3f}  (majority {base:.3f})')
Xs=StandardScaler().fit(W[tr]).transform(W)
ok=np.isin(code[te],np.unique(code[tr]))
clf=LogisticRegression(C=0.05,max_iter=3000).fit(Xs[tr],code[tr])
print(f'  linear probe (logistic) acc {(clf.predict(Xs[te])==code[te]).mean():.3f}; on codes seen in train {(clf.predict(Xs[te][ok])==code[te][ok]).mean():.3f} ({ok.sum()} clips)')

print('== 2. scene/video bias (train only) ==')
Xc=W[tr]-W[tr].mean(0); sim=norm(Xc)@norm(Xc).T; np.fill_diagonal(sim,-9)
nn=sim.argmax(1); same_vid=(vid[tr][nn]==vid[tr]).mean(); same_code=(code[tr][nn]==code[tr]).mean()
v=vid[tr]; chance=np.mean([(v==x).sum()-1 for x in v])/(len(v)-1)
c=code[tr]; chance_c=np.mean([(c==x).sum()-1 for x in c])/(len(c)-1)
print(f'  1-NN is from SAME source video: {same_vid:.3f} (chance {chance:.3f})')
print(f'  1-NN has same vault code:       {same_code:.3f} (chance {chance_c:.3f})')

print('== 3. stages ==')
Sk=S[hs]; mk=m[hs].reset_index(drop=True); trk=(mk.split=='train').values; tek=~trk
Xst=np.concatenate([Sk[:,i] for i in range(3)]); yst=np.repeat([0,1,2],len(Sk)); spl=np.tile(trk,3)
sc=StandardScaler().fit(Xst[spl]); c3=LogisticRegression(C=0.05,max_iter=3000).fit(sc.transform(Xst[spl]),yst[spl])
print(f'  stage classifier (approach/flight/landing) test acc {(c3.predict(sc.transform(Xst[~spl]))==yst[~spl]).mean():.3f} (chance 0.333)')
mu=Xst[spl].mean(0); Sc=norm(Sk-mu)
within=np.mean([np.sum(Sc[:,i]*Sc[:,j],1).mean() for i,j in [(0,1),(1,2),(0,2)]])
across=np.mean([(Sc[:,k]@Sc[:,k].T)[np.triu_indices(len(Sc),1)].mean() for k in range(3)])
print(f'  centered cosine: different stages same clip {within:.3f} | same stage across clips {across:.3f}')

print('== 4. score prediction preview (ridge, train->test) ==')
def ridge(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr); r=RidgeCV(alphas=np.logspace(0,5,21)).fit(sc.transform(Xtr),ytr); return r.predict(sc.transform(Xte)),r.alpha_
def within_srcc(p,y,codes):
    rs=[];ws=[]
    for c in np.unique(codes):
        k=codes==c
        if k.sum()>=5 and np.std(y[k])>0: rs.append(spearmanr(p[k],y[k])[0]); ws.append(k.sum())
    return np.average(rs,weights=ws), len(rs), sum(ws)
def rl2(p,y): r=y.max()-y.min(); return 100*np.mean(((p-y)/r)**2)
Dv=m.D.values.reshape(-1,1)
stage_cat=np.where(hs[:,None],S.reshape(len(S),-1),np.tile(W,3))  # fallback to whole if no stages
rows=[]
for target in ['E','Total','D']:
    y=m[target].values
    for fname,X in [('mean of train',None),('D only (difficulty)',Dv),('VideoMAE whole',W),('VideoMAE whole+stages',np.hstack([W,stage_cat])),('VideoMAE whole + D',np.hstack([W,Dv]))]:
        if target=='D' and 'D' in fname: continue
        if X is None: p=np.full(te.sum(),y[tr].mean()); a='-'
        else: p,a=ridge(X[tr],y[tr],X[te])
        s=spearmanr(p,y[te])[0] if np.std(p)>0 else np.nan
        w=within_srcc(p,y[te],code[te]) if np.std(p)>0 else (np.nan,0,0)
        rows.append([target,fname,s,w[0],np.mean((p-y[te])**2),rl2(p,y[te]),a])
R=pd.DataFrame(rows,columns=['target','features','SRCC','within-vault SRCC','MSE','R-l2 (x100)','ridge alpha'])
pd.set_option('display.width',200); print(R.round(3).to_string(index=False))
print('within-vault: test vault codes with >=5 clips:', within_srcc(np.random.rand(te.sum()),m.E.values[te],code[te])[1:])
import os; R.to_csv(os.path.join(args.out,'probe_results.csv'),index=False)
