"""Day 3 robustness: E prediction per stage (whole / approach / flight / landing / combos) with
95% bootstrap CIs over test VIDEOS, plus a permutation test (shuffled training labels).

Usage:
    python analysis/probe2.py --feats videomae_k400_all.npz --index videomae_k400_index.csv
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
from sklearn.linear_model import RidgeCV
warnings.filterwarnings('ignore')
F=np.load(args.feats,allow_pickle=True); m=pd.read_csv(args.index,dtype={'vault_code':str})
W=F['whole'].astype(np.float32); S=F['stages'].astype(np.float32); hs=F['has_stages']
code=m.vault_code.fillna('NA').values; vid=m.video_id.values
tr=(m.split=='train').values; te=(m.split=='test').values; y=m.E.values
stage_cat=np.where(hs[:,None],S.reshape(len(S),-1),np.tile(W,3))
def ridge(X):
    sc=StandardScaler().fit(X[tr]); r=RidgeCV(alphas=np.logspace(0,5,21)).fit(sc.transform(X[tr]),y[tr]); return r.predict(sc.transform(X[te]))
def within(p,yy,cc):
    rs=[];ws=[]
    for c in np.unique(cc):
        k=cc==c
        if k.sum()>=5 and np.std(yy[k])>0: rs.append(spearmanr(p[k],yy[k])[0]); ws.append(k.sum())
    return np.average(rs,weights=ws) if rs else np.nan
sets={'whole':W,'approach':np.where(hs[:,None],S[:,0],W),'flight':np.where(hs[:,None],S[:,1],W),'landing':np.where(hs[:,None],S[:,2],W),
      'stages only (3)':stage_cat,'whole+stages':np.hstack([W,stage_cat])}
rng=np.random.default_rng(0); tv=np.unique(vid[te]); yt=y[te]; ct=code[te]; vt=vid[te]
print(f'{"features":16s} SRCC [95% CI by video]      within-vault SRCC [95% CI]')
for n,X in sets.items():
    p=ridge(X); s=spearmanr(p,yt)[0]; w=within(p,yt,ct)
    bs=[];bw=[]
    for _ in range(1000):
        pick=rng.choice(tv,len(tv)); idx=np.concatenate([np.where(vt==v)[0] for v in pick])
        bs.append(spearmanr(p[idx],yt[idx])[0]); bw.append(within(p[idx],yt[idx],ct[idx]))
    print(f'{n:16s} {s:.3f} [{np.nanpercentile(bs,2.5):.2f}, {np.nanpercentile(bs,97.5):.2f}]      {w:.3f} [{np.nanpercentile(bw,2.5):.2f}, {np.nanpercentile(bw,97.5):.2f}]')
# permutation null for whole+stages within-vault: shuffle train labels
X=sets['whole+stages']; null=[]
for i in range(200):
    yp=y.copy(); yp[tr]=rng.permutation(y[tr])
    sc=StandardScaler().fit(X[tr]); r=RidgeCV(alphas=np.logspace(0,5,21)).fit(sc.transform(X[tr]),yp[tr]); p=r.predict(sc.transform(X[te]))
    null.append(within(p,yt,ct))
w0=within(ridge(X),yt,ct); print(f'permutation test (whole+stages, within-vault): observed {w0:.3f}, null 95th pct {np.percentile(null,95):.3f}, p={np.mean(np.array(null)>=w0):.3f}')
