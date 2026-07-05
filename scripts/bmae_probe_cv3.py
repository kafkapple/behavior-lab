import sys; sys.path.insert(0,"src")
import numpy as np, glob, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.stats import mode
from behavior_lab.models import get_model
SPLIT=[0,1,6,11,2,3,4,5,7,8,9,10,14,19,20,21,15,16,17,18,22,23,24,25]
def load(ds):
    if ds=="shot7m2":
        P=np.load("/node_data/joon/data/shot7m2/test/test_dictionary_poses.npy",allow_pickle=True).item()["sequences"]
        eps=P[list(P.keys())[0]]; L=np.load("/node_data/joon/data/shot7m2/test/benchmark_labels.npy",allow_pickle=True).item()
        la=np.asarray(L["label_array"]); fnm=L["frame_number_map"]; kps=[];gts=[];tot=0
        for e in list(eps.keys()):
            if e not in fnm: continue
            kp=np.asarray(eps[e],dtype=np.float32)[:,0,:,:]; st,en=fnm[e]; le=la[:,st:en]
            if le.shape[1]!=len(kp): continue
            kps.append(kp[:,SPLIT,:]); gts.append(np.argmax(le[5:17],axis=0)); tot+=len(kp)
            if tot>=7200: break
        kp=np.nan_to_num(np.concatenate(kps)[:7200]); gt=np.concatenate(gts)[:7200]
        kp=(kp-kp.mean((0,1),keepdims=True))/(kp.std((0,1),keepdims=True)+1e-6)
        return kp,gt,"checkpoints/hBehaveMAE_Shot7M2.pth","shot7m2",400
    kp=np.nan_to_num(np.load(f"outputs/{ds}/kp.npz")["keypoints"][:8000]).astype(np.float32)
    gt=np.load(f"outputs/{ds}/labels.npy")[:8000]; ck=sorted(glob.glob(f"/node_data/joon/outputs_bmae/{ds}/checkpoint-*.pth"))[-1]
    if ds=="ntu": kp=(kp-kp.mean((0,1),keepdims=True))/(kp.std((0,1),keepdims=True)+1e-6)
    else: kp=(kp-512.)/512.
    return kp,gt,ck,ds,90
def cv(X,y):
    skf=StratifiedKFold(3,shuffle=True,random_state=0); a=[];f=[];mp=[]; cls=np.unique(y)
    for tr,te in skf.split(X,y):
        c=LogisticRegression(max_iter=500).fit(X[tr],y[tr]); p=c.predict(X[te])
        a.append(accuracy_score(y[te],p)); f.append(f1_score(y[te],p,average="macro"))
        try: mp.append(average_precision_score(label_binarize(y[te],classes=cls),c.predict_proba(X[te]),average="macro"))
        except: mp.append(np.nan)
    return np.mean(a),np.mean(f),np.std(f),np.nanmean(mp)
def filt(E,Rw,gk):
    kc=np.array([c for c in np.unique(gk) if np.sum(gk==c)>=6])
    if len(kc)<2: return None
    mask=np.isin(gk,kc); rm={c:i for i,c in enumerate(kc)}
    g=np.array([rm[c] for c in gk[mask]]); return E[mask],Rw[mask],g,len(kc)
for ds in ["shot7m2","ntu","calms21"]:
    kp,gt,ck,cfg,W=load(ds); m=get_model("behavemae",checkpoint_path=ck,dataset=cfg); acc={}
    for i in range(0,len(kp),W):
        if i+W>len(kp): break
        h=m.encode_hierarchical(kp[i:i+W],target_frames=W)
        for lv,e in h.items():
            e=np.asarray(e); acc.setdefault(lv,[]).append(e.reshape(-1,e.shape[-1]))
    rawf=kp.reshape(len(kp),-1); best=None
    for lv in sorted(acc):
        E=np.concatenate(acc[lv]); nt=acc[lv][0].shape[0]; fpt=W//nt
        if fpt<1: continue
        gk=mode(gt[:len(E)*fpt].reshape(len(E),fpt),axis=1,keepdims=False).mode
        Rw=rawf[:len(E)*fpt].reshape(len(E),fpt,-1).mean(1)
        ff=filt(E,Rw,gk)
        if ff is None: continue
        Ef,Rwf,g,nc=ff; rH=cv(Ef,g)
        if best is None or rH[1]>best[1]: best=(lv,rH[1],rH,cv(Rwf,g),len(g),nc)
    if best is None: print(f"CV {ds}: no valid level"); continue
    lv,_,rH,rR,n,nc=best
    print(f"FINALCV {ds}({n}tok,{nc}cls,{lv}): hBehaveMAE acc={rH[0]:.3f} f1={rH[1]:.3f}±{rH[2]:.3f} mAP={rH[3]:.3f} | raw acc={rR[0]:.3f} f1={rR[1]:.3f} mAP={rR[3]:.3f}")
