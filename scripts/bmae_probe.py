import sys; sys.path.insert(0,"src")
import numpy as np, glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode
from behavior_lab.models import get_model
ds=sys.argv[1]; kpdim=int(sys.argv[2])  # ds, kp-count
ck=sorted(glob.glob(f"/node_data/joon/outputs_bmae/{ds}/checkpoint-*.pth"))[-1]
kp=np.nan_to_num(np.load(f"outputs/{ds}/kp.npz")["keypoints"][:8000]).astype(np.float32)
gt=np.load(f"outputs/{ds}/labels.npy")[:8000]
if ds=="ntu": kpn=(kp-kp.mean((0,1),keepdims=True))/(kp.std((0,1),keepdims=True)+1e-6)
else: kpn=(kp-512.)/512.
m=get_model("behavemae", checkpoint_path=ck, dataset=ds)
W=90; accL={}
for i in range(0,len(kpn),W):
    if i+W>len(kpn): break
    h=m.encode_hierarchical(kpn[i:i+W], target_frames=W)
    for lv,e in h.items():
        e=np.asarray(e); accL.setdefault(lv,[]).append(e.reshape(-1,e.shape[-1]))
raw=kp.reshape(len(kp),-1)
best=(None,-1)
for lv in sorted(accL):
    E=np.concatenate(accL[lv]); nt=accL[lv][0].shape[0]; fpt=W//nt
    gtk=mode(gt[:len(E)*fpt].reshape(len(E),fpt),axis=1,keepdims=False).mode
    n=len(E); tr=int(n*0.7)
    if len(set(gtk[:tr].tolist()))<2: continue
    clf=LogisticRegression(max_iter=400).fit(E[:tr],gtk[:tr]); a=accuracy_score(gtk[tr:],clf.predict(E[tr:]))
    if a>best[1]: best=(lv,a)
# raw baseline at frame res
rawk=raw[:len(gt)]; ntr=int(len(rawk)*0.7)
clf=LogisticRegression(max_iter=400).fit(rawk[:ntr],gt[:ntr]); araw=accuracy_score(gt[ntr:],clf.predict(rawk[ntr:]))
print(f"PROBE {ds}: hBehaveMAE(trained,{best[0]}) acc={best[1]:.3f} | raw-pose acc={araw:.3f} | ckpt={ck.split(chr(47))[-1]}")
