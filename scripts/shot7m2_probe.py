import sys; sys.path.insert(0,"src")
# 🔴 260814: shot7m2 데이터(3.4G)와 outputs_bmae 체크포인트는 gpu03 정리로 **양쪽 다 소실**됐다.
# 가중치 checkpoints/hBehaveMAE_Shot7M2.pth 는 있으나 입력 데이터가 없어 현재 실행 불가.
# 재취득 후 아래 env 로 경로를 주입할 것 (기본값은 죽은 gpu03 경로가 아니라 빈 값 → 즉시 실패).
import os as _os
BMAE_OUT = _os.environ.get("BL_BMAE_OUT", "")
SHOT7M2 = _os.environ.get("BL_SHOT7M2", "")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from behavior_lab.models import get_model
SPLIT=[0,1,6,11,2,3,4,5,7,8,9,10,14,19,20,21,15,16,17,18,22,23,24,25]
P=np.load(f"{SHOT7M2}/test/test_dictionary_poses.npy",allow_pickle=True).item()["sequences"]
eps=P[list(P.keys())[0]]
L=np.load(f"{SHOT7M2}/test/benchmark_labels.npy",allow_pickle=True).item()
la=np.asarray(L["label_array"]); fnm=L["frame_number_map"]
kps=[]; gts=[]; tot=0
for e in list(eps.keys()):
    if e not in fnm: continue
    kp=np.asarray(eps[e],dtype=np.float32)[:,0,:,:]; st,en=fnm[e]; le=la[:,st:en]
    if le.shape[1]!=len(kp): continue
    kps.append(kp); gts.append(np.argmax(le[5:17],axis=0)); tot+=len(kp)
    if tot>=7200: break
kp=np.nan_to_num(np.concatenate(kps)[:7200]); gt=np.concatenate(gts)[:7200]
kp24=kp[:,SPLIT,:]; kp24=(kp24-kp24.mean((0,1),keepdims=True))/(kp24.std((0,1),keepdims=True)+1e-6)
m=get_model("behavemae", checkpoint_path="checkpoints/hBehaveMAE_Shot7M2.pth", dataset="shot7m2")
W=400; acc7=[]
for i in range(0,len(kp24),W):
    if i+W>len(kp24): break
    h=m.encode_hierarchical(kp24[i:i+W],target_frames=W)
    e=np.asarray(h["level_7"]); acc7.append(e.reshape(-1,e.shape[-1]))
E=np.concatenate(acc7)  # (n_tok, dim)
ntok_win=acc7[0].shape[0]; fpt=W//ntok_win  # frames per token
# align GT to token resolution (majority per token window)
gt_tok=gt[:len(E)*fpt].reshape(len(E),fpt)
from scipy.stats import mode
gt_tok=mode(gt_tok,axis=1,keepdims=False).mode
# raw pose baseline at token resolution (mean-pool)
raw=kp24.reshape(len(kp24),-1)[:len(E)*fpt].reshape(len(E),fpt,-1).mean(1)
n=len(E); tr=int(n*0.7)
def probe(X):
    clf=LogisticRegression(max_iter=500,C=1.0).fit(X[:tr],gt_tok[:tr])
    p=clf.predict(X[tr:]); return accuracy_score(gt_tok[tr:],p), f1_score(gt_tok[tr:],p,average="macro")
a1,f1=probe(E); a2,f2=probe(raw)
print(f"PROBE hBehaveMAE-L7: acc={a1:.3f} f1={f1:.3f} | raw-pose: acc={a2:.3f} f1={f2:.3f}")
