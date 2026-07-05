import sys,json,numpy as np
J=sys.argv[1]
d=json.load(open(J)); a0=d["annotator-id_0"]; keys=list(a0.keys())[:3]
kps=[]; labs=[]
for k in keys:
    s=a0[k]; kp=np.array(s["keypoints"],dtype=np.float32).transpose(0,1,3,2).reshape(-1,14,2)
    kps.append(kp); labs.append(np.array(s["annotations"]))
kp=np.nan_to_num(np.concatenate(kps)[:8000]); lab=np.concatenate(labs)[:8000].astype(int)
np.savez("outputs/calms21/kp.npz", keypoints=kp); np.save("outputs/calms21/labels.npy", lab)
print("PREP", kp.shape, "labelcounts", np.bincount(lab).tolist())
