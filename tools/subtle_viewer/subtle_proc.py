"""Subprocess worker (SUBTLE can crash the main process on macOS). usage:
  subtle_proc.py train <in.npz> <out.npz> <model.pkl> <seed|none> [clustering]
  subtle_proc.py map   <in.npz> <out.npz> <model.pkl>
in.npz = keypoints (concatenated (T, K, 3)) + lengths; out.npz = labels concatenated in the same order."""
import sys
import multiprocessing

import numpy as np

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.path.insert(0, "src")  # cwd = repo root
    from behavior_lab.models.discovery.subtle_wrapper import CLUSTERINGS, SUBTLE

    mode, src, out, pkl = sys.argv[1:5]
    d = np.load(src)
    seqs = np.split(d["keypoints"], np.cumsum(d["lengths"])[:-1])
    assert [len(s) for s in seqs] == d["lengths"].tolist()
    if mode == "train":
        seed = None if sys.argv[5] == "none" else int(sys.argv[5])
        model = SUBTLE(fps=20, seed=seed, clustering=sys.argv[6] if len(sys.argv) > 6 else CLUSTERINGS[0])
        raw = model.fit_raw(seqs)
        model._mapper.save(pkl)
        np.savez(out, embeddings=raw["embeddings"], subclusters=raw["subclusters"], superclusters=raw["superclusters"])
    else:
        model = SUBTLE(fps=20).load(pkl)
        data = model._mapper.run([model._preprocess(s) for s in seqs])  # one Data per recording
        np.savez(out, embeddings=np.concatenate([x.Z for x in data]), subclusters=np.concatenate([x.y for x in data]),
                 superclusters=np.concatenate([x.Y for x in data]))
    print("ok", mode)
