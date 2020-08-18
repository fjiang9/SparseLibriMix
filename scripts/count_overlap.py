import argparse
import json
import numpy as np
import random
from pathlib import Path
import os


n_src = 5
metadata_path = '../metadata/sparse_{}_0.2/metadata.json'.format(n_src)

with open(metadata_path, "r") as f:
    meta_files = json.load(f)

overlap_list = np.zeros([len(meta_files), n_src+1])
for i, mixture in enumerate(meta_files):
    # for each mixture
    # we sort all sub_utts by their sub_utt number
    sub_utts = []
    for k in mixture:
        if k.startswith("s"): # is a source
            for sub in mixture[k]:
                sub_utts.append(sub)
        if k == "noise":
            mix_length = mixture[k][0]['stop']

    sub_utts = sorted(sub_utts, key= lambda x : x["sub_utt_num"])
    c_speakers = [mixture[k][0]["spk_id"] for k in mixture.keys() if k.startswith("s")]

    fs = 16000
    count = np.zeros([n_src, int(fs*mix_length)], dtype=int)
    for sub_utt in sub_utts:
        st = int(sub_utt['start'] * fs)
        ed = int(sub_utt['stop'] * fs)
        count[int(sub_utt["source"][1:])-1, st:ed] = 1
    count = np.sum(count, axis=0)
    mix_len = len(count)
    # print(len(np.where(count == 0)[0])/mix_len)
    # print(len(np.where(count == 1)[0])/mix_len)
    # print(len(np.where(count == 2)[0])/mix_len)
    # print(len(np.where(count == 3)[0])/mix_len)
    overlap_list[i][0] = len(np.where(count == 0)[0])/mix_len
    for k in range(n_src):
        overlap_list[i][k+1] = len(np.where(count == k+1)[0])/mix_len


print(np.mean(overlap_list, axis=0))






