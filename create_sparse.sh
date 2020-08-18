#!/bin/bash

librispeech_subdir=/storageNVME/fei/data/speech/Librimix/LibriSpeech/test-clean/ # path to LibriSpeech test-clean
noise_dir=/storageNVME/fei/data/speech/Librimix/wham_noise/tt # path to test WHAM noises
metadata_dir=./metadata
out_dir=/storageNVME/fei/data/speech/Librimix/SparseLibriMix # output directory
stage=0
all_fs="8000 16000"
all_nspks="2 3"
all_overlap="0.2 0.4 0.6 0.8 1"

set -e
mkdir -p $out_dir

if [[ $stage -le 0 ]]; then
    for fs in $all_fs; do
      for n_speakers in $all_nspks; do
        for ovr_ratio in 0 $all_overlap; do
          echo "Making mixtures for ${n_speakers} speakers and overlap ${ovr_ratio}"
          python scripts/make_mixtures.py $metadata_dir/sparse_${n_speakers}_${ovr_ratio}/metadata.json \
            $librispeech_subdir $out_dir/wav${fs}/sparse_${n_speakers}_${ovr_ratio} --noise_dir $noise_dir --rate $fs
          done
      done
    done
fi