import argparse
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
import pyloudnorm
from scipy.signal import resample_poly
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--json", default="/home/fei/SparseLibriMix/metadata/sparse_5_0.2/metadata.json")  # choose n_speakers and overlap_ratio
parser.add_argument("--librispeech_dir", default="/storageNVME/fei/data/speech/Librimix/LibriSpeech/test-clean/")
parser.add_argument('--out_dir', help='output data dir of mixture', default="/storageNVME/fei/data/speech/Librimix/SparseLibriMix/wav8000/sparse_5_0.2")
parser.add_argument("--noise_dir", type=str, default="/storageNVME/fei/data/speech/Librimix/wham_noise/tt")
parser.add_argument('--rate', type=int, default=8000,
                    help='sampling rate')

# parser.add_argument("json")
# parser.add_argument("librispeech_dir")
# parser.add_argument('out_dir',help='output data dir of mixture')
# parser.add_argument("--noise_dir", type=str, default="")
# parser.add_argument('--rate', type=int, default=16000,
#                     help='sampling rate')

def main(args):
    if not args.noise_dir:
        print("Generating only clean version")

    with open(args.json, "r") as f:
        total_meta = json.load(f)

    # Dictionary that will contain all metadata
    md_dic = {}
    # Create Dataframes
    dir_name = args.json.split('/')[-2]
    n_src = int(dir_name.split('_')[1])
    print(n_src, dir_name)
    md_dic[f'mixture_{dir_name}_mix_clean'] = create_empty_mixture_md(n_src, 'mix_clean')
    if args.noise_dir:
        md_dic[f'mixture_{dir_name}_mix_noisy'] = create_empty_mixture_md(n_src, 'mix_noisy')

    for mix in tqdm(total_meta):
        # filename = mix["mixture_name"]
        sources_list = [x for x in mix.keys() if x != "mixture_name"]

        sources = {}
        utt_id_list = ['' for i in range(n_src)]
        maxlength = 0
        for source in sources_list:
            # read file optional resample it
            source_utts = []
            for utt in mix[source]:
                if utt["source"] != "noise": # speech file
                    utt["file"] = os.path.join(args.librispeech_dir, utt["file"])
                else:
                    if args.noise_dir:
                        utt["file"] = os.path.join(args.noise_dir, utt["file"])
                    else:
                        continue

                utt_fs = sf.SoundFile(utt["file"]).samplerate
                audio, fs = sf.read(utt["file"], start=int(utt["orig_start"]*utt_fs),
                                stop=int(utt["orig_stop"]*utt_fs))

                #assert len(audio.shape) == 1, "we currently not support multichannel"
                if len(audio.shape) > 1:
                    audio = audio[:, utt["channel"]] #TODO
                audio = audio - np.mean(audio) # zero mean cos librispeech is messed up sometimes
                audio = resample_and_norm(audio, fs, args.rate, utt["lvl"])
                audio = np.pad(audio, (int(utt["start"]*args.rate), 0), "constant") # pad the beginning
                source_utts.append(audio)
                maxlength = max(len(audio), maxlength)
                if source != "noise":
                    utt_id = utt["utt_id"]
            sources[source] = source_utts
            if source != "noise":
                utt_id_list[int(source[1:])-1] = utt_id

        filename = '_'.join(utt_id_list)

        # pad everything to same length
        for s in sources.keys():
            for i in range(len(sources[s])):
                tmp = sources[s][i]
                sources[s][i] = np.pad(tmp,  (0, maxlength-len(tmp)), 'constant')

        # mix n sum
        tot_mixture = None
        abs_source_path_list = ['' for i in range(n_src)]
        for indx, s in enumerate(sources.keys()):
            if s == "noise":
                continue
            source_mix = np.sum(sources[s], 0)
            os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)
            sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
            if indx == 0:
                tot_mixture = source_mix
            else:
                tot_mixture += source_mix
            abs_source_path_list[int(s[1:])-1] = os.path.join(args.out_dir, s, filename + ".wav")

        os.makedirs(os.path.join(args.out_dir, "mix_clean"), exist_ok=True)
        sf.write(os.path.join(args.out_dir, "mix_clean", filename + ".wav"), tot_mixture, args.rate)

        add_to_mixture_metadata(md_dic[f'mixture_{dir_name}_mix_clean'], filename,
                                os.path.join(args.out_dir, "mix_clean", filename + ".wav"),
                                abs_source_path_list,
                                maxlength, "mix_clean")

        if args.noise_dir:
            s = "noise"
            source_mix = np.sum(sources[s], 0)
            os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)
            sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
            tot_mixture += source_mix
            os.makedirs(os.path.join(args.out_dir, "mix_noisy"), exist_ok=True)
            sf.write(os.path.join(args.out_dir, "mix_noisy", filename + ".wav"), tot_mixture, args.rate)

        # Save the metadata files
        metadata_path = os.path.join('/'.join(args.out_dir.split('/')[:-1]), 'metadata')
        os.makedirs(metadata_path, exist_ok=True)
        for md_df in md_dic:
            # Save the metadata in out_dir ./data/wavxk/mode/subset
            save_path_mixture = os.path.join(metadata_path, md_df + '.csv')
            md_dic[md_df].to_csv(save_path_mixture, index=False)


def resample_and_norm(signal, orig, target, lvl):

    if orig != target:
        signal = resample_poly(signal, target, orig)

    #fx = (AudioEffectsChain().custom("norm {}".format(lvl)))
    #signal = fx(signal)

    meter = pyloudnorm.Meter(target, block_size=0.1)
    loudness = meter.integrated_loudness(signal)
    signal = pyloudnorm.normalize.loudness(signal, loudness, lvl)

    return signal


def create_empty_mixture_md(n_src, subdir):
    """ Create the mixture dataframe"""
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['mixture_ID'] = {}
    mixture_dataframe['mixture_path'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
    elif subdir == 'mix_noisy':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    elif subdir == 'mix_single':
        mixture_dataframe["source_1_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    mixture_dataframe['length'] = {}
    return mixture_dataframe


def add_to_mixture_metadata(mix_df, mix_id, abs_mix_path, abs_sources_path,
                             length, subdir, abs_noise_path=None):
    """ Add a new line to mixture_df """
    sources_path = abs_sources_path
    if subdir == 'mix_clean':
        noise_path = []
    elif subdir == 'mix_single':
        sources_path = [abs_sources_path[0]]
    if abs_noise_path is not None:
        row_mixture = [mix_id, abs_mix_path] + sources_path + [abs_noise_path] + [length]
    else:
        row_mixture = [mix_id, abs_mix_path] + sources_path + [length]
    mix_df.loc[len(mix_df)] = row_mixture


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)



















