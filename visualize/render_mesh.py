import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='path to folder with mp4 and results.npy')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    npy_path = os.path.join(params.input_path, 'results.npy')
    assert os.path.exists(npy_path)
    items = np.load(npy_path, allow_pickle=True).item()

    for sample_idx in range(items['num_samples']):
            text = items['text'][sample_idx]
            text = text.replace(' ', '_')
            for rep_idx in range(items['num_repetitions']):
                res_dir = os.path.join(params.input_path, f'{text}_{rep_idx}')
                if os.path.exists(res_dir):
                    shutil.rmtree(res_dir)
                os.makedirs(res_dir)
                npy2obj = vis_utils.npy2obj(npy_path, sample_idx, rep_idx,
                                            device=params.device, cuda=params.cuda)
                print('Saving obj files to [{}]'.format(os.path.abspath(res_dir)))
                for frame_i in tqdm(range(npy2obj.real_num_frames)):
                    npy2obj.save_obj(os.path.join(res_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

