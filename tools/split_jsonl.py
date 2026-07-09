import os
import os.path as osp
import time
import itertools
import shutil
import glob
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
import numpy as np

def save_lines(lines, filename):
    os.makedirs(osp.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(lines)
    del lines

def get_part_jsonls(save_dir, total_line_number, ext='.jsonl', chunk_size=1000, bucket_size=1000):
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    chunk_id2save_files = {}
    missing = False
    parts = int(np.ceil(total_line_number / chunk_size))
    for chunk_id in range(1, parts+1):
        if chunk_id == parts:
            num_of_lines = total_line_number - chunk_size * (parts-1)
        else:
            num_of_lines = chunk_size
        bucket = (chunk_id-1) // args.bucket_size + 1
        chunk_id2save_files[chunk_id] = osp.join(save_dir, f'{bucket:06d}', f'{chunk_id:04d}_{parts:04d}_{num_of_lines:09d}{ext}')
        if not osp.exists(chunk_id2save_files[chunk_id]):
            missing = True
    return missing, chunk_id2save_files

def split_large_txt_files(all_lines, chunk_id2save_files):
    chunk_id = 1
    total = len(all_lines)
    pbar = tqdm.tqdm(total=len(chunk_id2save_files))
    chunk = []
    futures = []

    max_workers = 128
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for line in all_lines:
            chunk.append(line)
            cur_chunk_size = int(osp.splitext(osp.basename(chunk_id2save_files[chunk_id]))[0].split('_')[-1])
            if len(chunk) >= cur_chunk_size:
                futures.append(
                    executor.submit(save_lines, chunk, chunk_id2save_files[chunk_id])
                )
                pbar.update(1)
                chunk = []
                chunk_id += 1

        if len(chunk):
            raise ValueError("last chunk not save, means misalign data!")

        for future in as_completed(futures):
            future.result()

    pbar.close()


from multiprocessing import Manager
lock = Manager().Lock()
def read_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    global pbar
    with lock:
        pbar.update(1)
    return lines

def read_jsonls(jsonl_files, worker):
    global pbar
    from multiprocessing.pool import ThreadPool
    pbar = tqdm.tqdm(total=len(jsonl_files))
    print(f'[Data Loading] Reading {len(jsonl_files)} meta files...')
    all_lines = []
    if len(jsonl_files) == 1:
        lines_num = int(osp.splitext(jsonl_files[0])[0].split('_')[-1])
        pbar = tqdm.tqdm(total=lines_num)
        with open(jsonl_files[0], 'r') as f:
            for line in f:
                pbar.update(1)
                all_lines.append(line)
    else:
        with ThreadPool(worker) as pool:
            for img_metas in pool.starmap(read_jsonl, [(bin_file,) for bin_file in jsonl_files]):
                all_lines.extend(img_metas)
    np.random.shuffle(all_lines)
    return all_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_folder_list', type=str, default='', nargs='+', help='patha pathb pathc')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--bucket_size', type=int, default=10000)
    parser.add_argument('--worker', type=int, default=128)
    args = parser.parse_args()

    global pbar
    t1 = time.time()
    jsonl_files = []
    for item in args.jsonl_folder_list:
        jsonl_files += glob.glob(osp.join(item, '*.jsonl'))
        # jsonl_files += glob.glob(osp.join(item, '*/*.jsonl' ))
    np.random.shuffle(jsonl_files)

    pbar = tqdm.tqdm(total=len(jsonl_files))
    lines = read_jsonls(jsonl_files, args.worker)
    print(f'total {len(lines)} lines')
    line_num = len(lines)
    missing, chunk_id2save_files = get_part_jsonls(args.save_dir, line_num, chunk_size=args.chunk_size, bucket_size=args.bucket_size)
    
    split_large_txt_files(lines, chunk_id2save_files)
    t2 = time.time()
    print(f'split takes {t2-t1}s')
