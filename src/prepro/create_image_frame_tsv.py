import os.path as op
import json
import os
import sys
from pathlib import Path    
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)
from PIL import Image
import io
import base64
import cv2

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    pass
                else:
                    raise

def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    if not is_py2:
        sep = sep.encode()
    with open(tsv_file_name_tmp, 'wb') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx, open(tsv_8b_file_tmp, 'wb') as fp8b:
        assert values is not None
        count = 0  # 添加计数器
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(map(lambda v: v.encode('utf-8') if isinstance(v, unicode) else str(v), value)) + '\n'
            else:
                value = map(lambda v: v if type(v) == bytes else str(v).encode(), value)
                v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)
            count += 1  # 增加计数器
            print(f"Written line {count}: {v[:50]}...")  # 添加调试信息，检查写入的每一行数据
    print(f"Total lines written: {count}")  # 输出总计数
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    os.rename(tsv_8b_file_tmp, tsv_8b_file)

def resize_and_to_binary(img_path, target_image_size):
    if img_path is None:
        if target_image_size < 0:
            target_image_size = 256
        resized = np.zeros((target_image_size, target_image_size, 3), dtype = "uint8")
        s = (target_image_size, target_image_size)
    else:
        im = cv2.imread(img_path)
        height, width, channels = im.shape
        s = (width, height)
        if target_image_size > 0:
            s = min(width, height)
            r = target_image_size / s
            s = (round(r * width), round(r * height))
            resized = cv2.resize(im, s)
        else:
            resized = im

    binary = cv2.imencode('.jpg', resized)[1].tobytes()
    encoded_base64 = base64.b64encode(binary)
    return encoded_base64, s

def load_tsv_to_mem(tsv_file, sep='\t'):
    data = []
    with open(tsv_file, 'r') as fp:
        for _, line in enumerate(fp):
            data.append([x.strip() for x in line.split(sep)])
    return data

def get_image_binaries(list_of_paths, image_size=56):
    batch = []
    is_None = [v is None for v in list_of_paths]
    assert not any(is_None) or all(is_None)
    for img_path in list_of_paths:
        if img_path is None or isinstance(img_path, str):
            x, shape = resize_and_to_binary(img_path, target_image_size=image_size)
        else:
            raise ValueError(f'img_path not str, but {type(img_path)}')
        batch.append(x)
    return batch, shape

def prepare_single_video_frames(caption_id, vid_path, num_frames=28):
    previous_image_path = None
    images = []
    local_data_path = vid_path.replace("datasets", "_datasets")
    if not op.exists(local_data_path) and not op.exists(vid_path):
        images = [None]*num_frames
        return None

    video_id = Path(vid_path).stem
    for i in range(num_frames):
        current_image_path = op.join(data_path, str(caption_id).zfill(5), f'{video_id}_frame{(i+1):04d}.jpg')
        if not op.exists(current_image_path):
            if previous_image_path:
                current_image_path = previous_image_path 
            else:
                print(f'The first image for {video_id} does not exists')
                images = [None]*num_frames
                return images
        images.append(current_image_path)
        previous_image_path = current_image_path
    return images

def process_video_chunk(item, image_size=256, num_frames=28):
    caption_id = item.get('video_id')
    if caption_id is None:
        raise KeyError("'video_id' not found in item")
    
    vid_path = f'/root/raw_mmau/train3/{str(caption_id).zfill(6)}.mp4'
    
    images = prepare_single_video_frames(caption_id, vid_path, num_frames)
    if images is None:
        return None
    image_binaries, image_shape = get_image_binaries(images, image_size)
    
    resolved_data_vid_id = f"{str(caption_id).zfill(5)}/{item.get('vidName', 'unknown')}"
    line_item = [str(resolved_data_vid_id), json.dumps({"class": -1, "width": image_shape[0], "height": image_shape[1]})]
    line_item += image_binaries
    return line_item
def main():
    output_folder = f"/root/raw_mmau/frameval_tsv"
    os.makedirs(output_folder, exist_ok=True)
    global data_path

    image_size = 256
    num_frames = 28
    data_path = f"/root/raw_mmau/train3/train_D/28frames/"
    num_workers = 32
    video_info_tsv = '/root/raw_mmau/train.json'

    with open(video_info_tsv) as f:
        video_info = json.load(f)

        if not isinstance(video_info, list):
            raise ValueError("The video info file is not a list")

    if len(video_info) == 0:
        raise ValueError("The video info file is empty")

    print(f"Loaded {len(video_info)} video info entries")

    resolved_visual_file = f"{output_folder}/all_{num_frames}frames_img_size{image_size}.img.tsv"
    print("generating visual file for", resolved_visual_file)

    from functools import partial
    worker = partial(
        process_video_chunk, image_size=image_size, num_frames=num_frames)

    def gen_rows():
        with mp.Pool(num_workers) as pool, tqdm(total=len(video_info)) as pbar:
            for idx, line_item in enumerate(pool.imap(worker, video_info, chunksize=8)):
                pbar.update(1)
                if line_item is not None:
                    print(f"Processed line item {idx}: {line_item[0]}")  # 添加调试信息，检查处理的每一行数据
                    yield(line_item)

    tsv_writer(gen_rows(), resolved_visual_file)

if __name__ == '__main__':
    main()