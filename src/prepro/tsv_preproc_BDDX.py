import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
sys.path.append("/data/hdd01/jinbu/SwinBERT-main")
import os.path as op
import json, yaml, code, io
import numpy as np
import pandas as pd
from src.utils.tsv_file_ops import tsv_writer
from src.utils.tsv_file_ops import generate_linelist_file
from collections import defaultdict

data_vid_id = "/root/raw_mmau/train3//{}"
dataset_path = '/root/raw_mmau/'
BDDX_anns = '/root/raw_mmau/train.json'
BDDX_subtitle = '/root/raw_mmau/train.json'

visual_file = "/root/raw_mmau/train_D/{}.img.tsv"
cap_file = "/root/raw_mmau/train_D/{}.caption.tsv"
linelist_file = "/root/raw_mmau/train_D/{}.linelist.tsv"
label_file = "/root/raw_mmau/train_D/{}.label.tsv"

def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str

def generate_caption_linelist_file(caption_tsv_file, save_file=None):
    num_captions = []
    for row in tsv_reader(caption_tsv_file):
        num_captions.append(len(json.loads(row[1])))

    cap_linelist = ['\t'.join([str(img_idx), str(cap_idx)]) 
            for img_idx in range(len(num_captions)) 
            for cap_idx in range(num_captions[img_idx])
    ]
    save_file = config_save_file(caption_tsv_file, save_file, '.linelist.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))
    return save_file

def dump_tsv_gt_to_coco_format(caption_tsv_file, outfile):
    annotations = []
    images = []
    cap_id = 0
    caption_tsv = tsv_reader(caption_tsv_file)

    for cap_row  in caption_tsv:
        image_id = cap_row[0]
        key = image_id
        caption_data = json.loads(cap_row[1])
        count = len(caption_data)
        for i in range(count):
            action = caption_data[i]['action']
            justification = caption_data[i]['justification']
            annotations.append(
                        {'image_id': image_id, 'action': action, 'justification': justification,
                        'id': cap_id})
            cap_id += 1

        images.append({'id': image_id, 'file_name': key})

    with open(outfile, 'w') as fp:
        json.dump({'annotations': annotations, 'images': images,
                'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'},
                fp)
 
def process_new():
    with open(BDDX_anns, 'r') as f:
        database = json.load(f)

        if not isinstance(database, list):
            raise ValueError("The annotation file is not a list")

    img_label = []
    rows_label = []
    caption_label = []

    for video_info in database:
        video_name = video_info.get('video_id')
        if video_name is None:
            raise KeyError("'video_id' not found in video_info")
        for event in video_info.get('events', []):
            sample_id = video_name
            action = event.get('description', '')
            justification = event.get('justification', '')

            resolved_video_name = f'{video_name}'
            resolved_data_vid_id = f'all_{resolved_video_name}'
            output_captions = []

            output_captions.append({"action": action, "justification": justification})
            caption_label.append([str(resolved_data_vid_id), json.dumps(output_captions)]) 
            rows_label.append([str(resolved_data_vid_id), json.dumps(output_captions)]) 
            img_label.append([str(resolved_data_vid_id), str(resolved_data_vid_id)])

    print(f"img_label: {len(img_label)} entries")
    print(f"rows_label: {len(rows_label)} entries")
    print(f"caption_label: {len(caption_label)} entries")

    resolved_visual_file = visual_file.format('all')
    print("generating visual file for", resolved_visual_file)
    tsv_writer(img_label, resolved_visual_file)

    resolved_label_file = label_file.format('all')
    print("generating label file for", resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    resolved_linelist_file = linelist_file.format('all')
    print("generating linelist file for", resolved_label_file)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_cap_file = cap_file.format('all')
    print("generating cap file for", resolved_cap_file)
    tsv_writer(caption_label, resolved_cap_file)
    print("generating cap linelist file for", resolved_cap_file)
    resolved_cap_linelist_file = generate_caption_linelist_file(resolved_cap_file)

    gt_file_coco = op.splitext(resolved_cap_file)[0] + '_coco_format.json'
    print("convert gt to", gt_file_coco)
    dump_tsv_gt_to_coco_format(resolved_cap_file, gt_file_coco)

    out_cfg = {}
    all_field = ['img', 'label', 'caption', 'caption_linelist', 'caption_coco_format']
    all_tsvfile = [resolved_visual_file, resolved_label_file, resolved_cap_file, resolved_cap_linelist_file, gt_file_coco]
    for field, tsvpath in zip(all_field, all_tsvfile):
        out_cfg[field] = tsvpath.split('/')[-1]
    out_yaml = 'alltrain.yaml'
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file:', op.join(dataset_path, out_yaml))

def main():
    process_new()

if __name__ == '__main__':
    main()