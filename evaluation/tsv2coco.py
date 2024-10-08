import pandas as pd
import json
import re

# 读取TSV文件
tsv_file = '/root/lic/ADAPT-main/outputCATSEE/multitask/sensor_course_speed/checkpoint-XX-XXXX/pred.BDDX.testing_28frames.beam1.max105.tsv'
df = pd.read_csv(tsv_file, sep='\t', header=None, names=['video_id', 'captions1', 'captions2'])

# 将TSV文件转换为两个 COCO 格式
generated_captions1 = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "caption"}
    ]
}
generated_captions2 = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "caption"}
    ]
}

ann_id1 = 1
ann_id2 = 1
image_ids = set()
for _, row in df.iterrows():
    video_id = int(re.findall(r'\d+', row['video_id'])[0])  # 提取video_id中的数字部分
    captions1 = json.loads(row['captions1'])
    captions2 = json.loads(row['captions2'])
    
    # 添加image信息
    if video_id not in image_ids:
        image_info = {"id": video_id, "file_name": f"{video_id}.jpg"}
        generated_captions1["images"].append(image_info)
        generated_captions2["images"].append(image_info)
        image_ids.add(video_id)
    
    # 处理第一个caption，表示description
    for caption_entry in captions1:
        generated_captions1["annotations"].append({
            "id": ann_id1,
            "image_id": video_id,
            "caption": caption_entry['caption'],
            "category_id": 1
        })
        ann_id1 += 1
        
    # 处理第二个caption，表示justification
    for caption_entry in captions2:
        generated_captions2["annotations"].append({
            "id": ann_id2,
            "image_id": video_id,
            "caption": caption_entry['caption'],
            "category_id": 1
        })
        ann_id2 += 1

# 保存为两个新的JSON文件
generated_json_file1 = 'generated_captions1.json'
generated_json_file2 = 'generated_captions2.json'
with open(generated_json_file1, 'w') as f:
    json.dump(generated_captions1, f, ensure_ascii=False, indent=4)
with open(generated_json_file2, 'w') as f:
    json.dump(generated_captions2, f, ensure_ascii=False, indent=4)

print(f"Generated COCO format JSON saved to {generated_json_file1} and {generated_json_file2}")








