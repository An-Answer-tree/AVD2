import json
import re

# 读取ground truth JSON文件
ground_truth_file = '/root/raw_mmau/test.json'
with open(ground_truth_file, 'r') as f:
    ground_truth_data = json.load(f)

# 去除特殊符号并规范化文本的函数
def clean_text(text):
    # 去除特殊符号 [CLS] 和 [SEP]
    text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
    # 规范化分号和空格
    text = re.sub(r'\s*;\s*', '; ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 将ground truth文件转换为两个 COCO 格式
ground_truth_captions1 = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "caption"}
    ]
}
ground_truth_captions2 = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "caption"}
    ]
}

ann_id1 = 1
ann_id2 = 1
image_ids = set()
for item in ground_truth_data:
    video_id = item['video_id']
    
    # 添加image信息
    if video_id not in image_ids:
        image_info = {"id": video_id, "file_name": f"{video_id}.jpg"}
        ground_truth_captions1["images"].append(image_info)
        ground_truth_captions2["images"].append(image_info)
        image_ids.add(video_id)
    
    for event in item['events']:
        description = clean_text(event['description'])
        justification = clean_text(event['justification'])
        ground_truth_captions1["annotations"].append({
            "id": ann_id1,
            "image_id": video_id,
            "caption": description,
            "category_id": 1
        })
        ann_id1 += 1
        ground_truth_captions2["annotations"].append({
            "id": ann_id2,
            "image_id": video_id,
            "caption": justification,
            "category_id": 1
        })
        ann_id2 += 1

# 保存为两个新的COCO格式JSON文件
ground_truth_json_file1 = 'ground_truth_captions1.json'
ground_truth_json_file2 = 'ground_truth_captions2.json'
with open(ground_truth_json_file1, 'w') as f:
    json.dump(ground_truth_captions1, f, ensure_ascii=False, indent=4)
with open(ground_truth_json_file2, 'w') as f:
    json.dump(ground_truth_captions2, f, ensure_ascii=False, indent=4)

print(f"Ground truth COCO format JSON saved to {ground_truth_json_file1} and {ground_truth_json_file2}")





