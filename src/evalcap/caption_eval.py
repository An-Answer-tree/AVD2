import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import csv
import os
import shutil
from tqdm import tqdm
with open("../cvpr-nice-val/pred.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    img_names = [row['public_id'] for row in reader]

save_path = "outputs/pred.csv"
if os.path.exists(save_path):
    os.remove(save_path)
with open(save_path, "a+") as f:
    f.write(f"public_id,caption\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)



for img_name in tqdm(img_names):

    raw_image = Image.open(f"../cvpr-nice-val/val/{img_name}.jpg")

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    caption = model.generate({"image": image})


    with open(save_path, "a+") as f:
        f.write(f"{img_name},{caption}\n")
