# AVD2: Accident Video Diffusion for Accident Video Description

### The First Work to Generate Accident Videos:
![The Teaser](./images/teaser.png)

### This repository is an official implementation of AVD2: Accident Video Diffusion for Accident Video Description.
Created by Cheng Li, Keyuan Zhou, Tong Liu, Yu Wang, Mingqiao Zhuang, Huan-ang Gao, Bu Jin and Hao Zhao from Institute for AI Industry Research(AIR), Tsinghua University.

### Our Framework:
![The Framework Architecture](./images/Framework.png)


### Our AVD2 Project Video is available at:
[AVD2 Project Video: https://youtu.be/iGdSIofB_k8](https://youtu.be/iGdSIofB_k8)

# Introduction
We propose a novel framework, AVD2 (Accident Video Diffusion for Accident Video Description), which enhances transparency and explainability in autonomous driving systems by providing detailed natural language narrations and reasoning for accident scenarios. AVD2 jointly tackles both the accident description and prevention tasks, offering actionable insights through a shared video representation.This repository includes (will be released soon) the full implementation of AVD2, along with the training and evaluation setups, the generated accident dataset EMMAU dataset and the conda environment.

# Note
We have uploaded the required environment of our AVD2 system.  
We have released the whole raw EMM-AU dataset (including raw MM-AU dataset and the raw generation videos.  
We have released the whole processed dataset of the EMM-AU dataset.  
We have released the instructions and codes for the data augmentation (including super-resolution code and the instructions for Open-Sora finetuning).  
We have released the checkpoint file of our fintuned improved Open-Sora 1.2 model.   
We have released the data preprocessing codes ("/root/src/prepro/") and the model evaluation codes ("/root/lic/ADAPT-main/src/evalcap/") of the project.    


# Getting Started Environment
Create conda environment:
```bash
conda create --name AVD2 python=3.8
```
Install torch:
```bash
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
Install apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
rm -rf apex
```
Install mpi4py:
```bash
conda install -c conda-forge mpi4py openmpi
```
Install other dependencies and packages
```bash
pip install -r requirements.txt
```

# More Details for Framework
Our novel AVD2 framework is based on the Action-aware Driving Caption Transformer (ADAPT) and Self Critical Sequence Training (SCST).  
The codes and more information about ADAPT and SCST can be found and referenced here:  
[ADAPT: https://arxiv.org/pdf/2302.00673](https://arxiv.org/pdf/2302.00673)  
[ADAPT codes: https://github.com/jxbbb/ADAPT/tree/main?tab=MIT-1-ov-file](https://github.com/jxbbb/ADAPT/tree/main?tab=MIT-1-ov-file)  
[SCST: https://arxiv.org/abs/1612.00563](https://arxiv.org/abs/1612.00563)  
[SCST codes: https://github.com/ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)  

# Dataset
This part includes the Dataset Preprocessing code, the Raw Dataset (including the whole EMM-AU dataset), the codes and steps to do the data augmentation and the Processed Dataset.

## Dataset Preprocessing
Need to change the name of the train/val/test dataset and the locations.
```bash
cd src
cd prepro
sh preprocess.sh
```

## Raw Dataset Download
EMM-AU(Enhanced MM-AU Dataset) contains "Raw MM-AU Dataset" and the "Enhanced Generated Videos".  
| Parts             | Download             |
|-------------------|----------------------|
| Raw MM-AU Dataset | [Official Github Page](https://github.com/jeffreychou777/LOTVS-MM-AU?tab=readme-ov-file#datasets-download) |
| Our Enhanced Generated Videos     | [HuggingFace](https://huggingface.co/datasets/secsecret/EMM-AU/blob/main/EMM-AU(Enhanced%20Generated%20Videos).zip)         |  

## Processed Dataset Download
You can download the [Processed_EMM-AU_Dataset](https://huggingface.co/datasets/secsecret/EMM-AU/tree/main) in our HuggingFace.

## Data Augmentation
We utilized Project [Open-Sora 1.2](https://github.com/hpcaitech/Open-Sora) to inference the "Enhanced Part" of EMM-AU. You can reference Open-Sora Official GitHub Page for installation.
### Fine-tuning for Open-Sora
Before fine-tuning, you need to prepare a csv file. [HERE IS A METHOD](https://github.com/hpcaitech/Open-Sora/tree/feature/mirror_v1.2/tools/datasets#dataset-to-csv)  
An example ready for training:
```csv
path, text, num_frames, width, height, aspect_ratio
/absolute/path/to/image1.jpg, caption, 1, 720, 1280, 0.5625
/absolute/path/to/video1.mp4, caption, 120, 720, 1280, 0.5625
/absolute/path/to/video2.mp4, caption, 20, 256, 256, 1
```
Then use the bash command to train new model or fine-tuned model(based on YOUR_PRETRAINED_CKPT).  
You can also change the training config in "configs/opensora-v1-2/train/stage3.py"
```bash
# one node
torchrun --standalone --nproc_per_node 8 scripts/train.py \
    configs/opensora-v1-2/train/stage3.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/stage3.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```
### Inference with Open-Sora
You can Download our [pretrained model](https://huggingface.co/datasets/secsecret/EMM-AU/tree/main) for Accident Videos Generation.
```bash
# text to video
python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall"

# batch generation(need a txt file, each line has a single prompt)
python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt-path YOUR_PROMPT.TXT \
  --batch-size 1 \
  --loop 1 \
  --save-dir YOUR_SAVE_DIR \
  --ckpt-path YOUR_CHECKPOINT
```
### RRDBNet Super-Resolution
The conda environment for the super-resolution part can be installed as:
```bash
conda create --name S_R python=3.8
cd src/Super_resolution
pip install -r requirements.txt
```
Then running the RRDBNet model code within the Real-ESRGAN framework to do the super-resolution steps for the dataset.
```bash
python realesrgan_utils.py
```

# Download Our Fine-tuned Open-Sora 1.2 model for Video Generation
You can download the [pretrained_model_for_video_generation](https://huggingface.co/datasets/secsecret/EMM-AU/tree/main) in our HuggingFace.

# Visualization
## This is the example of the accident frames of our EMMAU dataset:  
![The example frame](./images/EMMAU_accident_example.png)  

## This is the visualization of the Understanding ability of our AVD2 system (comparred with the ChatGPT-4o & ground truth):  
### Accident example 1:  
![Example of EMMAU 1](./images/1_accident_2.png)  
<span style="color:black">**AVD2 Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A vehicle changes lanes with the same direction to ego-car; Vehicles don't give way to normal driving vehicles when turning or changing lanes.  
<span style="color📘">**Avoidance:**</span>
Before turning or changing lanes, vehicles should turn on the turn signal in advance, observe the surrounding vehicles and control the speed. When driving, vehicles should abide by traffic rules, and give the way for the normal running vehicles. Vehicles that will enter the main road should give way to the vehicles which drive on the main road or leave the main road. Vehicles that drive on the auxiliary road should give way to the vehicles which drive off the main road.

<span style="color:black">**ChatGPT-4o Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A vehicle approaches a busy intersection and fails to notice another car coming from the side; The vehicle abruptly brakes to avoid a collision, but the close proximity creates a dangerous situation.   
<span style="color📘">**Avoidance:**</span>
Drivers should always reduce speed when approaching intersections and remain alert to traffic from all directions. Maintaining a safe distance and carefully observing other vehicles is essential to prevent accidents at intersections.

<span style="color:black">**GroundTruth**</span>  
<span style="color👱‍♂️">**Description:**</span>
 Lead vehicle stops; Vehicles do not give way to normal driving vehicles when turning or changing lanes.   
<span style="color📘">**Avoidance:**</span>
Before turning or changing lanes, vehicles should turn on the turn signal in advance, observe the surrounding vehicles and control the speed. When driving, vehicles should abide by traffic rules, and give the way for the normal running vehicles. Vehicles that will enter the main road should give way to the vehicles which drive on the main road or leave the main road. Vehicles that drive on the auxiliary road should give way to the vehicles which drive off the main road.  

### Accident example 2:  
![Example of EMMAU 2](images/2_accident_2.png)  
<span style="color:black">**AVD2 Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
A vehicle changes lanes with the same direction to ego-car; Vehicles don't give way to normal driving vehicles when turning or changing lanes.    
<span style="color📘">**Avoidance:**</span>
Ego-cars should not exceed the speed limit during driving, slow down when passing intersections or crosswalks, especially for areas with many pedestrians.

<span style="color:black">**ChatGPT-4o Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A vehicle makes a sharp turn at an intersection without signaling; The vehicle behind is forced to brake abruptly due to insufficient reaction time.  
<span style="color📘">**Avoidance:**</span>
Drivers should signal well in advance before making turns at intersections. Maintaining a safe distance from other vehicles and anticipating sudden turns can help prevent accidents.

<span style="color:black">**GroundTruth**</span>  
<span style="color👱‍♂️">**Description:**</span>
Vehicles meet on the road; Vehicles drive too fast with short braking distance.  
<span style="color📘">**Avoidance:**</span>
Vehicles should not exceed the speed limit during driving, especially in areas with many pedestrians. Vehicles should slow down when passing intersections or crosswalks, and observe the traffic carefully.

# Acknowledgements
We are grateful for the support of the Institute for AIR at Tsinghua University, and Kairui Ding's help on our project, and the LOTVS-MMAU (Multi-Modal Accident video Understanding) team for open-sourcing and sharing the MM-AU dataset.
