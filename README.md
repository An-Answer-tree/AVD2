# CATS: Critical Adaptive Transformer for Self-driving Captions
![The Framework Architecture](./images/Framework.png)
### This repository is an official implementation of CATS: Critical Adaptive Transformer for Self-driving Captions.
Created by Cheng Li, Keyuan Zhou, Tong Liu, Yu Wang, Mingqiao Zhuang, Kairui Ding and Hao Zhao from Institute for AI Industry Research(AIR), Tsinghua University.
### Our Project Video is available at:

# Introduction
We propose a novel framework, CATS (Critical Adaptive Transformer for Self-driving Captions), which enhances transparency and explainability in autonomous driving systems by providing detailed natural language narrations and reasoning for accident scenarios. CATS jointly tackles both the accident description and prevention tasks, offering actionable insights through a shared video representation.This repository includes (will be released soon) the full implementation of CATS, along with the training and evaluation setups, the generated accident dataset EMMAU dataset and the conda environment.

# Note
We have uploaded the requirement environment of our CATS system.  
We have released the data preprocessing codes ("/root/src/prepro/") and the evaluation codes ("/root/lic/ADAPT-main/src/evalcap/") of the project.  
We have released the preprocessed dataset of the EMMAU dataset.  
We will release the entired code (including the checkpoints file) of the CATS system soon.  
We will release the dataset of the generated accident video (EMMAU dataset).  
We will upload the detailed instructions of the readme document.  

# Visualization
This is the example of the accident frames of our EMMAU dataset:  
![The example frame](./images/EMMAU_accident_example.png)  
This is the visualization of the Understanding ability of our CATS system (comparred with the ChatGPT-4 & ground truth):  
1. Accident example one

<span style="color:black">**CATS Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A vehicle changes lanes with the same direction to ego-car; Vehicles don't give way to normal driving vehicles when turning or changing lanes.  

<span style="color📘">**Description:**</span>
Before turning or changing lanes, vehicles should turn on the turn signal in advance, observe the surrounding vehicles and control the speed. When driving, vehicles should abide by traffic rules, and give the way for the normal running vehicles. Vehicles that will enter the main road should give way to the vehicles which drive on the main road or leave the main road. Vehicles that drive on the auxiliary road should give way to the vehicles which drive off the main road.



