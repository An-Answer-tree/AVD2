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
We have released the part of the preprocessed dataset of the EMMAU dataset ("/root/Processed_Dataset/"). And will released all of them.
We will soon release the entired code (including the checkpoints file) of the CATS system soon.  
We will soon release the dataset of the generated accident video (raw EMMAU dataset).   
We will soon upload the finetuned Open-Sora 1.2 model, which is used for generating the EMMAU Dataset.
We will soon upload the detailed instructions of the readme document (how to deployment the project). 

# More Details
Our novel CATS framework is based on the Action-aware Driving Caption Transformer (ADAPT) and Self Critical Sequence Training (SCST).  
The codes and more information about ADAPT and SCST can be found here:  
[ADAPT: https://github.com/jxbbb/ADAPT/tree/main?tab=MIT-1-ov-file](https://github.com/jxbbb/ADAPT/tree/main?tab=MIT-1-ov-file)  
[SCST: https://arxiv.org/abs/1612.00563](https://arxiv.org/abs/1612.00563)  

# Visualization
This is the example of the accident frames of our EMMAU dataset:  
![The example frame](./images/EMMAU_accident_example.png)  
This is the visualization of the Understanding ability of our CATS system (comparred with the ChatGPT-4 & ground truth):  

### Accident example 1:  
![Example of EMMAU 1](./images/1_accident_2.png)  
<span style="color:black">**CATS Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A vehicle changes lanes with the same direction to ego-car; Vehicles don't give way to normal driving vehicles when turning or changing lanes.  
<span style="color📘">**Avoidance:**</span>
Before turning or changing lanes, vehicles should turn on the turn signal in advance, observe the surrounding vehicles and control the speed. When driving, vehicles should abide by traffic rules, and give the way for the normal running vehicles. Vehicles that will enter the main road should give way to the vehicles which drive on the main road or leave the main road. Vehicles that drive on the auxiliary road should give way to the vehicles which drive off the main road.

<span style="color:black">**ChatGPT4 Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 A bus and a sedan collided at an intersection; The sedan ran a red light while the bus was crossing.   
<span style="color📘">**Avoidance:**</span>
Always obey traffic signals and signs. Increase vigilance when approaching intersections, and ensure the intersection is clear before proceeding, even if the light is green. Drivers should reduce speed and be prepared to stop if necessary to avoid accidents.

<span style="color:black">**GroundTruth**</span>  
<span style="color👱‍♂️">**Description:**</span>
 Lead vehicle stops; Vehicles do not give way to normal driving vehicles when turning or changing lanes.   
<span style="color📘">**Avoidance:**</span>
Before turning or changing lanes, vehicles should turn on the turn signal in advance, observe the surrounding vehicles and control the speed. When driving, vehicles should abide by traffic rules, and give the way for the normal running vehicles. Vehicles that will enter the main road should give way to the vehicles which drive on the main road or leave the main road. Vehicles that drive on the auxiliary road should give way to the vehicles which drive off the main road.  

### Accident example 2:  
![Example of EMMAU 2](images/2_accident_2.png)  
<span style="color:black">**CATS Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
A vehicle changes lanes with the same direction to ego-car; Vehicles don't give way to normal driving vehicles when turning or changing lanes.    
<span style="color📘">**Avoidance:**</span>
Ego-cars should not exceed the speed limit during driving, slow down when passing intersections or crosswalks, especially for areas with many pedestrians.

<span style="color:black">**ChatGPT4 Prediction**</span>  
<span style="color👱‍♂️">**Description:**</span>
 High-speed highway collision; One car attempted a sudden lane change without signaling, cutting off another.   
<span style="color📘">**Avoidance:**</span>
Always use turn signals when changing lanes or making turns to alert other drivers of your intentions. Keep a safe distance from other vehicles and check mirrors and blind spots before making any maneuvers. Avoid sudden lane changes, especially at high speeds, and always be aware of the flow of traffic around you.

<span style="color:black">**GroundTruth**</span>  
<span style="color👱‍♂️">**Description:**</span>
Vehicles meet on the road; Vehicles drive too fast with short braking distance.  
<span style="color📘">**Avoidance:**</span>
Vehicles should not exceed the speed limit during driving, especially in areas with many pedestrians. Vehicles should slow down when passing intersections or crosswalks, and observe the traffic carefully.

# Acknowledgements
We are grateful for the support of the Institute for AI Research (AIR) at Tsinghua University, Mr. Bu Jin for his guidance and help on our project, and the LOTVS-MMAU (Multi-Modal Accident video Understanding) team for open-sourcing and sharing the MM-AU dataset.
