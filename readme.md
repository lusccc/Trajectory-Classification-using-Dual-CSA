This project is the code of the paper _Dual Supervised Autoencoder Based Trajectory Classification Using Enhanced Spatio-Temporal Information_

# Abstract
With the rapid development of mobile internet and location awareness techniques, massive spatial-temporal data is collected every day.
		Trajectory classification is critically important to many real world applications such as human mobility understanding and urban planning. A growing number of studies \hl{took} advantage of the deep learning method to learn the high-level features of trajectory data for accurate estimation. However, some of these studies didn't interpret spatial-temporal information well, more importantly, they didn't fully utilize the high-level features extracted by neural networks. To overcome these drawbacks, in this paper, spatial information is enhanced using the proposed stop state and turn state\hl{, at the same time,} the stronger time information is extracted by Recurrence Plot (RP). Moreover, a novel Dual Convolutional neural networks based Supervised Autoencoder (Dual-CSA) is proposed by making the network aware of predefined class centroids. Experiments conducted on real-world dataset demonstrate that Dual-CSA can learn the high-level features well. The highest accuracy of Dual-CSA is 88.835% proves the superiority of our method.

# Requirements
numba v0.49.1,
geopy v1.22.0,
pyts v0.11.0,
pydot v1.4.1,
sklearn v0.23.0,
matplotlib v3.2.1,
pandas v1.0.3,
seaborn v0.10.1,
tensorflow v2.1.0,
keras v2.3.1

You can simply install them by `pip install -r requirements.txt`.
# Usage
In this paper, we use GeoLife as our dataset. Please download it from https://www.microsoft.com/en-us/download/details.aspx?id=52367. Put all users trajectory folder under `geolife_raw` directory.  

For your convenience, you can run the `run.sh` under the project path by `bash run.sh`. The code inside `run.sh` is:
```shell script
python trajectory_extraction.py
python trajectory_segmentation_and_features.py --trjs_path "data/geolife_extracted/trjs_train.npy" --labels_path "data/geolife_extracted/labels_train.npy" --save_file_suffix train
python trajectory_segmentation_and_features.py --trjs_path "data/geolife_extracted/trjs_test.npy" --labels_path "data/geolife_extracted/labels_test.npy" --save_file_suffix test
python MF_RP_mat.py --trjs_segs_features_path "data/geolife_features/trjs_segs_features_train.npy" --save_path "data/geolife_features/RP_mats_train.npy"
python MF_RP_mat.py --trjs_segs_features_path "data/geolife_features/trjs_segs_features_test.npy" --save_path "data/geolife_features/RP_mats_test.npy"
python PEDCC.py
python Dual_SAE.py
```


Specifically, the function of each file is as follows.

(1) `params.py` is the place set your own parameters, including movement features or auxiliary features used, loss weights, etc.  
  
(2) `trajectory_extraction.py` is used to extract trajectory of each user and with its corresponding label. It also will split the dataset into train set and test set.  
(3) `trajectory_segmentation_and_features.py` is used to segment the trajectory and calculate its movement features.  
(4) `MF_RP_mat.py` is used to calculate Recurrence Plots of movement features. It may require a lot of memory. Please contract us at 582066450@qq.com if you need a version run on hard disk.  
(5) `PEDCC.py` is used to generate the predefined centroids of class proposed in paper _A Classification Supervised Auto-Encoder Based on Predefined Evenly-Distributed Class Centroids_. Please refer detail in https://github.com/anlongstory/CSAE.  
(6) `Dual_SAE.py` is used to train the proposed Dual_CSA model. You can specify parameters in the code including loss weights, epoch number, etc.

Note: the other file we did not mention is mainly used to run the comparison experiments. We will elaborate it in the feature.

**If you have any question, please feel free to contact us at 58206645@qq.com**


