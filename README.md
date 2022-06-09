## Feature Extraction Techniques for Word Recognition of Non-Native English Speakers

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> P. Berjon\*, R. Sharma\*, A. Nag, and S. Dev, Frequency-centroid features for word recognition of non-native English speakers, *IEEE Irish Signals & Systems Conference (ISSC)*, 2022. (\* Authors contributed equally.)

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

### Executive summary 
The objective of this work is to investigate complementary features which can aid the quintessential Mel frequency cepstral coefficients (MFCCs) in the task of closed, limited set word recognition for non-native English speakers of different mother-tongues. We utilize a subset of the [Speech Accent Archive database](https://accent.gmu.edu/index.php) for our experiments in this paper. 

### Code
All codes are written in `python3` and can be found in `./scripts/`.
+ `cnn.py` : Used with a right part of the dataset (for example, French MFCCs), it gives the results shown in the article. Our data folder contains many different features, but the specific results shown in Figure 3 and Table 2 can be obtained by using this file on the `./data/../mfcc/` and `./data/../mfcc+fc+ic/` folders for each accent.
+ `data_generation_noise.py` : Used with the original dataset, it automatically gives the noisy dataset. This python file is only used if you want to recreate the dataset. The files given in the data folder should be enough.
+ `multiple_features.py`: Give the different codes used to determine the different features extraction techniques results. Same as before: you can use it to recreate the dataset with chosen features (mfcc, mfcc+fc, etc.).

### Datasets
The dataset used in our case study can be found in the folder `./data/`.
