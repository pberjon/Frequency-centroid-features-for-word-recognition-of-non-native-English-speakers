## Feature Extraction Techniques for Word Recognition of Non-Native English Speakers

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> P. Berjon, S. Dev, R. Sharma, and A. Nag, Feature Extraction Techniques for Word Recognition of Non-Native English Speakers, *under review*.


### Executive summary 


### Code
All codes are written in `python3` and can be found in `./scripts/`.
+ `cnn.py` : Used with a right part of the dataset (for examples french MFCCs), it gives the results shown in the article. Our data folder contains many different features, but the specific results shown in Figure 3 and Table 2 can be obtained by using this file on the mfcc and mfcc+fc+ic folders for each accent.
+ `data_generation_noise.py` : Used with the original dataset, it gives the noisy dataset.
+ `multiple_features.py`: Give the different codes used to determine the different features extraction techniques results.



### Datasets
The dataset used in our case study can be found in the folder `./data/`.
