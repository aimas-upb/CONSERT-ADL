AutoEncoder based approach to wearable sensor activity recognition
==================================================================

Datasets
--------

## A public domain dataset for human activity recognition using smartphones
  - paper: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge Luis Reyes-Ortiz. 2013. A public domain dataset for human activity recognition using smartphones.. In ESANN
  - dataset from original Activity2Vec paper

## PAMAP2
  - paper: Zheng, Y., Liu, Q., Chen, E., Ge, Y., Zhao, J.L., 2014. Time series classification using multi-channels deep convolutional neural networks, in: International Conference on Web-Age Information Management, Springer. pp. 298–310.
  
Information from this paper:
  - Data Set. We use the weakly labeled PAMAP2 data set for activity classification . It records 19 physical activities performed by 9 subjects. On a machine
with Intel I5-2410 (2.3GHz) CPU and 8G Memory (our experimental platform),
according to the estimation, it will cost nearly a month for 1-NN (DTW-5%) on
this data set if we use all the 19 physical activities. Hence, currently, we only
consider 4 out of these 19 physical activities in our work, which are ‘standing’,
‘walking’, ‘ascending stairs’ and ‘descending stairs’. And each physical activity
corresponds to a 3D time series. Moreover, 7 out of these 9 subjects are chosen.
Because the other two either have different physical activities or have different
dominant hand/foot.

Information about the dataset:
  - The Dataset is free to download as a .zip file, having 650 MB, 1.61 GB unarchived.
  - Link to the dataset: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
  - Link to download full dataset description: https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf
  - The individual datasets are stored as .dat files.
  - The archive also contains information about the individuals, the activities they perform, etc.
