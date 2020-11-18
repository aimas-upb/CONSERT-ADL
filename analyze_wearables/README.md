AutoEncoder based approach to wearable sensor activity recognition
==================================================================

Datasets
--------

## A public domain dataset for human activity recognition using smartphones
  - paper: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge Luis Reyes-Ortiz. 2013. A public domain dataset for human activity recognition using smartphones.. In ESANN
  - dataset from original Activity2Vec paper
  
Information from this paper:
  - A set of experiments were carried out to obtain the HAR dataset. A group of 30 volunteers with ages ranging from 19 to 48 years were selected for this task. Each person
was instructed to follow a protocol of activities while wearing a waist-mounted Samsung Galaxy S II smartphone. The six selected ADL were standing, sitting, laying
down, walking, walking downstairs and upstairs. Each subject performed the protocol
twice: on the first trial the smartphone was fixed on the left side of the belt and on the
second it was placed by the user himself as preferred. There is also a separation of 5
seconds between each task where individuals are told to rest, this facilitated repeatability (every activity is at least tried twice) and ground trough generation through the
visual interface.
  - The tasks were performed in laboratory conditions but volunteers were asked to perform freely the sequence of activities for a more naturalistic dataset.

Information about the dataset:
  - The Dataset is free to download as a .zip file, having 58.2 MB, 269 MB unarchived.
  - Link to the dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/
  - The individual datasets are stored as .txt files.
  
  The dataset includes the following files:

- 'README.txt' : Constains details about the collected data, the existing files in the archive, aditional notes and license agreement

- 'features_info.txt': Shows information about the variables used on the feature vector.

- 'features.txt': List of all features.

- 'activity_labels.txt': Links the class labels with their activity name.

- 'train/X_train.txt': Training set.

- 'train/y_train.txt': Training labels.

- 'test/X_test.txt': Test set.

- 'test/y_test.txt': Test labels.

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
