# ECG-public-data

```
# Cinc2020
# https://moody-challenge.physionet.org/2020/
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz
https://storage.googleapis.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB-XL.tar.gz
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz

# Ribeiro2020
# https://github.com/antonior92/automatic-ecg-diagnosis
https://zenodo.org/record/3765780/files/data.zip

https://zenodo.org/record/4916206/files/exams.csv
https://zenodo.org/record/4916206/files/exams_part0.zip
https://zenodo.org/record/4916206/files/exams_part1.zip
https://zenodo.org/record/4916206/files/exams_part2.zip
https://zenodo.org/record/4916206/files/exams_part3.zip
https://zenodo.org/record/4916206/files/exams_part4.zip
https://zenodo.org/record/4916206/files/exams_part5.zip
https://zenodo.org/record/4916206/files/exams_part6.zip
https://zenodo.org/record/4916206/files/exams_part7.zip
https://zenodo.org/record/4916206/files/exams_part8.zip
https://zenodo.org/record/4916206/files/exams_part9.zip
https://zenodo.org/record/4916206/files/exams_part10.zip
https://zenodo.org/record/4916206/files/exams_part11.zip
https://zenodo.org/record/4916206/files/exams_part12.zip
https://zenodo.org/record/4916206/files/exams_part13.zip
https://zenodo.org/record/4916206/files/exams_part14.zip
https://zenodo.org/record/4916206/files/exams_part15.zip
https://zenodo.org/record/4916206/files/exams_part16.zip
https://zenodo.org/record/4916206/files/exams_part17.zip

# Zheng2020
# https://figshare.com/collections/ChapmanECG/4560497/2
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651326/ECGData.zip
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653762/AttributesDictionary.xlsx
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653771/Diagnostics.xlsx
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651293/ConditionNames.xlsx
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651296/RhythmNames.xlsx

# Ptb_xl
# https://physionet.org/content/ptb-xl/1.0.1/
https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip

```
# Directory structure
```
tree -L 2

    ├── cinc2020
│   ├── PhysioNetChallenge2020_Training_2
│   ├── PhysioNetChallenge2020_Training_2.tar.gz
│   ├── PhysioNetChallenge2020_Training_CPSC
│   ├── PhysioNetChallenge2020_Training_CPSC.tar.gz
│   ├── PhysioNetChallenge2020_Training_E
│   ├── PhysioNetChallenge2020_Training_E.tar.gz
│   ├── PhysioNetChallenge2020_Training_PTB
│   ├── PhysioNetChallenge2020_Training_PTB.tar.gz
│   ├── PhysioNetChallenge2020_Training_PTB-XL
│   ├── PhysioNetChallenge2020_Training_PTB-XL.tar.gz
│   ├── PhysioNetChallenge2020_Training_StPetersburg
│   └── PhysioNetChallenge2020_Training_StPetersburg.tar.gz
├── ptb_xl
│   ├── example_physionet.py
│   ├── LICENSE.txt
│   ├── ptbxl_database.csv
│   ├── RECORDS
│   ├── records100
│   ├── records500
│   ├── scp_statements.csv
│   └── SHA256SUMS.txt
├── ribeiro2020_test
│   ├── annotations
│   ├── attributes.csv
│   ├── ecg_tracings.hdf5
│   └── README.md
├── ribeiro2020_train
│   ├── exams.csv
│   ├── exams_part0.hdf5
│   ├── exams_part0.zip
│   ├── exams_part10.hdf5
│   ├── exams_part10.zip
│   ├── exams_part11.hdf5
│   ├── exams_part11.zip
│   ├── exams_part12.hdf5
│   ├── exams_part12.zip
│   ├── exams_part13.hdf5
│   ├── exams_part13.zip
│   ├── exams_part14.hdf5
│   ├── exams_part14.zip
│   ├── exams_part15.hdf5
│   ├── exams_part15.zip
│   ├── exams_part16.hdf5
│   ├── exams_part16.zip
│   ├── exams_part17.hdf5
│   ├── exams_part17.zip
│   ├── exams_part1.hdf5
│   ├── exams_part1.zip
│   ├── exams_part2.hdf5
│   ├── exams_part2.zip
│   ├── exams_part3.hdf5
│   ├── exams_part3.zip
│   ├── exams_part4.hdf5
│   ├── exams_part4.zip
│   ├── exams_part5.hdf5
│   ├── exams_part5.zip
│   ├── exams_part6.hdf5
│   ├── exams_part6.zip
│   ├── exams_part7.hdf5
│   ├── exams_part7.zip
│   ├── exams_part8.hdf5
│   ├── exams_part8.zip
│   ├── exams_part9.hdf5
│   └── exams_part9.zip
└── zheng2020
    ├── AttributesDictionary.xlsx
    ├── ConditionNames.xlsx
    ├── Diagnostics.xlsx
    ├── ECGData
    ├── RhythmNames.xlsx
    └── zheng.zip

```