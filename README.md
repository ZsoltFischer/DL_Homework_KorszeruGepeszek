# Deep Learning Homework – Korszerű Gépészek
Homework repository of team "Korszeru Gepeszek" for deep learning course: BMEVITMAV45 (2019/20/1)
The project is carried out in Google Colaboratory using Python 3.7.
`Deep_Learning_Document.pdf` contains the full documentation of the project.

# Talker separation problem
The aim of our homework is to create a network that is capable of separating the speech of two speakers who speak simultanously. The method of recording is presumed to be single channeled (mono/monaural).

# Data sets
The data source is the audiobook form of the Charles Dickens novel Hard Times, read by ten different speakers. To create input data we split and mix these signals on top of each other.

# Dependencies
- `tensorflow` for the model, using tensorflow 1.x
- `librosa`, soundfile for audio processing
- `h5py` for saving the generated dataset in HDF5 format
- `mir_eval` for SDR calculation

# Results
Output samples can be found in the `model_example_outputs` folder.
