# Distracted-Driver
Implemented a Deep Learning approach using transfer learning for the Kaggle Distracted Driver competition (https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview).

As a solution for the Distracted Driver competition I implemented a deep learning model using transfer learning. The model used a pretrained ResNet50 from tensorflow as a feature extractor.


## Project Structure:

- common
  - constants.py contains the necessary constants defined during this project
  - run_pipelins.py is the runs preprocesses the data and trains the model
- model
  - transfer_learning_model.py defines the pretrained ResNet50 + classifier architecture 
- preprocessing
  - read_data.py creates the training and validation generators
  - train_val_split.py creates the necessary directories and populates them with the required images for the image generators
  - data_pipeline.py generates and preprocesses the data (by running the above two scripts)
- visualisation_tools
  - class_plots.py generates a plot to visualise the data
