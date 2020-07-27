# Animal_detection
This system detects poachers and animals in infrared photographs of the BIRDSAI dataset (https://sites.google.com/view/elizabethbondi/dataset?authuser=0) taken by drones over the African safari.

Method Tried: Detectron2

This repository includes:

- A data visualizer for the BIRDSAI dataset (AnimalsDataVisualizer.ipynb)

- Our system (AnimalsNotebook.ipynb)

# Requirements
- Access to Google Colab

# Installation
1. Clone this repository.

# AnimalsDataVisualizer.ipynb
AnimalsDataVisualizer.ipynb is a data visualizer for the BIRDSAI dataset.

AnimalsDataVisualizer.ipynb reads in the images and annotations of an infrared video in the BIRDSAI dataset and reconstructs the original infrared video with annotations.

# Getting Started with AnimalsDataVisualizer.ipynb
- Download AnimalsDataVisualizer.ipynb.
- Upload AnimalsDataVisualizer.ipynb to your Google Drive.
- Open AnimalsDataVisualizer.ipynb using Google Colab.

# Running AnimalsDataVisualizer.ipynb
- Uncomment the first two cells.
- Run the cells one by one in order. When the cell tells you to enter the name of the video you want, do so.
- After you have run all of the cells, the video will be under the directory "/content" in the file system of Google Colab.
- Download the video and run it using VLC or another video playing application.

# AnimalsNotebook.ipynb
AnimalsNotebook.ipynb contains our system for detecting poachers and animals.

# Getting Started with AnimalsNotebook.ipynb
- Download AnimalsNotebook.ipynb.
- Upload AnimalsNotebook.ipynb to your Google Drive.
- Open AnimalsNotebook.ipynb using Google Colab.

# Running AnimalsNotebook.ipynb
- Get a GPU with at least 16280 MiB from Google Colab.
- Uncomment the second, third, and sixth cells.
- Run the cells one by one in order.
- The cells with images contain sample outputs.
- The average precision (AP) will show up in the last cell.
