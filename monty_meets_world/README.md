# Monty Meets World

For some visualizations and results of this project see: https://docs.google.com/presentation/d/1JfhU-CovXLi44hlCpAdEwRLgvXSZ3ypY9C81SSNZTaI/edit?usp=sharing

## Overview

The project is made up of 4 components:

1) an iOS app to take pictures with the Apple TrueDepth camera and stream them.
2) A server script to run on the laptop to receive the images
3) Custom Dataloaders and Environment to read in the 2D rgbd data and move a small patch over them to send data to Monty
4) A new dataset of meshes of 10 real-world objects

To run a live demo, follow steps in sections 1-3.

To test on an already created dataset of real-world images, simply run experiments from `monty_world_experiments.py` for this on the cloud infrastructure or copy the respective pre-trained models and dataset and run them locally.

## 1. Running The App -> Make Dataset or Run Live Demo

To run the App, open the project in this folder in xCode and run it on your device. The app is currently not in the App store.

saveImage, uploadRGBData and uploadDepthData were added by us
to the original project from the Apple Developer website
https://developer.apple.com/documentation/avfoundation/additional_data_capture/streaming_depth_data_from_the_truedepth_camera

## 2. Running the Server Script -> Receive Data from App

To receive the camera images from the iOS device, you need to run a local server. To do that run

`python src/tbp/monty/frameworks/dataset_utils/server.py`

(there won't be any output until you take a picture in the iOS App). For streaming to work you need to go to the system settings of your app and change the URL to your local wifi IP address.

## 3. Processing Data in Monty

To process an offline created dataset, use the `SaccadeOnImageDataLoader` is the dataloader class in your config. To run a live streaming experiment use the `SaccadeOnImageFromStreamDataLoader` . Some already created configs for this can be found in `monty_world_experiments.py`. Recommended experiments for those two scenarios are `world_image_on_scanned_model` and `world_image_from_stream_on_scanned_model` respectively.

To visualize the matching procedure (especially useful for a live demo) set `show_sensor_output` in the experiment_args to True.