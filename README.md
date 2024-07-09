# DeepVOG
<p align="center"> 
<img width="320" height="240" src="ellipsoids.png">
</p>
DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network. Currently it is available for offline gaze estimation of eye-tracking video clips.

## Citation
DeepVOG has been peer-reviewed and accepted as an original article in the Journal of Neuroscience Method (Elsevier). 
The manuscript is available open access and can be downloaded free of charge [here](https://doi.org/10.1016/j.jneumeth.2019.05.016). If you use DeepVOG or some part of the code, please cite (see [bibtex](citations.bib)):

Yiu YH, Aboulatta M, Raiser T, Ophey L, Flanagin VL, zu Eulenburg P, Ahmadi SA. DeepVOG: Open-source Pupil Segmentation and Gaze Estimation in Neuroscience using Deep Learning. Journal of neuroscience methods. vol. 324, 2019, DOI: https://doi.org/10.1016/j.jneumeth.2019.05.016

## Table of Contents
- [Getting Started](#getting-started)
    - [Installation](#installation)
        - [Using pip](#using-pip)
        - [Using Docker](#using-docker)
    - [Use DeepVOG as CLI](#use-deepvog-as-cli)
    - [Use DeepVOG as python module](#use-deepvog-as-python-module)
- [Documentations](#documentations)
- [Demo code](#demo-code)
- [Limitations](#limitations)
- [Annotation tools](#annotation-tools)
- [Fine-tuning DeepVOG](#fine-tuning-deepvog)
- [Testing](#testing)
- [Authors](#authors)
- [Links to other related papers](#links-to-other-related-papers)
- [License](#license)
- [Acknowledgement](#acknowledgement)




## Getting Started


### Installation

#### Using pip

```bash
# Install DeepVOG via pip
$ pip install -U deepvog

# Verify that it is installed. It should show the help document.
$ python -m deepvog -h
```
#### Using Docker

##### Build from Dockerfile:

Download the [Dockerfile](docker/Dockerfile) from this repository. and run

```bash
# Move to the same directory that contains the Dockerfile
$ docker build -t YOUR_IMAGE_NAME .
```
##### Pull from Docker Hub:
The image is pre-built using the [Dockerfile](docker/Dockerfile) with deepvog pre-installed.
```bash
$ docker pull yyhhoi/deepvog:v1.2.1
```

##### Run your Docker container:
```bash
$ docker run -it --gpus=all yyhhoi/deepvog:v1.2.1 bash

# deepvog package comes pre-installed
$ python -m deepvog -h

```


### Use DeepVOG as CLI
The command-line interface (CLI) allows you to fit/infer single video, or multiple of them by importing a csv table. They can be simply called by:
```
$ python -m deepvog --fit /PATH/video_fit.mp4 /PATH/eyeball_model.json

$ python -m deepvog --infer /PATH/video_infer.mp4 /PATH/eyeball_model.json /PATH/results.csv

$ python -m deepvog --table /PATH/list_of_operations.csv
```
DeepVOG first fits a 3D eyeball model from a video clip. Base on the eyeball model, it estimates the gaze direction on any other videos if the relative position of the eye with respect to the camera remains the same. 

You can fit an eyeball model and infer the gaze directions from the same video clip. For clinical use, users may prefer fitting the eyeball model with a specific calibration paradigm and use it for gaze estimate. <br/>

In addition, you will need to specify your camera parameters such as focal length, if your parameters differ from default values.
```
$ python -m deepvog --fit /PATH/video_fit.mp4 /PATH/eyeball_model.json --flen 12 --vid-shape 240,320 --sensor 3.6,4.8 --batchsize 32 --gpu 0
```



### Use DeepVOG as python module
For more flexibility, you may import the module directly in python.

Run the script [demo.py](demp.py), which will fit a model and infer the gaze from [demo/demo.mp4](demo/demo.mp4).
```python
import deepvog

# Load our pre-trained network
model = deepvog.load_DeepVOG()

# Initialize the class. It requires information of your camera's focal length and sensor size, which should be available in product manual.
inferer = deepvog.GazeInferer(model, focal_length=12, video_shape=(240, 360), sensor_size=(3.6, 4.2)) 

# Fit an eyeball model from "demo.mp4". The model will be stored as the "inferer" instance's attribute.
input_video = 'demo/demo_video_subsampled.mp4'
inferer.process(input_video, mode="Fit", batch_size=2, ransac=ransac)


# After fitting, infer gaze from "demo.mp4" and output the results into "demo_result.csv"
inferer.process(input_video, mode="Infer", 
                output_record_path="demo/demo_output_results.csv", 
                output_video_path="demo/demo_output_video.mp4", batch_size=2)


# You may save the eyeball model to "demo_model.json" for subsequent gaze inference
eye_ball_model_path = "demo/fitted_eyeball_model.json"
inferer.save_eyeball_model(eye_ball_model_path) 

# By loading the eyeball model, you don't need to fit the model again, and can infer the gaze by calling inferer.process(mode='infer') directly.
inferer.load_eyeball_model(eye_ball_model_path) 

```
## Documentations
Please refer to [doc/documentation.md](doc/documentation.md) for the CLI documentation and the required CSV table format. Alternatively, you can also type `$ python -m deepvog -h` for usage examples.


## Demo code

Demo video is located at [demo](demo). After installing DeepVOG, you can move to that directory and run the following commands:

```
$ python -m deepvog --fit ./demo.mp4 ./demo_eyeball_model.json -v ./demo_visualization_fitting.mp4 -m -b 256

$ python -m deepvog --infer ./demo.mp4 ./demo_eyeball_model.json ./demo_gaze_results.csv -b 32 -v ./demo_visualization_inference.mp4 -m
```

The -v argument draws the visualization of fitted ellipse and gaze vector to a designated video. The -m argument draws the segmented heatmap of pupil side by side. The -b argument controls the batch size. For more details of arguments, see [doc/documentation.md](doc/documentation.md).

In the results, you should be able to see the visualization in the generated video "demo_visualization_inference.mp4", as shown below.

<p align="center"> 
<img width="640" height="240" src="demo/demo_result.png">
</p>

In addtion, you can also test out the --table mode by:
```
$ python -m deepvog --table demo_table_mode.csv
```

## Limitations

DeepVOG is intended for pupil segmentation and gaze estimation under the assumptions below:

1. Video contains only single eye features (pupil, iris, eyebrows, eyelashes, eyelids...etc), for example the [demo video](demo). Videos with facial or body features may compromise its accuracy.
2. DeepVOG was intended for eye video recorded by head-mounted camera. Hence, It assumes fixed relative position of the eye with respect to the camera.  

For more detailed discussion of the underlying assumptions of DeepVOG, please refer to the [paper](https://doi.org/10.1016/j.jneumeth.2019.05.016).  

## Annotation tools
See [annotation_tool/README.md](annotation_tool/README.md).

## Fine-tuning DeepVOG
A minimal training script is provided, see [finetune.py](finetune.py). The original training script is no longer available.

## Testing
See [test](test) for module testing.

## Authors

* **Yiu Yuk Hoi** - *Implementation and validation*
* **Seyed-Ahmad Ahmadi** - *Research study concept*
* **Moustafa Aboulatta** - *Initial work*

## Links to other related papers
- [U-Net: Convolutional Networks for Biomedical Image Segmentation
](https://arxiv.org/abs/1505.04597)
- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- [A fully-automatic, temporal approach to single camera, glint-free 3D eye model fitting](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/)

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) License - see the [LICENSE](LICENSE.txt) file for details

## Acknowledgement

We thank our fellow researchers at the German Center for Vertigo and Balance Disorders for help in acquiring data for training and validation of pupil segmentation and gaze estimation. In particular, we would like to thank Theresa Raiser, Dr. Virginia Flanagin and Prof. Dr. Peter zu Eulenburg.

DeepVOG was created with support from the German Federal Ministry of Education and Research (BMBF) in connection with the foundation of the German Center for Vertigo and Balance Disorders (DSGZ) (grant number 01 EO 0901), and a stipend of the Graduate School of Systemic Neurosciences (DFG-GSC 82/3).
