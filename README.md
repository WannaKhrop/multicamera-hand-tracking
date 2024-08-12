# Multicamera Handtracking

## Description
This project is aimed at stabilization of mediapipe handtracking results.
We use several deep cameras Intel RealSense to capture images of a hand from different
positions and make fusion to get preciser results.

## Getting started

### Poetry
To run this project you need to install Python environment manager [Poetry](https://python-poetry.org/docs/).
After copying a repository you need to install release dependencies using the following command:
```
poetry install --without dev
```

If you are going to make changes in the repository, then you have to execute the following commands:
```
poetry install
pre-commit install
```

### MediaPipe
For any type of use you have to download a model from the official web-site of [MediaPipe] (https://ai.google.dev/edge/mediapipe/solutions/guide).

MediaPipe models are organized in file with .task extention. This solution is configured for model that detects [hand landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models).

You need to put this file (.task) in the folder "models". 

### Intel RealSense
To run this project you need to make sure that deep cameras Intel RealSense are connected to your PC and do not disturb each other. This check can be done using RealSense Tools.

The latest version of RealSense SDK and RealSense Viewer can be found [here](https://github.com/IntelRealSense/librealsense/releases/tag/v2.55.1).

### Camera calibration
Before running the project it's required to calibrate available cameras and find transformations
martrixes from camera coordinates to world coordinates for each camera.

You need to store points clouds during your calibration procedure for each camera in a .csv file as:
```
camera_x,camera_y,camera_z,world_x,world_y,world_z
c_x1,c_y1,c_z1,w_x1,w_y1,w_z1
c_x2,c_y2,c_z2,w_x2,w_y2,w_z2
...
```
Minimal amount of points is four. However, the larger the amoumt of points, the better 
transformation will be found.

You need to put these files in data-folder and run script to find a transformation.
```
poetry run python find_transformation.py [--mode ...]
```
Possible modes are: mse and umeyama.

After that the results will be stored in transformations-folder for each camera.

### Run the project
Use the following command to start the project:
```
poetry run python main.py
```

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

Do not forget to write your changes in this README.md

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Authors and acknowledgment
Author: Ivan Khrop

Supervisor: Fabian Mikula

This development was prepared in terms of Large Master Project at the University of Bayreuth.

Year 2024.

## License
Feel free to use it.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
