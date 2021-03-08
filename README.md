
# Facial Expression Recognition
### A simple deep learning project for recognizing human expressions.</br>

*Follow the jupyter notebook for code execution and details.*

* Pytorch implementation : [Notebook](facial-expression-recognition-in-pytorch.ipynb)

* Tensorflow implementation : [Notebook](facial-expression-recognition-in-tensorflow.ipynb)

After completing training, make sure to save the model as a [json](model.json) file and model weights as a [h5](model_weights.h5) file since they are used in the flask code.

I'm using tensorflow during the test-time prediction, refer [model.py](model.py) for complete details. You can specify the videos in [camera.py](camera.py) line no 14 and play around with it. *If you're cloning this project make sure to get a good accuracy for better performance* :alien: .

## Setting up environment

The below steps will guide you towards running the project in your local.

1. Clone this repo.
2. Create a virtual environment ```pip3 install virtualenv``` :point_right: ```virtualenv env``` :point_right: ```source env/bin/activate```
3. pip install these packages:[flask, opencv-python, tensorflow, numpy]
4. Run your project : ```python3 main.py```

## Demo video:
https://user-images.githubusercontent.com/55920093/110316776-3243f780-8031-11eb-8bfb-24556efed78c.mp4
