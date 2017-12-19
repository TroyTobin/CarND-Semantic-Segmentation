# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Implementation
The steps for this project are as follows
 - Load pre-trained VGG-16 model
 - Load in the initial input layers of the VGG model
   - effectively removing the fully connected layers
 - Perform several layers of 1x1 convolutions
 - Train the neural network, optimizing for reduction of loss.
   - Train on labeled test images
 - Save the trained model
 
### Testing
 - The saved model can be loaded and used to "classify" the pixels in an input image
 - The implementation is able to process either a single image or a video file
  
### Running
The main process is parameterized on the commandline, making its operation flexible.  It also allows a number of different training parameters to be used.  This includes specifying the optimizer, keep rate, learning rate, epochs and batch size.

```
usage: main.py --mode <train|classify> --model_file <file> --video <video file> --image <image file> --out_file <output file> --epochs <num> --batch_size <size> --keep_prob <prob> --optimizer <name> --learn_rate <rate>
```
#### Training the network
When training, the progam must be run with the following set,
 - optimizer
 - keep rate
 - learning rate
 - epochs
 - batch size

For example,
```
python main.py --mode train --epochs 20 --batch_size 3 --keep_prob 0.7 --learn_rate 0.0001 --optimizer 'Momentum'
```
The effect is,
 - A new model is trained and saved as `vgg_seg_model_<epoch>_<batch_size>_<keep_rate>_<learn_rate>_<optimizer>`
 - A graph is created showing the loss at each epoch
 - Sample test images are run showing the semantic segmentation result.
 
#### Results 
Running several different training parameters the following was found.
 - The best optimizer out of (Adam, Adagrad, Adadelta, GradientDescent, Momentum) was `Momentum`
 - The best training parameters were as follows,
   - Keep rate `0.75`
   - Learning rate `0.0005 `
   - Momentum `0.9`
   - Batch size `3`
   - Epochs `20`
   
   
![alt text](https://github.com/TroyTobin/CarND-Semantic-Segmentation/blob/master/loss_vs_epoch_0.750000_0.000500.png "Momentum Loss")
![alt text](https://github.com/TroyTobin/CarND-Semantic-Segmentation/blob/master/um_000017a.png "Momentum output")


#### Running the network
The program is able to run semantic segmentation over either a single image, or over a video file.
To run, the user must specify the following, optimizer, model file, input image/video, output image/video

For example,
```
python main.py --optimizer Adam --mode classify --video data\videos\project_video.mp4 --out_file project_processed_adam.mp4 --model_file vgg_seg_model_15_3_0.500000_0.000100_Adam
```


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
