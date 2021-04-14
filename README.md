# Reproducing Fast-RCNN for Apples

## Introduction
In this blog we are going to reproduce Figure 2 from Deep Fruit Detection in Orchards [^deep-fruit-detection]. This paper uses the well-known Faster R-CNN [^fast-rcnn] to detect whether fruit is present in pictures, specifically pictures of orchards. Automatic detection of the presence of fruit can be useful for a variety of applications, including yield mapping and robotic harvesting [^deep-fruit-detection]. The paper's authors mainly focus on detecting apples, mangos, and almonds. However, we limit ourselves to detect apples only.

## Method

### Preprocessing

The VGG-16 network requires the images to be 224 x 224 but the apple images were 202 x 308. In order to have apple images adhere to the required dimensions, we first rescale the images to have a shorter side of 256 pixels and then center crop the middle 224 x 224 pixels. 

### Region Proposal network (RPN)
The project's goal was to implement Faster R-CNN from scratch, using PyTorch. Faster R-CNN is a deep neural network that extends upon VGG16, a network famous for a achieving a high classification accuracy on ImageNet, when it was first published. Because training VGG16 on ImageNet is time-consuming, we used PyTorch's pre-trained VGG16 and removed the final few layers, just like the authors of Deep Fruit Detection in Orchards.

The result of this cut-off version of VGG16 is an output of dimensions 14 x 14 x 512 that serves as an input to the remainder of Faster R-CNN and is called 'feature mapping' from this point onwards. The next step is the Regional Proposal Network (RPN), which receives the feature mapping as its input. The RPN has two distinct outputs; the first being a set of boxes (called 'anchors'), per feature (so 14 x 14 sets of anchors), along with an indication per anchor of how likely it is that an anchor belongs to the foreground (apple) or background (not apple). The second output of the RPN is per anchor a suggestion on how it should be moved and reshaped so as to capture an apple better. This 'suggestion', further referred to as a regression, is formulated as a change in the x and y coordinate of the anchor's center, and a change in its width and height.

For a visual example, see 'Verifying functionality'.

### Region-based Convolutional Neural Network (R-CNN)
The output of the RPN is, for every feature in the feature map and every corresponding anchor, an indication of 'backgroundness' and 'foregroundness' and the regressions. R-CNN is a neural network that takes these region proposals as inputs, and attempts to assign a probability of the region being a specific class or not.

The first step of the R-CNN is Region Of Interest pooling (ROI). This is done in order to ensure equal dimensions of the inputs to the CNN. Recall that RPN's proposals include regressions, or suggestions on how to adjust the anchor's shape and location. These proposals have to be mapped from pixel coordinates to feature coordinates. The proposals in pixel space refer to sub-images in the original image of varying sizes, whereas the feature coordinates can be at most 14 x 14, because that's how many features we start with. In order map these coordinates, we have to analyse how the anchors overlap the feature map boxes; a visualisation of the overlap is presented below. The yellow boxes show the min and max feature coordinates we attributed to that specific region proposal (the blue box).

![roi_mapping](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/roi_features.png)


These feature map boxes (still of varying sizes) are now max-pooled to all adhere to a fixed size of 7 x 7, still of depth 512. The R-CNN consists of 2 fully connected layers so we need to flatten the 7 x 7 x 512 boxes to 1 x 25088 inputs. These are then fed to the 2 fully connected layers which produce an output of 2 entries, corresponding to background and apple probabilities.

In our case, we limited ourselves to apples only (because we disliked mangos and almonds more). Note that the complete implementation of R-CNN is more involved than what we describe and implemented, due to the project's limited scope.

### Simplifications
Due to time constraints and the project's limited scope, we were forced to make some simplifications, thereby deviating from Faster R-CNN and the Deep Fruit paper. These devations are listed below.

- Apples only: We restrict ourselves to train only for detecting apples, instead of apples, mangos, and almonds.
- We only train the network on images that contain at least one apple.
- One anchor per feature: We restrict ourselves to have Faster R-CNN train only using the first indicated apple in a picture. Images in the dataset often contain multiple objects and therefore mulitple labels. This complicates the calculation of the IoU metric for the bounding boxes drastically, as this requires us to find the nearest ground-truth box for every anchor, before being able to calculate the IoU of an anchor. In this project we therefore decide to train on only 1 object per image. This is expected to worsen performance, because the RPN will be trained to classify the other apples in a picture as background.
- The full implementation of R-CNN adds another set of regressions and trains these, too. We have decided to omit this second round of regression calculations.
- For training the detector we deviate from the procedure of the paper, section Training elaborates upon this.

### Verifying Functionality
In order to verify that the implementation of the RPN is correct we perform an experiment in which we test the network on the same image it was trained on. This should cause an overfit to this specific training image and thus result in high accuracy.

This first experiment only considers the region proposals provided as output by the RPN. In the figure below (1), it can be observed that the RPN is able to learn which of the anchors overlap with the ground truth bounding box (the apple, marked in green). Note that all bounding boxes are rectangular in shape, though in the image they overlap.  

The second experiment takes the regressions of the alleged foreground boxes into account. From the results, displayed in the figure below (2), it becomes apparent that the RPN is also able to learn the anchor regressions when it overfits to one image.

![overfit_1](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/overfit1.png)
![overfit_2](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/overfit2.png)

In order to verify functionality of the entire pipeline, thus including the R-CNN, we want to perform another overfitting test. Here we show that the box that the R-CNN attributes the highest probability of capturing an apple, actually fits the apple quite well.
![overfit_3](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/overfit3.png)

### Training 

Due to not having access to required resources we regarded the procedure described in [^fast-rcnn-other] infeasable. Instead, we use randomly selected (image-)batches of size 30 each iteration and let the network train for 15000 iterations.

## Results

In this section we analyze the outputs of images in which the network claims to have found an apple with high probability (>80%). In these images we drew the 5 boxes to which the R-CNN attributes the highest probability of containing an apple. The results are displayed below. In some images (the first 4) the detector has failed miserably. To the contrary, in other images (the last 4) the detector seems to have learned how to detect an apple properly.

![bad_1](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/bad1.png)
![bad_2](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/bad2.png)
![bad_3](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/bad3.png)
![bad_4](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/bad4.png)
![good_1](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/good1.png)
![good_2](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/good2.png)
![good_3](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/good3.png)
![good_4](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/good4.png)
 
### Discussion


From the results it's visible that our simplified implementation performs a lot worse than the paper's. This was to be expected due to the many concessions we had to make in favour of time and complexity. We conjecture that the most severe performance degradation stems from the fact that only the first label of the apples is used for detecting foreground in the RPN. This means that other apples are regarded as background and this does not support learning of detecting apples. Another thing that we expect has a discouraging effect on learning is that some ground truth boxes might be cut out of the image due to the center crop operation.


[^deep-fruit-detection]: Bargoti, S., & Underwood, J. (2017, May). Deep fruit detection in orchards. In 2017 IEEE International Conference on Robotics and Automation (ICRA) (pp. 3626-3633). IEEE.

[^fast-rcnn]: Girshick, R. (2015). Fast r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 1440-1448).


[^fast-rcnn-other]: Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. arXiv preprint arXiv:1506.01497.

## Running 
The code of this project is provided on: `https://github.com/KoningR/faster-rcnn-for-apples`. In order to run the code, you should have the files and folders adhere to the structure displayed in the screenshot below.

![file_structure](https://github.com/koningr/faster-rcnn-for-apples/blob/main/images/files.png)

The `dataset/images` folder contains apple images only and `dataset/annotations` contains the corresponding labels as `csv` files. `train.txt`and `test.txt` contain the desired filenames (without ".png")

It is possible to store and load trained weight configurations. The code includes `load_model` and `save_model` boolean options to configure this. 