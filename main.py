import gc
import itertools
from random import sample

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageDraw
from torchvision import models


def load_images(filename, device, training=True):
    with open(filename) as reader:
        lines = reader.readlines()
        count = len(lines)

        # Initialise empty matrices and extend them, one row per image that contains an apple.
        data = torch.zeros(0, 3, 224, 224, device=device)
        boxes = torch.zeros(0, 4, device=device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Randomly sample at most 40 images per batch when training.
        if training:
            lines = sample(lines, min(40, count))

        for image_file in lines:
            # Only load annotations if we are training.
            if training:
                try:
                    row = pd.read_csv("dataset/annotations/" + image_file[:-1] + ".csv").iloc[0]
                except IndexError:
                    # Skip if there is no foreground/apple in the image.
                    continue

                # Parse the annotations.
                true_x, true_y, true_rad = scale_label(row[1], row[2], row[3])

                # Calculate the foreground box in pixel coordinates.
                image_box = torch.empty((1, 4), device=device)
                image_box[0, 0] = true_x - true_rad
                image_box[0, 1] = true_y - true_rad
                image_box[0, 2] = true_x + true_rad
                image_box[0, 3] = true_y + true_rad

                boxes = torch.cat((boxes, image_box), dim=0)

            image = Image.open("dataset/images/" + image_file[:-1] + ".png").convert('RGB')
            prep_image = preprocess(image).unsqueeze(0).to(device)

            data = torch.cat((data, prep_image), dim=0)

    # Hacky way of retrying when there were no foreground images in the batch.
    # You can use this for very small batches (BUT BE CAUTIOUS)
    # if data.shape[0] == 0:
    #     return load_images(filename, device, training)

    return data, boxes


def get_vgg16_features():
    # Load a pretrained VGG and cut off the last few layers.
    vgg16 = models.vgg16(pretrained=True)
    feature_extractor = nn.Sequential(*list(vgg16.features.children())[:-1])

    return feature_extractor


def scale_label(x_center, y_center, radius):

    # Because the images are cropped, so should the labels.

    # Assume smallest side is always 202 pixels.
    # 1.28 * 202
    img_height = 256
    # 1.28 * 308
    img_width = 388

    y_center *= 1.27
    y_center = y_center - ((img_height - 224) / 2)

    x_center *= 1.27
    x_center = x_center - ((img_width - 224) / 2)

    radius *= 1.27

    return x_center, y_center, radius


def cuda_iou(boxes1, boxes2):
    # Adapted from https://medium.com/@venuktan/vectorized-intersection-
    # over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    # Modified for Torch CUDA.

    x11, y11, x12, y12 = torch.split(boxes1, 1, dim=1)
    x21, y21, x22, y22 = torch.split(boxes2, 1, dim=1)
    xA = torch.max(x11, x21.T)
    yA = torch.max(y11, y21.T)
    xB = torch.min(x12, x22.T)
    yB = torch.min(y12, y22.T)

    interArea = torch.clamp((xB - xA + 1), min=0) * torch.clamp((yB - yA + 1), min=0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea.T - interArea)
    return iou


def anchors_to_feature_subset(image_index, x_index, y_index, regression, feature_mapping):
    # Map the anchor centers to pixel coordinates.
    roi_anchor_centers = torch.stack((x_index, y_index), dim=0).T
    roi_anchor_centers = roi_anchor_centers * 16

    # Get the regressions corresponding to the anchors.
    anchor_deltas = regression[
                    image_index, :,
                    x_index,
                    y_index]

    # Calculate the boxes with the regressions added.
    x_min = roi_anchor_centers[:, 0] - half_size + anchor_deltas[:, 0] - anchor_deltas[:, 2] / 2
    x_max = roi_anchor_centers[:, 0] + half_size + anchor_deltas[:, 0] + anchor_deltas[:, 2] / 2
    y_min = roi_anchor_centers[:, 1] - half_size + anchor_deltas[:, 1] - anchor_deltas[:, 3] / 2
    y_max = roi_anchor_centers[:, 1] + half_size + anchor_deltas[:, 1] + anchor_deltas[:, 3] / 2
    anchor_coords = torch.stack((x_min, y_min, x_max, y_max), dim=0).T

    # Map the regressed boxes back to feature space.
    regressed_feature_coords = torch.clamp(torch.floor(anchor_coords / 16), 0, 13).int()

    adaptive_max_pool = nn.AdaptiveMaxPool2d((7, 7))
    roi_matrix = torch.empty(regressed_feature_coords.shape[0], 512, 7, 7, device=device)

    # Fill the ROI matrix.
    for feature_range in range(regressed_feature_coords.shape[0]):
        x_min_index, y_min_index, x_max_index, y_max_index = regressed_feature_coords[feature_range, :]

        # Get the features corresponding to the regressed boxes from the feature mapping.
        # Add 1 because the upper bound of the range is exclusive.
        feature_subset = feature_mapping[
              image_index[feature_range], :,
              x_min_index:x_max_index + 1,
              y_min_index:y_max_index + 1]

        # Map the features to a 7 x 7 matrix.
        roi_matrix[feature_range] = adaptive_max_pool(feature_subset)

    return roi_matrix


def divmod(tensor, divisor):
    # Helper function for calculating divmod with PyTorch.
    divs = tensor // divisor
    mods = torch.remainder(tensor, divisor)

    return divs, mods


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.convy = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0)
        self.convz = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y = self.convy(x)
        z = self.convz(x)
        return y, z


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()

        self.lin1 = nn.Linear(512 * 7 * 7, 4096)
        self.lin2 = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        return x


if __name__ == '__main__':
    epochs = 0
    save_model = False
    load_model = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features = get_vgg16_features()
    rpn = RPN()
    rcnn = RCNN()

    features = features.to(device)
    rpn = rpn.to(device)
    rcnn = rcnn.to(device)

    if load_model:
        rpn_state = torch.load("rpn_model", map_location=device)
        rpn.load_state_dict(rpn_state["state_dict"])

        rcnn_state = torch.load("rcnn_model", map_location=device)
        rcnn.load_state_dict(rcnn_state["state_dict"])

    # Create all initial anchor boxes.
    size = 64
    half_size = size / 2
    boxes_matrix = torch.ones((196, 4), device=device)
    for i in range(14):
        for j in range(14):
            index = 14 * i + j
            boxes_matrix[index, :] = torch.tensor([16 * i - half_size, 16 * j - half_size,
                                               16 * i + half_size, 16 * j + half_size])

    # Start training.
    for i in range(epochs):
        print("Starting epoch", i)

        train_data, true_boxes = load_images("train.txt", device)

        # Obtain VGG feature mapping and perform the RPN forward pass.
        feature_tensor = features(train_data)
        (classifications, regressions) = rpn(feature_tensor)

        # Calculate IOUs for all true boxes.
        iou = cuda_iou(true_boxes, boxes_matrix)

        # Make a matrix of all coordinates of boxes that are 'foreground'.
        foreground_y, foreground_x = divmod(torch.nonzero((iou > 0.5), as_tuple=False), 14)
        foreground_coords = torch.cat((foreground_x, foreground_y), dim=1)

        # Make a matrix of all coordinates of boxes that are 'background'.
        background_y, background_x = divmod(torch.nonzero((iou < 0.1), as_tuple=False), 14)
        background_coords = torch.cat((background_x, background_y), dim=1)

        # Calculate the number of foreground boxes per image.
        foreground_images, foreground_counts = torch.unique_consecutive(foreground_coords[:, 0], return_counts=True)

        # Sample just as many background boxes per image as there are foreground boxes.
        sampled_background_features = torch.empty((0, 4), device=device)
        # Go through all images.
        for j in range(foreground_images.shape[0]):
            # Get all background boxes corresponding to the current image.
            foreground_sample_count_in_image = foreground_counts[j]
            background_samples_in_image = background_coords[background_coords[:, 0] == j]

            # Randomly take as many as needed.
            sampled_background_features_indices = torch.randperm(
                background_samples_in_image.shape[0])[:foreground_sample_count_in_image]
            sampled_background_features_in_image = background_samples_in_image[sampled_background_features_indices]

            sampled_background_features = torch.cat((sampled_background_features, sampled_background_features_in_image),
                                                    dim=0)

        # Combine foreground and background boxes for a matrix of all boxes for training.
        batch_feature_indices = torch.cat((foreground_coords, sampled_background_features), dim=0).long()

        y_pred_classification = classifications[
                                batch_feature_indices[:, 0], :,
                                batch_feature_indices[:, 3],
                                batch_feature_indices[:, 1]]
        y_true_classification = torch.cat((torch.ones(foreground_coords.shape[0], device=device),
                                           torch.zeros(sampled_background_features.shape[0], device=device)), dim=0).long()

        # Regression part.

        # Map the foreground boxes to pixel coordinates.
        fore_anchor_centers = torch.stack((foreground_coords[:, 3], foreground_coords[:, 1]), dim=0).T
        fore_anchor_centers = fore_anchor_centers * 16

        # Calculate the center of the true foreground box in each image.
        true_x_center = torch.abs(true_boxes[:, 0] + true_boxes[:, 2]) / 2
        true_y_center = torch.abs(true_boxes[:, 1] + true_boxes[:, 3]) / 2

        # Calculate the required change needed to shift the anchor centers to the true box.
        delta_x = true_x_center[foreground_coords[:, 0]] - fore_anchor_centers[:, 0]
        delta_y = true_y_center[foreground_coords[:, 0]] - fore_anchor_centers[:, 1]

        # Calculate the (x_max, y_max) coordinate (bottom-right corner) of each foreground box.
        pred_max_coords_in_batch = torch.stack(
            (fore_anchor_centers[:, 0] + delta_x, fore_anchor_centers[:, 1] + delta_y),
            dim=0).T + half_size
        true_max_coords_in_batch = true_boxes[foreground_coords[:, 0]][:, 2:]
        # Calculate the required change needed to reshape the boxes to the true box.
        delta_wh = (true_max_coords_in_batch - pred_max_coords_in_batch) * 2

        y_true_regression = torch.cat((torch.stack((delta_x, delta_y), dim=0).T, delta_wh), dim=1)
        y_pred_regression = regressions[
                            foreground_coords[:, 0], :,
                            foreground_coords[:, 3],
                            foreground_coords[:, 1]]

        # ROI part.
        roi_features = anchors_to_feature_subset(batch_feature_indices[:, 0],
                                                 batch_feature_indices[:, 3],
                                                 batch_feature_indices[:, 1],
                                                 regressions,
                                                 feature_tensor)

        # Flatten the ROI features to match the dimensions of the R-CNN.
        rcnn_pred = rcnn(torch.flatten(roi_features, start_dim=1))

        # Loss calculations.

        # Optimise both RPN and R-CNN.
        optimizer = optim.SGD(itertools.chain(*[rpn.parameters()] + [rcnn.parameters()]), lr=0.01)
        optimizer.zero_grad()

        classification_loss_function = nn.CrossEntropyLoss()
        classification_loss = classification_loss_function(y_pred_classification, y_true_classification)

        regression_loss_function = nn.SmoothL1Loss(size_average=None, reduce=None, reduction="mean")
        regression_loss = regression_loss_function(y_pred_regression, y_true_regression)

        rcnn_loss_function = nn.CrossEntropyLoss()
        rcnn_loss = rcnn_loss_function(rcnn_pred, y_true_classification)

        combined_loss = classification_loss + regression_loss + rcnn_loss
        combined_loss.backward(retain_graph=True)

        optimizer.step()

        if save_model and i % 500 == 0 and i > 0:
            print("Saving model parameters...")

            rpn_state = {
                "state_dict": rpn.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(rpn_state, "rpn_model")

            rcnn_state = {
                "state_dict": rcnn.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(rcnn_state, "rcnn_model")

        print(combined_loss)
        print("---------------")

    # Testing.
    test_data, true_boxes = load_images("test.txt", device, False)
    test_data = test_data.to(device)
    true_boxes = true_boxes.to(device)

    # Perform the VGG and RPN forward pass.
    test_feature_tensor = features(test_data)
    (test_classifications, test_regressions) = rpn(test_feature_tensor)

    # Create a matrix containing all possible anchors for all test images.
    index_list = torch.empty((0, 3), device=device)

    y_index, x_index = divmod(torch.arange(0, 14 * 14, device=device), 14)
    anchor_index_list = torch.stack((x_index, y_index), dim=0).T

    for image in range(test_regressions.shape[0]):
        image_index = torch.ones((14 * 14), device=device) * image
        image_index = torch.stack((image_index, x_index, y_index), dim=0).T
        index_list = torch.cat((index_list, image_index), dim=0).long()

    # Perform ROI pooling.
    roi_test_features = anchors_to_feature_subset(index_list[:, 0],
                                                  index_list[:, 1],
                                                  index_list[:, 2],
                                                  test_regressions,
                                                  test_feature_tensor)

    # Perform R-CNN forward pass.
    apples = rcnn(torch.flatten(roi_test_features, start_dim=1))
    soft_max = nn.Softmax(dim=1)
    apples = soft_max(apples)

    # Find the boxes with the k highest probabilities of being an apple.
    topk_values, topk_indices = torch.topk(apples[:, 1], 5)

    topk_ys, topk_xs = divmod(topk_indices, 14)
    print("Top k highest probabilities:", topk_values)

    # Render 1 image with corresponding boxes.

    image = Image.open("dataset/images/" + "20130320T005032.000483.Cam6_24" + ".png").convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    prep = preprocess(image)
    test_image = ImageDraw.Draw(prep)

    # Go through all best boxes.
    for xy in range(topk_xs.shape[0]):
        x = topk_xs[xy]
        y = topk_ys[xy]

        # Add the original anchor box in red.
        test_image.rectangle((x * 16 - half_size, y * 16 - half_size, x * 16 + half_size, y * 16 + half_size),
                             outline="red")

        # Add the regressed box in orange.
        test_deltas = test_regressions[0, :, x, y]
        test_x_offset = test_deltas[0]
        test_y_offset = test_deltas[1]
        test_delta_width = test_deltas[2]
        test_delta_height = test_deltas[3]
        test_image.rectangle((x * 16 - half_size + test_x_offset - test_delta_width / 2,
                              y * 16 - half_size + test_y_offset - test_delta_height / 2,
                              x * 16 + half_size + test_x_offset + test_delta_width / 2,
                              y * 16 + half_size + test_y_offset + test_delta_height / 2), outline="orange")

    prep.show()
    prep.save("results.png")

    # Perform naive cleanup.
    train_data = None
    test_data = None
    true_boxes = None
    boxes_matrix = None
    features = None
    rpn = None
    rcnn = None

    torch.cuda.empty_cache()
    gc.collect()
