import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--GPU', help = "Option to use GPU. Optional", type = str)

def loading_model (file_path):
    checkpoint = torch.load (file_path) #loading checkpoint from a file
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: #vgg13 as only 2 options available
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False #turning off tuning of the model

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    py_img = transform_img(im)
    return py_img

def predict(image_path, model, topk=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    if device == 'cuda':
        image = image.cuda()
    with torch.no_grad ():
        output = model.forward(image)
    output_prob = torch.exp (output)
    probs, indeces = output_prob.topk (topk)
    probs = probs.cpu().numpy() 
    indeces = indeces.cpu().numpy() 
    
    probs = probs.tolist()[0]
    indeces = indeces.tolist()[0]
    
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    classes = [mapping [item] for item in indeces]
    classes = np.array (classes)
    
    return probs, classes

#setting values data loading
args = parser.parse_args ()
file_path = args.image_dir
print(file_path)
#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model = loading_model (args.load_dir)
print(model)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#calculating probabilities and classes
probs, classes = predict (file_path, model, nm_cl, device)

#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )