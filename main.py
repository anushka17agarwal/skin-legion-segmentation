from unicodedata import decimal
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data import SkinData
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from model import DeepLabv3
import segmentation_models_pytorch as smp

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS= 2
IMAGE_HEIGHT= 160
IMAGE_WIDTH= 240
PIN_MEMORY= True
LOAD_MODEL = True
train_path= "train_data"
test_path= "test_data"
print(DEVICE)
#transformation
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)
val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
train_dataset = SkinData(train_path, transform=train_transform)
test_dataset= SkinData(test_path, transform= val_transforms)
print(len(test_dataset))
train_loader= DataLoader(train_dataset, batch_size= 10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size= 10, shuffle= False)
dataiter= iter(test_loader)
images, labels = dataiter.next()
print(images.shape)
#model = DeepLabv3().to(DEVICE)
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
print("Hi Let's train")
# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1, 
    activation=ACTIVATION,
)
model = model.to(DEVICE)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

trainSteps = len(train_dataset) // 10
testSteps = len(test_dataset) // 10

loss_fn = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr= LEARNING_RATE)
NUM_EPOCHS=15
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

for e in tqdm(range(NUM_EPOCHS)):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	best_val_loss= 100
	# loop over the training set
	for (i, (x, y)) in enumerate(train_loader):
		# send the input to the device
        
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
        
		# perform a forward pass and calculate the training loss
        
		pred = model(x)
        
		loss = loss_fn(pred, y.unsqueeze(1))
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# # switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# num_correct = 0
		# num_pixels = 0
		#dice_score = 0
		#acuracy = 0

		#print("entering evaluation mode")
		# loop over the validation set
		for (x, y) in test_loader:
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
            #y = y.unsqueeze(1)
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalTestLoss += loss_fn(pred, y.unsqueeze(1))
			#print("prediction made")
			pred = (pred > 0.5).float()
			#num_correct += (pred == y).sum()
			#num_pixels += torch.numel(pred)
			#dice_score += (2 * (pred * y).sum()) / ((pred + y).sum() + 1e-8)
		# print(num_correct, "No correct", num_pixels,)
		# Accuracy= num_correct/ num_pixels 
		# print("Accuracy", Accuracy)
		# dice = dice_score/len(test_loader)
		# print("DICE", dice )

		

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	#H["train_loss"].append(avgTrainLoss.numpy())
	#H["test_loss"].append(avgTestLoss.numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
	path= "model"
	if(avgTestLoss < best_val_loss):
		print("Before saving", best_val_loss)
		best_val_loss= avgTestLoss
		print("after saving", best_val_loss)
		torch.save(model, path)