from unicodedata import decimal
import torch, gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import DeepLabv3
from data import SkinData
from torch.utils.data import Dataset, DataLoader
from utils import (

    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
gc.collect()
torch.cuda.empty_cache()

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCH= 10
NUM_WORKERS= 1
IMAGE_HEIGHT= 160
IMAGE_WIDTH= 240
PIN_MEMORY= True
LOAD_MODEL = True
train_path= "train_data"
test_path= "test_data"

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

train_loader= DataLoader(train_dataset, batch_size= 3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= 3, shuffle=True)
dataiter= iter(test_loader)
images, labels = dataiter.next()
print(images.shape)

def train_fn(loader, model, optimizer, loss_fn, scaler ):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data= data.to(device= DEVICE).float()
        targets= targets.float().unsqueeze(1).to(device =  DEVICE)
               #forward


        with torch.cuda.amp.autocast():
            predictions= model(data)
            loss= loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        #update tqdm loop
        loop.set_postfix(loss= loss.item())


model = DeepLabv3().to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
for epoch in range(NUM_EPOCH):
    train_fn(train_loader, model, optimizer, loss_fn, scaler)
    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    #check accuracy
    check_accuracy(test_loader, model, device=DEVICE)

    # print some examples to a folder
    save_predictions_as_imgs(
        test_loader, model, folder="saved_images/", device=DEVICE
    )
