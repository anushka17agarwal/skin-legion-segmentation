import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from utils import create_dir
import segmentation_models_pytorch as smp
from PIL import Image
create_dir("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



#Calculate mterics

def calculate_metrics(y_true, y_pred):
    #ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    #prediction
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask



#model

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1, 
    activation=ACTIVATION,
)
model = model.to(DEVICE)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


path= "test_data"
test_image_dir = os.path.join(path ,"image_data")
test_mask_dir = os.path.join(path,"groundtruth" )
# print(test_image_dir, mask_dir)

image1= os.listdir(test_image_dir)
# print(image1)
image= []
for i in image1:
            if(i[-3:] == "jpg"):
                image.append(i)
mask = os.listdir(test_mask_dir)

test_x = []
test_y= []
for i in image:
    x= os.path.join(test_image_dir, i)
    test_x.append(x)
for m in mask:
   x= os.path.join(test_mask_dir, i.replace(".jpg", "_segmentation.png"))
   test_y.append(x)

print("mask 1", test_y[1])
""" Hyperparameters """
H = 160
W = 240
size = (W, H)
checkpoint_path = "model"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
time_taken = []

for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        # print(x, y)  
        image = np.array(Image.open(x).convert("RGB"))
  
        image = cv2.resize(image, size)
        #print(image.shape)
        x = np.transpose(image, (2, 0, 1))      ## (3, 160, 240))
        
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        #print(x.shape)
          
        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
        #print(y.shape)


        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            
 
            score = calculate_metrics(y, pred_y)
            metrics_score= list(map(add, metrics_score, score))
            print(metrics_score)
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)


        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results/{name}.png", cat_images)

        


        
