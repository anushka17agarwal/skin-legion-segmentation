import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
class SkinData(Dataset):
    def __init__(self, path, transform=None):
        self.image_dir = os.path.join(path, "image_data")
        self.mask_dir = os.path.join(path,"groundtruth" )
        self.image1= os.listdir(self.image_dir)
        self.image= []
        for i in self.image1:
            if(i[-3:] == "jpg"):
                self.image.append(i)
        self.mask = os.listdir(self.mask_dir)
        self.transform = transform
        
        

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        #print(self.image)
        img_path= os.path.join(self.image_dir, self.image[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image= image.astype(float)
        mask_path = os.path.join(self.mask_dir, self.mask[index])
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
       
