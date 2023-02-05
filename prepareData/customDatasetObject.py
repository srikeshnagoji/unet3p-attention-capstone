
from torch.utils.data import Dataset
import cv2 
import albumentations as A
from albumentations.pytorch import ToTensor

class BrainMRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)
        
        augmented = self.transform(image=image,
                                   mask=mask)
        
        image = augmented["image"]
        mask = augmented["mask"]
#         mask = np.expand_dims(augmented["mask"], axis=0)# Do not use this
        
        return image, mask

    def get_image_and_mask(self, idx):
        image = cv2.imread(self.df.loc[idx, "image_path"])
        mask = cv2.imread(self.df.loc[idx, "mask_path"], 0)
        PATCH_SIZE = 128
        trans_ = A.Compose([
            A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
            ToTensor(),
        ])

        augmented = trans_(image=image,
                                   mask=mask)
        
        image = augmented["image"]
        mask = augmented["mask"]

        
        return image, mask