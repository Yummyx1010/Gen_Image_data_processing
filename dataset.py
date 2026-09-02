import os
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



# ==========================
# Dataset path
# ==========================

DATASET_ROOT = r"D:\prog\datasets\Unbiased Tiny GenImage"



# ==========================
# Shared preprocessing
# ==========================

transform = transforms.Compose([

    # ensure RGB
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),


    # resize + crop
    transforms.Resize(256),

    transforms.CenterCrop(224),


    # PIL -> Tensor
    transforms.ToTensor(),


    # ImageNet normalization
    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])



# ==========================
# Dataset class
# ==========================

class GenImageDataset(Dataset):


    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None
    ):


        self.data = pd.read_csv(
            csv_file
        )

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):


        row = self.data.iloc[idx]


        img_path = os.path.join(

            self.root_dir,

            row["image_path"]

        )


        image = Image.open(
            img_path
        )


        label = int(
            row["label"]
        )


        generator = row["generator"]



        if self.transform:

            image = self.transform(
                image
            )


        return {

            "image": image,

            "label": label,

            "generator": generator,

            "path": row["image_path"]

        }



# ==========================
# Test
# ==========================


if __name__ == "__main__":


    dataset = GenImageDataset(

        csv_file="train.csv",

        root_dir=DATASET_ROOT,

        transform=transform

    )


    loader = DataLoader(

        dataset,

        batch_size=4,

        shuffle=True

    )


    batch = next(
        iter(loader)
    )


    print(
        "Image shape:",
        batch["image"].shape
    )


    print(
        "Labels:",
        batch["label"]
    )


    print(
        "Generators:",
        batch["generator"]
    )