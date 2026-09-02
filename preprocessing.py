from PIL import Image
from torchvision import transforms
import os


# =========================
# Dataset path
# =========================

DATASET_PATH = r"D:\prog\datasets\Unbiased Tiny GenImage"


# =========================
# Preprocessing pipeline
# =========================

preprocess = transforms.Compose([

    # ensure RGB
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),

    # resize shorter side to 256
    transforms.Resize(256),

    # center crop 224x224
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


# =========================
# Test function
# =========================

def process_image(image_path):

    image = Image.open(
        image_path
    )

    tensor = preprocess(
        image
    )

    return tensor



# =========================
# Test images
# =========================

test_images = [

    # 改成你的真实图片路径
    r"D:\prog\datasets\Unbiased Tiny GenImage\ADM\0_adm_85.jpg",

    r"D:\prog\datasets\Unbiased Tiny GenImage\BigGAN\000_biggan_00093.jpg",

    r"D:\prog\datasets\Unbiased Tiny GenImage\Nature\n13054560_1470.jpeg"

]


for img_path in test_images:

    if os.path.exists(img_path):

        tensor = process_image(
            img_path
        )

        print(
            img_path
        )

        print(
            tensor.shape
        )

        print("----------------")


    else:

        print(
            "File not found:",
            img_path
        )