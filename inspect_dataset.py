import os
from collections import Counter
import pandas as pd


# ==========================
# Modify this path
# ==========================

DATASET_PATH = r"D:\\prog\\datasets\\Unbiased Tiny GenImage"


# ==========================
# Image extensions
# ==========================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ==========================
# Scan dataset
# ==========================

records = []


for generator in os.listdir(DATASET_PATH):

    folder_path = os.path.join(
        DATASET_PATH,
        generator
    )

    # skip non-folder
    if not os.path.isdir(folder_path):
        continue


    count = 0
    formats = Counter()


    for filename in os.listdir(folder_path):

        ext = os.path.splitext(filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:

            count += 1
            formats[ext] += 1


    records.append({

        "generator": generator,

        "image_count": count,

        "jpg": formats[".jpg"],

        "jpeg": formats[".jpeg"],

        "png": formats[".png"]

    })


# ==========================
# Save result
# ==========================


df = pd.DataFrame(records)


print("\nDataset Summary:")
print(df)


df.to_csv(
    "dataset_statistics.csv",
    index=False
)


print(
    "\nSaved: dataset_statistics.csv"
)