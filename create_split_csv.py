import os
import random
import pandas as pd


# ==========================
# Dataset path
# ==========================

DATASET_PATH = r"D:\prog\datasets\Unbiased Tiny GenImage"


# ==========================
# Generator split
# ==========================

TRAIN_GENERATORS = [
    "ADM",
    "BigGAN",
    "stable_diffusion_v_1_5"
]


VAL_GENERATORS = [
    "VQDM"
]


TEST_GENERATORS = [
    "glide",
    "Midjourney",
    "wukong"
]


REAL_FOLDER = "Nature"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ==========================
# Random seed
# ==========================

random.seed(42)



records = []



# ==========================
# Scan fake generators
# ==========================

def collect_generator_images(generator, split):

    folder = os.path.join(
        DATASET_PATH,
        generator
    )

    for filename in os.listdir(folder):

        ext = os.path.splitext(filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:

            records.append({

                "image_path":
                    os.path.join(
                        generator,
                        filename
                    ),

                "label": 1,

                "generator":
                    generator,

                "split":
                    split

            })



# Train fake

for g in TRAIN_GENERATORS:

    collect_generator_images(
        g,
        "train"
    )



# Validation fake

for g in VAL_GENERATORS:

    collect_generator_images(
        g,
        "validation"
    )



# Test fake

for g in TEST_GENERATORS:

    collect_generator_images(
        g,
        "test"
    )



# ==========================
# Collect real images
# ==========================


real_records = []


real_folder = os.path.join(
    DATASET_PATH,
    REAL_FOLDER
)


for filename in os.listdir(real_folder):

    ext = os.path.splitext(filename)[1].lower()

    if ext in IMAGE_EXTENSIONS:

        real_records.append({

            "image_path":
                os.path.join(
                    REAL_FOLDER,
                    filename
                ),

            "label": 0,

            "generator":
                REAL_FOLDER

        })



# shuffle real images

random.shuffle(
    real_records
)



# split real images

n = len(real_records)


train_end = int(
    n * 0.6
)

val_end = int(
    n * 0.8
)



for item in real_records[:train_end]:

    item["split"] = "train"

    records.append(item)



for item in real_records[train_end:val_end]:

    item["split"] = "validation"

    records.append(item)



for item in real_records[val_end:]:

    item["split"] = "test"

    records.append(item)



# ==========================
# Save CSV
# ==========================


df = pd.DataFrame(records)


# shuffle final dataset

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)



df.to_csv(
    "all_dataset.csv",
    index=False
)


df[df["split"]=="train"].to_csv(
    "train.csv",
    index=False
)


df[df["split"]=="validation"].to_csv(
    "validation.csv",
    index=False
)


df[df["split"]=="test"].to_csv(
    "test.csv",
    index=False
)



print("Dataset split completed!")

print("\nOverall:")
print(
    df.groupby(
        ["split","label"]
    ).size()
)


print("\nGenerator distribution:")
print(
    df.groupby(
        ["split","generator"]
    ).size()
)