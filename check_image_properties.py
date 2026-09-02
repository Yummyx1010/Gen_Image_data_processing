import os
from PIL import Image
from collections import Counter
import pandas as pd


# ==========================
# 修改成你的数据路径
# ==========================

DATASET_PATH = r"D:\\prog\\datasets\\Unbiased Tiny GenImage"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


records = []


# ==========================
# Scan images
# ==========================

for generator in os.listdir(DATASET_PATH):

    folder_path = os.path.join(
        DATASET_PATH,
        generator
    )


    if not os.path.isdir(folder_path):
        continue


    size_counter = Counter()
    format_counter = Counter()

    total = 0


    print(f"\nProcessing {generator}...")


    for filename in os.listdir(folder_path):

        ext = os.path.splitext(filename)[1].lower()


        if ext not in IMAGE_EXTENSIONS:
            continue


        image_path = os.path.join(
            folder_path,
            filename
        )


        try:

            with Image.open(image_path) as img:

                width, height = img.size

                img_format = img.format


                size_counter[
                    f"{width}x{height}"
                ] += 1


                format_counter[
                    img_format
                ] += 1


                total += 1


        except Exception as e:

            print(
                "Error:",
                image_path,
                e
            )


    # top 5 resolutions

    common_sizes = size_counter.most_common(5)


    records.append({

        "generator": generator,

        "total_images": total,

        "unique_resolutions":
            len(size_counter),

        "most_common_resolution":
            common_sizes[0][0]
            if common_sizes
            else None,

        "resolution_distribution":
            str(common_sizes),

        "formats":
            str(dict(format_counter))

    })


df = pd.DataFrame(records)


print("\nImage Property Summary:")
print(df)


df.to_csv(
    "image_property_statistics.csv",
    index=False
)


print(
    "\nSaved image_property_statistics.csv"
)