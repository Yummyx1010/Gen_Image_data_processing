import os
from PIL import Image
from collections import Counter
import pandas as pd


# 修改成你的数据路径

DATASET_PATH = r"D:\\prog\\datasets\\Unbiased Tiny GenImage"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


results = []


for generator in os.listdir(DATASET_PATH):

    folder = os.path.join(
        DATASET_PATH,
        generator
    )


    if not os.path.isdir(folder):
        continue


    resolutions = Counter()

    widths = []
    heights = []


    for filename in os.listdir(folder):

        ext = os.path.splitext(filename)[1].lower()

        if ext not in IMAGE_EXTENSIONS:
            continue


        path = os.path.join(
            folder,
            filename
        )


        try:

            with Image.open(path) as img:

                w, h = img.size

                resolutions[
                    f"{w}x{h}"
                ] += 1

                widths.append(w)
                heights.append(h)


        except:
            pass


    most_common = resolutions.most_common(3)


    results.append({

        "generator": generator,

        "num_images": len(widths),

        "unique_resolution_count":
            len(resolutions),

        "main_resolution":
            most_common[0][0],

        "main_resolution_count":
            most_common[0][1],

        "top3_resolution":
            str(most_common),

        "avg_width":
            round(sum(widths)/len(widths),2),

        "avg_height":
            round(sum(heights)/len(heights),2)

    })


df = pd.DataFrame(results)


print(df)


df.to_csv(
    "generator_resolution_analysis.csv",
    index=False
)


print(
    "Saved generator_resolution_analysis.csv"
)