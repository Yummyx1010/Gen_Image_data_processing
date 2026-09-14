import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import GenImageDataset, transform
from models.baseline_a import BaselineA


# =========================================================
# 1. Configuration
# =========================================================

MY_DATA_ROOT = r"D:\AA-SemesterTwo\Course\COMPSCI 760\Week 7\Project update\Dataset"

TEST_CSV = "test.csv"

CHECKPOINT = "baseline_a_debug.pth"

TEST_GENERATORS = [
    "glide",
    "Midjourney",
    "wukong"
]

BATCH_SIZE = 32


# =========================================================
# 2. Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# 3. Load full test dataset and CSV
# =========================================================

test_dataset = GenImageDataset(
    csv_file=TEST_CSV,
    root_dir=MY_DATA_ROOT,
    transform=transform
)

test_df = pd.read_csv(TEST_CSV)


# =========================================================
# 4. Inspect test.csv structure
# =========================================================

print("\n===================================")
print("Test CSV Structure")
print("===================================")

print("\nGenerator distribution:")
print(test_df["generator"].value_counts())

print("\nLabel distribution:")
print(test_df["label"].value_counts())

print("\nGenerator x Label:")
print(
    pd.crosstab(
        test_df["generator"],
        test_df["label"]
    )
)


# =========================================================
# 5. Load trained Baseline A
# =========================================================

model = BaselineA().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()


# =========================================================
# 6. Evaluate one unseen generator
# =========================================================

def evaluate_generator(generator_name):

    # -----------------------------------------------------
    # Select:
    # 1. ALL real images in test set: label == 0
    # 2. Fake images from target generator only:
    #    label == 1 AND generator == generator_name
    # -----------------------------------------------------

    selected_indices = test_df[
        (test_df["label"] == 0)
        |
        (
            (test_df["label"] == 1)
            &
            (test_df["generator"] == generator_name)
        )
    ].index.tolist()


    subset = Subset(
        test_dataset,
        selected_indices
    )


    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    all_labels = []
    all_probs = []


    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)

            labels = batch["label"].to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probabilities.cpu().numpy()
            )


    # -----------------------------------------------------
    # Check label distribution
    # -----------------------------------------------------

    unique_labels, counts = np.unique(
        all_labels,
        return_counts=True
    )

    label_distribution = dict(
        zip(
            unique_labels.astype(int),
            counts
        )
    )


    print(
        f"\n{generator_name} label distribution:",
        label_distribution
    )


    # -----------------------------------------------------
    # Safety check:
    # AUROC requires both class 0 and class 1
    # -----------------------------------------------------

    if len(unique_labels) < 2:

        print(
            f"Warning: {generator_name} contains only one class."
        )

        return {
            "generator": generator_name,
            "num_images": len(all_labels),
            "accuracy": np.nan,
            "auroc": np.nan
        }


    # -----------------------------------------------------
    # Convert probabilities to class predictions
    # -----------------------------------------------------

    predictions = [
        1 if p >= 0.5 else 0
        for p in all_probs
    ]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    auroc = roc_auc_score(
        all_labels,
        all_probs
    )


    return {
        "generator": generator_name,
        "num_images": len(all_labels),
        "accuracy": accuracy,
        "auroc": auroc
    }


# =========================================================
# 7. Evaluate each unseen generator
# =========================================================

results = []


for generator in TEST_GENERATORS:

    result = evaluate_generator(
        generator
    )

    results.append(
        result
    )


    print(
        f"\nGenerator: {generator}"
    )

    print(
        f"Images: {result['num_images']}"
    )

    print(
        f"Accuracy: {result['accuracy']:.4f}"
    )

    print(
        f"AUROC: {result['auroc']:.4f}"
    )


# =========================================================
# 8. Cross-generator statistics
# =========================================================

aurocs = [
    result["auroc"]
    for result in results
    if not np.isnan(result["auroc"])
]


if len(aurocs) > 0:

    mean_auroc = np.mean(
        aurocs
    )

    cross_generator_sd = np.std(
        aurocs,
        ddof=0
    )

    worst_generator_auroc = np.min(
        aurocs
    )


    print("\n===================================")
    print("Baseline A Cross-Generator Results")
    print("===================================")

    print(
        f"Mean AUROC: {mean_auroc:.4f}"
    )

    print(
        f"Cross-generator SD: {cross_generator_sd:.4f}"
    )

    print(
        f"Worst-generator AUROC: {worst_generator_auroc:.4f}"
    )

else:

    print(
        "\nNo valid AUROC values were produced."
    )


# =========================================================
# 9. Save per-generator results
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "baseline_a_test_results.csv",
    index=False
)

print(
    "\nResults saved to baseline_a_test_results.csv"
)