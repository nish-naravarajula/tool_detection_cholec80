from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def load_one_video(video_id):
    path = DATA_DIR / "tool_annotations" / f"video{video_id:02d}-tool.txt"
    df = pd.read_csv(path, sep="\t")

    df["video"] = video_id
    df["second"] = df["Frame"] // 25
    df["split"] = "train" if video_id <= 40 else "test"
    return df


def load_all_labels():
    tables = [load_one_video(v) for v in range(1,81)]
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    labels = load_all_labels()
    labels = labels[["video", "second", "Frame", "split"] + TOOLS]
    labels.to_csv(DATA_DIR / "labels.csv", index=False)

    print("Total rows:", len(labels))                   
    print(labels.groupby("split")[TOOLS].sum())