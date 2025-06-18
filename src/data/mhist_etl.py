import os
from pathlib import Path
from argparse import ArgumentParser
from collections.abc import Iterable
from PIL import Image
import pandas as pd
import numpy as np

class MHIST_ETL:
    BASE_FILENAME = "mhist_"
    REQUIRED = ["images", "annotations.csv"]
    DIAG_MAP = {"SSA": 1, "HP": 0}

    def __init__(self, 
                 indir: str | Path, 
                 outdir: str | Path, 
                 props: Iterable[int],
                 seed: int = None):
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.props = props
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self._setup()
    
    def _validate_indir_structure(self):
        if not os.path.isdir(self.indir):
            raise FileNotFoundError(f"{self.indir} is not a directory")
        present = set(os.listdir(self.indir))
        return all(req in present for req in self.REQUIRED)

    def _setup(self):
        """
        Validate the input directory as a real directory witih the necessary
        structure, and create the output directory if it does not exist.
        """
        # verify valid input directory
        if not self._validate_indir_structure():
            raise FileNotFoundError(
                "Invalid input directory structure: " + \
                    f"{self.indir} does not contain required files: " + \
                        f"{self.REQUIRED}")

        # make output directory
        try:
            os.makedirs(self.outdir, exist_ok=True)
            print(f"Created output directory: {self.outdir}")
        except Exception as e:
            raise e
    
    def _extract(self) -> dict[str, list[np.ndarray | int]]:
        """
        Using pandas, open the annotations.csv and use it to build the images
        and labels lists.
        """
        extracted = {"images": [], "labels": []}
        annotations = pd.read_csv(self.indir / "annotations.csv")

        # iterate over the CSV, loading images and labels into lists for splitting
        for _, row in annotations.iterrows():
            image_name = annotations["Image Name"][row.name]
            label = self.DIAG_MAP[annotations["Majority Vote Label"][row.name]]
            with Image.open(self.indir / "images" / image_name) as image:
                extracted["images"].append(np.array(image))
            extracted["labels"].append(label)
        return extracted

    def _prep_indices(self, n: int) -> dict[str, list[int]]:
        """
        Use lists of indices to set up the trainn/val/test splits.
        """
        indices = np.arange(n)
        self.rng.shuffle(indices)
        train_stop = int(n * self.props[0])
        val_stop = train_stop + int(n * self.props[1])
        test_stop = val_stop + int(n * self.props[2])
        return {"train": indices[:train_stop],
                "val": indices[train_stop:val_stop],
                "test": indices[val_stop:test_stop]}


    def _transform(self, 
                   extracted: dict[str, list]) -> dict[str, dict[str, np.ndarray]]:
        """
        Split the dataset.
        """
        n = len(extracted["images"])
        indices = self._prep_indices(n)
        transformed = {key: {"images": [], "labels": []} for key in indices.keys()}
        for split in indices:
            for idx in indices[split]:
                transformed[split]["images"].append(extracted["images"][idx])
                transformed[split]["labels"].append(extracted["labels"][idx])
            transformed[split]["images"] = np.array(transformed[split]["images"])
            transformed[split]["labels"] = np.array(transformed[split]["labels"])
        return transformed

    def _load(self, transformed: dict[str, dict[str, np.ndarray]]) -> None:
        for split in transformed:
            print(f"Saving {split} data...")

            # set up paths and filenames
            label_filename = self.BASE_FILENAME + split + "_labels.npy"
            label_path = self.outdir / label_filename
            image_filename = self.BASE_FILENAME + split + "_images.npy"
            image_path = self.outdir / image_filename

            # sanity checks, then save data
            img_n = transformed[split]["images"].shape[0]
            label_n = transformed[split]["labels"].shape[0]
            assert img_n == label_n, "Images and labels do not match!"
            np.save(label_path, transformed[split]["labels"])
            np.save(image_path, transformed[split]["images"])
            
            print(f"Saved {img_n} images to {image_path}")
            print(f"Saved {label_n} images to {label_path}")
        print(f"Finished saving data to {self.outdir}")

    def run(self):
        extracted = self._extract()
        transformed = self._transform(extracted)
        self._load(transformed)
        print("In MHIST_ETL: run complete")       


if __name__ == '__main__':
    # preliminaries
    parser = ArgumentParser(prog="MHIST ETL Pipeline",
                            description="Loads MHIST data into numpy array splits")
    parser.add_argument("--indir", "-i", type=str)
    parser.add_argument("--outdir", "-o", type=str)
    parser.add_argument("--props", "-p", type=float, nargs=3)
    parser.add_argument("--seed", "-s", type=int)
    args = parser.parse_args()
    print(f"Load data from: {args.indir}")
    print(f"Save data to: {args.outdir}")
    print(f"Use proportions: {args.props}")
    print(f"Use seed: {args.seed}")

    etl = MHIST_ETL(indir=args.indir,
                    outdir=args.outdir,
                    props=args.props,
                    seed=args.seed)
    etl.run()
    os.listdir(args.outdir)
    print("Out of MHIST_ETL: Completed loading and splitting of MHIST data")