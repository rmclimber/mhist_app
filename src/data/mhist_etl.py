import sys
import os
from pathlib import Path
from argparse import ArgumentParser
from collections.abc import Iterable
import numpy as np

class MHIST_ETL:
    BASE_FILENAME = "mhist_"
    REQUIRED = ["images", "annotations.csv"]

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
        # verify valid input directory
        if not self._validate_indir_structure():
            raise FileNotFoundError(
                "Invalid input directory structure: " + \
                    f"{self.indir} does not contain required files: " + \
                        f"{self.REQUIRED}")

        # make output directory
        try:
            os.makedirs(self.outdir, exist_ok=True)
        except Exception as e:
            raise e
    
    def _extract(self):
        pass

    def _transform(self):
        pass

    def _load(self):
        pass

    def run(self):
        self._extract()
        self._transform()
        self._load()
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