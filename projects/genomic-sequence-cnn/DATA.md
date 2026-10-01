# Data

The sequence datasets used by this project are stored in the original GENOME-541 repository:

https://github.com/janisshin/GENOME-541/tree/main/hw1/data/encode-chip

The notebook expects the data at:

```
data/encode-chip/
```

Each transcription-factor dataset contains FASTA sequence files for train, validation, and test splits. Labeled development datasets also include corresponding label files.

DNA sequences are 150 bp long and are converted by `util.py` into 4 × 150 one-hot encoded tensors using channel order:

```
A, C, G, T
```

The portfolio repository intentionally does not duplicate the larger raw sequence files. The source repository should be used to recover the exact data used in the original analysis.
