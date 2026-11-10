# Data

The public datasets are downloaded, not shipped:

* CIFAR-10 / CIFAR-100: `python -c "import torchvision; torchvision.datasets.CIFAR10(root='data', download=True)"`
  (the paper reads `data/cifar-10-batches-py/`; ~180 MB)
* TPC-H (only for the `code/e16_tpch.py` replication): DuckDB generates it locally with
  `INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.5);` -- no raw data download is needed.

Every table in the paper is regenerated from `results/*.csv`, and the CSVs are shipped, so a
reviewer can check the manuscript numbers without re-running a training job.
