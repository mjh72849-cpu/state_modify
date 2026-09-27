# External model assets

`SE-600M` links to the existing SE checkpoint, configuration, and 5,120-D
protein embeddings. External assets stay in their original storage location;
code and configuration should reference them through this directory.

`state-env` links to the existing Conda environment. From the repository root,
use `external/state-env/bin/python` for tests and utility scripts.

Add future pretrained assets here as symbolic links, for example
`external/ST-SE-Tahoe`, rather than copying multi-gigabyte checkpoints into the
repository.
