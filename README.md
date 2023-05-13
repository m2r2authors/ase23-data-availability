# M2R2: A Simple Multimodal Neural Retrieve-and-Rerank Approach for Query-based Direct API Recommendation

This is the official implementation of M2R2.

## Environment

1. torch==1.13.1
2. Other requirements can be found in `setup.ipynb`

## Data & Checkpoints

1. The data and checkpoints can be found in https://drive.google.com/drive/folders/1D2cpo1jKbanGiba4tWzqeQaq1H2jXSEC?usp=share_link.
2. Extract the `_data.zip` files into the `{language}-expr/data/` folder and `_checkpoints.zip` into the `{language}-expr/save/` folder.
3. NOTE: To reproduce the same checkpoints, you need to reuse the preprocessed training data.

## Pipeline

1. Check `pipeline-{test-name}.ipynb` in `java-expr/` for the JDK experiment pipeline.
2. Check `complete_pipeline.ipynb` in `python-expr/` for the Python ML library experiment pipeline.