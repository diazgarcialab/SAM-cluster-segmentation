# SAM Cluster Segmentation

<p align="center">
  <img src="./static/img/logo_cluster_grapes_tn.png" />
</p>

## Description

## Prerequisites

```
Python >= 3.11
```

## Installation

```bash
make install
```

## Usage

#### Edit the configuration file

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit the `.env` file to match your environment/dataset directory and names, and ROI settings, etc.
See the example images in `data/input/cluters/1_side_example` where `1_side_example` is the dataset name,
the code will search for images in all subfolders. The code
will iterate over all images with format defined by the `IMG_EXT` variable in the `.env` file.

### Execute the Segmentation over your dataset

```bash
make run
```

The RAW output including segments in RLE COCO format will be saved in `data/output/clusters/<dataset>`, the output will maintain the same folder structure as the input dataset.

### Visualize the results

We include a `jupyter` notebook to visualize the results.


## Paper Figures

The paper figures are generated using the `R` script.

```bash
make figures
```
