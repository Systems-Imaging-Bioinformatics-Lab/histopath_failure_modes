# Failure Mode Analysis of Deep Learning Histopathology Models

This repository documents code used in support of our final project for BIOINF 590: Image Processing and Advanced Machine Learning for Cancer Bioinformatics at The University of Michigan.

This project uses the 3-way classifier from *Coudray, N., Ocampo, P.S., Sakellaropoulos, T. et al. Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning. Nat Med 24, 1559–1567 (2018) doi:10.1038/s41591-018-0177-5*** We thank the authors for their willingness to provide their pretrained models to enable this work. 

**Authors:** Jeremy Kaplan and Nick Wang

## Setup

### Cloning
This project utilizes git submodules to reference the code already procuced for DeepPATH as part of Coudray et al.
To properly clone this repository use the following command:

    git clone --recurse-submodules git@github.com:Systems-Imaging-Bioinformatics-Lab/histopath_failure_modes.git

### Checkpoints
You'll also need the (large) checkpoints files to properly use the models. You can retrain them using the instructions in DeepPATH, or they can be provided. Either way, they'll need to be placed in a directory named checkpoints in the repository root. In each checkpoint used you'll also need to create a file named `checkpoint` to tell TensorFlow the absolute paths to the file:

    model_checkpoint_path: "<absolute_path_to_repo>/checkpoints/run2a_3D_classifier/model.ckpt-400000"
    all_model_checkpoint_paths: "<absolute_path_to_repo>/checkpoints/run2a_3D_classifier/model.ckpt-400000"

### Conda 
You'll want to setup a conda environment so that all the proper dependencies are all set. It should (hopefully) be as simple as

    conda env create -f environment.yml
You should then have a conda env named b590prj all set to go 

### Work Directory
Lastly, set up a work directory where the snakemake workflow will find input files by creating your working directory structure
    
    mkdir work/input

## Predict Slide Classes

### Downloading Slide Images
The pipeline starts at downloading slide images from the GDC. We conducted this analysis using images from the TCGA-LUAD and TCGA-LUSC studies. A manifest of all LUSC and LUAD slides is available under `work/input/all`.

If you want to use a different set of images, you'll leed to use the [GDC legacy portal](https://portal.gdc.cancer.gov/legacy-archive/search/f) to enable the metadata files to be compatible. 

1. Add the SVS images you want to your cart(and ONLY svs images)
2. Download your cart and the acompanying METADATA file(should be metadata.date.json). The metadata is different than the manifest file, which we won't need.
3. Unzip the cart download into `work/input`. Rename the directory to a useful phrase to refer to this dataset
4. Move the `metadata.*.json*` file into the your new `work/input/<dataset>` directory. This ensures the metadata is associated with the cart properly

## To run the pipeline
In the repo root directory run:

    conda activate b590prj
    snakemake -j 

This will run for awhile depending on your input dataset size, and when done will create a directory `work/output/<dataset>` with per tile scores for each class.

## Handling GPU limited steps
Some steps in the pipeline(`predict` and `make_tf_record`) can use a GPU if provided and the tensorflow GPU libraries are installed. However, by default snakemake will not consider the number of GPUs in a machine rate-limiting, and will run all GPU-requiring steps at the same time if all pre-requisites are met. This may yield suboptimal performance, or an error due to not enough GPU memory. To ensure snakemake only runs GPU steps when there are enough GPUs, you can run snakemake as:

    snakemake -j --resources gpu=$n

where `$n` is the number of GPUs available on your system.