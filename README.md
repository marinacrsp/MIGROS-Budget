# MIGROS Budget: Medical Image Generation, Retraining on a Strict Budget

This is the repository for the project for the DL course.\
Authors: Luca Anceschi, Marina Crespo Aguirre, Luca Drole 


<img src="https://github.com/user-attachments/assets/4a0fa0a8-1b4f-4a7d-8210-908252b2d439" alt="Image generation example" width="600"/> 

## Table of Contents 

- [Introduction](##introduction)
- [Datasets](##Datasets)
- [Environment](##Environment)
- [Training](##Training)
- [Computing the FID](#computing-the-fid)
- [Example](#example)

## Introduction
Low-rank adaptation presents a promising avenue for fine-tuning diffusion models at a relatively low computational cost. We are especially interested in assessing whether Self-Expanding Low-Rank adaptation can be used to produce realistic MRI data.


## Datasets
For this project, two publicly available datasets were used.

1. BraTS2021: Brain Tumor Segmentation 2021 Challenge dataset 
Dataset available at [link](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

2. PI-CAI: Prostate Imaging: Cancer AI challenge 
Dataset available at [link](https://zenodo.org/records/6624726)
Labels are available at the following [GitHub](https://github.com/DIAGNijmegen/picai_labels/tree/main), for this project only the expert-reviewed annotations were used (found under `/csPCa_lesion_delineations/human_expert/resampled`).

Minor pre-processing has been done with two notebooks available under [Processing](https://github.com/LucaAnce/MIGROS-Budget/tree/main/Processing) ([preprocess_data_brats](https://github.com/LucaAnce/MIGROS-Budget/tree/main/Processing/preprocess_data_brats.ipynb) and [preprocess_data_picai.ipynb](https://github.com/LucaAnce/MIGROS-Budget/tree/main/Processing/preprocess_data_picai.ipynb)). The datasets were processed to extract 2D slices and a `metadata.csv` document, which maps the relevant masks with a textual prompt. 
\
The *pre-processed datasets* are made available for convenience in the folders [Datasets/BraTS](https://github.com/LucaAnce/MIGROS-Budget/tree/main/Datasets/BraTS) and [Datasets/PICAI](https://github.com/LucaAnce/MIGROS-Budget/tree/main/Datasets/BraTS). Each folder then contains the training and testing set used, each with its `metadata.csv` file.



## Training
You can use the `experiment.sh` file to run our scripts on the INFK cluster. Make sure to change the relevant paths.

### Environment 
For setting up the environment, use the `environment.yml` file. 
Install environment:
```
conda env create --file environment.yml
```
### Configuration 
To use SeLoRA script, the following directories should be specified in `config.yaml` file:
- **`dataset_path`**: The directory where the training images are stored.
- **`report_name`**: The name of the `.csv` file containing the image names associated with the prompts. It should be in the folder defined by `dataset_path`.
- **`prompts_for_generation`**: The `.csv` file containing the prompts used for inference. Please structure it like the provided example if you want to use your own prompts.
- **`output/path`**: The directory where the output results should be saved.

You can modify the training settings within the `config.yaml` file:
- **`seed`**: The random seed for reproducibility.
- **`batch_size`**: Number of samples per training batch.
- **`learning_rate`**: The learning rate for the optimizer.
- **`epochs`**: The total number of training epochs.
- **`expansion_threshold`**: Threshold for SeLoRA's dynamic rank expansion (lambda).
- **`initial_rank`**: The initial rank of the LoRA matrices. It should be set to 1 for SeLoRA.


### Training SeLoRA
Mao et al. 2024 made the original implementation for the SeLoRA training available at this [repository](https://anonymous.4open.science/r/SeLoRA-980D). However, this was only used as a base, and many changes were made to adapt it to the project's needs. Please note that the code published by the authors seems not to run.
To run SeLoRA, run:
```bash
python selora_finetuning.py
```

### Training LoRA
To train a LoRA model, it is sufficient to change the initial rank to the desired rank and set a very high threshold. Consider that normally this value ranges from `1.0` to `1.3` for SeLoRA.

### Training StyleGAN
We use a StyleGAN3 as our baseline. We refer to the [original repository](https://github.com/NVlabs/stylegan3) for further details.
1. To train the StyleGAN, first clone the repository:
    ```bash
    git clone https://github.com/NVlabs/stylegan3.git
    ```
2. Then, you will have to pre-process the data
    ```bash
    python dataset_tool.py --source= /path/to/source --dest= path/to/postprocessed_dataset --resolution=256x256
    ```
3. Now, you can train the StyleGAN. You can regulate the number of iterations with the `kimg` paramter
    ```bash
    python train.py --outdir= /path/to/output_directory --cfg=stylegan3-t --data=path/to/postprocessed_dataset \
        --gpus=1 --batch=3 --gamma=2 --mirror=1 --kimg=25 --snap=6 \
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl \
        --tick=1 --mbstd-group 1 --metrics none --cbase=16384
    ```
4. Finally, you can perform inference like this:
    ```bash
    python gen_images.py --outdir=/path/to/output_directory --seeds=0-10  --network= /path/to/model
    ```
    Here you can specify any range of seeds to generate the corresponding number of images.

- NOTE 1: Notice how the model does not have an explicit `epochs` parameter. Rather, we use the `kimg` paramter to ensure that the number of update steps is roughly the same as the corresponding Low-rank adaptation setup. We also train for a fixed 25000 steps as the results were observed to be more visually convincing.
- NOTE 2: Notice how in this case, one needs to train 2 different models, one for cancer-negative images and one for cancer-positive images.

## Evaluation
We obtained our results using the evaluation protocol presented by De Wilde et al. (2023). 

### Computing the FID
The Frechet Inception Distance is a metric used to assess the quality of generated images. To compute the FID you will need a test dataset containing real images and a synthetic dataset. Then, you can run:

```bash
python compute_fid.py --orig /path/to/original --syn /path/to/synthetic
```

### BraTS postprocessing
After generating the images and assessing the FID, some simple post-processing (background removal) was done. This is used to correct for issues when the background was not homogenous and black (note this was only used for the generated brain images). This can be replicated with the [postprocess_generated_brats](/Processing/postprocess_generated_brats.ipynb) notebook.

### Computing Accuracy and AUC
To compute the AUC and Accuracy scores, run the `classifier_train.py` script. Specify the following arguments
```bash
python classifier_train.py  --test=/path/to/testset --train=/path/to/trainset
--syn=/path/to/generation/full_output --merge=True
--merge_path=/path/to/merged/folder/generation/
```
Determine the test set, train set, and set of synthetic images, as well as mode of AUC computation `--merge`, `False` for estimating the AUC score on the real images only, `True` for data augmentation with the synthetic outputs. Specify `merge_path`, or directory where the augmented training set of images should be created.   

## References
- Mao, Yuchen, et al. "SeLoRA: Self-Expanding Low-Rank Adaptation of Latent Diffusion Model for Medical Image Synthesis." arXiv preprint arXiv:2408.07196 (2024). [LINK](https://arxiv.org/abs/2408.07196)
- Karras, Tero, et al. "Alias-free generative adversarial networks." Advances in neural information processing systems 34 (2021): 852-863. [LINK](https://arxiv.org/abs/2106.12423)
- De Wilde, Bram, et al. "Medical diffusion on a budget: textual inversion for medical image generation." arXiv preprint arXiv:2303.13430 (2023). [LINK](https://arxiv.org/abs/2303.13430)


