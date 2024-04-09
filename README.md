# Diabetic Retinopathy Detection -- DR Detection Technology

Diabetic Retinopathy (DR) stands as a critical concern in the realm of health-
care, particularly due to its significant impact on vision loss within the global
population. This deep learning system developed by our team helps in detecting diabetic eye disease using retina images.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EVUlCrLvcOZKq9EJZKhLz7ABDfWXQJkoOsSG26b0F9xreA?e=OQkk5Q) | [Slack channel](https://sfucmpt340spring2024.slack.com/archives/C06DW4ZJX61) | [Project report](https://www.overleaf.com/project/65a57e36713c1064d79f06e3) |
|-----------|---------------|-------------------------|



## Demo Video
https://github.com/sfu-cmpt340/2024_1_project_09/assets/109203558/0228fa8c-bec3-4d1e-9664-8ea0954f3b77


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code for the entire network
├── sample_data                  ## some sample images to test the model on
├── main.py                      ## Main script to train the entire model
├── README.md                    ## You are here
├── requirements.yml             ## Use conda and download the dependencies
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate project-dependencies
python3 main.py 
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
Download dataset from https://www.kaggle.com/code/rinshinafebink/resnet18-final/input?select=gaussian_filtered_images                                                       ## dataset for DR
Download second dataset from https://ieee-dataport.s3.amazonaws.com/open/3754/A.%20Segmentation.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20Segmentation.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240409%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240409T040119Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=a95297b4008b53f350a3fa1951be8788f1420b34f02fab2249946586b29441a7         ## dataset for lesion
unzip A. Segmentation.zip
Rename A. Segmentation to data_lesion_detection
unzip gaussian_filtered_images.zip
Rename folders in gaussian_filtered_images to be :{'No_DR':0, 'Mild':1, 'Moderate':2, 'Severe':3, 'Proliferate_DR':4}
cd to where requirements.yml is
conda env create -f requirements.yml
conda activate project-dependencies
python3 main.py 
```

All models will be saved in src 
Plots will be saved in main repo


<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
