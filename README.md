# Anatomical Landmark Detector (Training)

A deep learning pipeline for anatomical landmark detection using U-Net. This project enables training and visual evaluation of landmark predictions on medical images (e.g., AP hip X-rays).

For collecting anatomical landmark dataset, please visit https://github.com/yehyunsuh/Anatomical-Landmark-Annotator 

For testing, please visit https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Testing   

---

## 📂 Directory Structure
```
Anatomical-Landmark-Detector-Training/
│
├── data/
│   ├── train_images/             # Directory containing input images
│   └── labels/
│       └── train_annotation.csv  # CSV with landmark coordinates
│
├── graph/                       # Output directory for training curves (auto-created)
├── visualization/               # Output directory for image overlays (auto-created)
├── weight/                      # Directory for saving model weights (auto-created)
│
├── main.py                      # Entry point to start training
├── model.py                     # U-Net model definition using SMP
├── data_loader.py               # Dataset and dataloader implementation
├── train.py                     # Training and evaluation loop
├── utils.py                     # Utility functions (e.g., seeding)
├── visualization.py             # Visualizations of results
├── requirements.txt             # Required Python packages
└── README.md                    # You’re here!
```

---

## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

```bash
git clone https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Training.git
cd Anatomical-Landmark-Annotator-Training
conda create -n annotator python=3.10 -y
conda activate annotator
pip3 install -r requirements.txt
```

### 2. Prepare Your Data

Place your training images under:
```
data/train_images/
```

Your annotation CSV should look like:
```
filename,width,height,channels,x1,y1,x2,y2,...,xn,yn
image1.jpg,1098,1120,3,123,145,245,310,...
image2.jpg,1400,1210,3,108,132,298,294,...
```

Make sure the number of (x, y) coordinate pairs matches `--n_landmarks`.

### 3. Run Training
```bash
python main.py --n_landmarks 5
```

### 🧩 Argument Reference

| Argument            | Description                                                   | Default                     |
|---------------------|---------------------------------------------------------------|-----------------------------|
| `--train_image_dir` | Directory with training images                                 | `./data/train_images`       |
| `--label_dir`       | Directory containing CSV annotation file                      | `./data/labels`             |
| `--train_csv_file`  | Filename of the CSV with image names and landmark coords       | `train_annotation.csv`      |
| `--image_resize`    | Resize images to this square size (must be divisible by 32)   | `512`                       |
| `--n_landmarks`     | Number of landmark points per image `(required)`                          | `2`                |
| `--batch_size`      | Batch size used for training                                  | `8`                         |
| `--epochs`          | Number of training epochs                                     | `350`                       |
| `--dilation_iters`  | Number of dilation iterations for expanding landmark masks    | `65`                        |
| `--erosion_freq`    | Apply erosion every N epochs to increase training difficulty  | `50`                        |
| `--erosion_iters`   | Number of erosion iterations applied during erosion phase     | `10`                        |
| `--seed`            | Random seed for reproducibility                               | `42`                        |

This will:   
- Train the model
- Save the best model to weight/
- Save overlays in visualization/
- Save loss/accuracy plots in graph/

## 📊 Visualization
- Green circles: Ground truth landmarks
- Red circles: Predicted landmarks
- Visual outputs are saved every 10 epochs under `visualization/Epoch{epoch}/`.

## 📈 Training Plots
Plots saved in the `graph/` folder:
- loss_curve.png: Training vs validation loss
- mean_landmark_error.png: Mean pixel distance
- mean_dice_score.png: Overall Dice score
- per_landmark_error.png: Error per landmark
- per_landmark_dice.png: Dice score per landmark

## Citation
If you find this helpful, please cite this [paper](https://openreview.net/forum?id=bVC9bi_-t7Y):
```
@inproceedings{
suh2023dilationerosion,
title={Dilation-Erosion Methods for Radiograph Annotation in Total Knee Replacement},
author={Yehyun Suh and Aleksander Mika and J. Ryan Martin and Daniel Moyer},
booktitle={Medical Imaging with Deep Learning, short paper track},
year={2023},
url={https://openreview.net/forum?id=bVC9bi_-t7Y}
}
```
Also, if you conduct research in Total Knee Arthroplasty or Total Hip Arthroplasty, check these papers:
- Mika, A. P., Suh, Y., Elrod, R. W., Faschingbauer, M., Moyer, D. C., & Martin, J. R. (2024). Novel Dilation-erosion Labeling Technique Allows for Rapid, Accurate and Adjustable Alignment Measurements in Primary TKA. Computers in Biology and Medicine, 185, 109571. https://doi.org/10.1016/j.compbiomed.2024.109571.  
- Mohsin, M., Suh, Y., Chandrashekar, A., Martin, J., & Moyer, D. (2025). Landmark prediction in large radiographs using RoI-based label augmentation. Proc. SPIE 13410, Medical Imaging 2025: Clinical and Biomedical Imaging, 134100E, 14. https://doi.org/10.1117/12.3047290
- Chan, P. Y., Baker, C. E., Suh, Y., Moyer, D., & Martin, J. R. (2025). Development of a deep learning model for automating implant position in total hip arthroplasty. The Journal of Arthroplasty. https://doi.org/10.1016/j.arth.2025.01.032