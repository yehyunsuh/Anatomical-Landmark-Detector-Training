# Landmark-Detection

## ⚙️ Environment Setting
Set up the Conda environment and install dependencies:

```bash
git clone https://github.com/yehyunsuh/Anatomical-Landmark-Detector.git
cd Anatomical-Landmark-Detector
conda create -n detector python=3.10 -y
conda activate detector
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```