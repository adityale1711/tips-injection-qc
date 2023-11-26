# Tips Injection Detector for Quality Control

## Dependencies
- opencv-python==4.8.1.78
- imutils==0.5.4
- numpy==1.26.0
- tensorflow==2.10.1
- tensorflow-gpu==2.10.1 (if using GPU)
- matplotlib==3.8.2
- scikit-learn==1.3.2
- pandas==2.1.3

Also install cudatoolkit=11.2 and cudnn=8.1.0 if using Nvidia GPU

if you're using conda environment, just run this command:
> conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

## How to run detector
1. Clone the repository
> git clone https://github.com/adityale1711/tips-injection-qc.git
2. Change directory into the repository folder
> cd tips-injection-qc
3. Install requires dependencies
> pip install requirements.txt
4. Run predict.py
> python predict.py