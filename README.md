# invoice-recognition


## Setup
pip install torch==1.4.0 torchvision==0.5.0
pip install -q mmcv terminaltables
git clone --branch new "https://github.com/KoryakovDmitry/mmdetection.git"
cd mmdetection
pip install -r requirements/optional.txt
python setup.py install
python setup.py develop
pip install -r requirements.txt
pip install pillow==6.2.1 
pip install mmcv==0.5.1
pip install pycocotools
pip install pytesseract

(tesseract with another packages) 

## Download weights

https://drive.google.com/uc?id=1OFUdP7XPT4MQV5pZ-SZxd_-vy7mB_FQf

