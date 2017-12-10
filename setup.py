from os import path
import subprocess
import requests

HERE = path.abspath(path.dirname(__file__))
TRAIN_URL = 'https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/train.7z'
PACKAGES = ['scipy', 'pandas', 'numpy']

# Download necessary packages
package_args = ' '.join(PACKAGES)
subprocess.call('sudo pip install {}'.format(package_args), shell=True)

# Download training set from Kaggle
# local_fn = path.join('t', TRAIN_URL.split('/')[-1])
# r = requests.get(TRAIN_URL, stream=True)
# r.url

