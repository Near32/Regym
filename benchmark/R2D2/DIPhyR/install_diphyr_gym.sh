#/bin/sh
git clone https://github.com/Near32/DIPhyR-Gym.git
cd DIPhyR-Gym
git submodule update --init --recursive
pip install -r diphyrgym/thirdparties/pybulletgym/requirements.txt
pip install -e diphyrgym/thirdparties/pybulletgym/
python setup.py manual_develop_install

