#git clone https://github.com/mila-iqia/babyai
git clone https://github.com/Near32/babyai
# Installation from a fork of the original repo.
# The fork patches a bug in the BabyAI Bot...
cd babyai
git checkout patch-1
pip install blosc==1.5.1
pip install -e .
export BABYAI_STORAGE=`pwd`
