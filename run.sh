# CUDA_VISIBLE_DEVICES=1
python setup.py develop --no_cuda_ext
python ./basicsr/train_wflow.py -opt ./options/Options/redo/Deblurring_JRNet.yml