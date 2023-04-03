CUDA_VISIBLE_DEVICES=$1 WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" $2
