#!/bin/sh

# this is for testing
default_dir='/home/data/MLDS_hw2_1_data/testing_data'
TEST_DIR=${1:-$default_dir}
echo $TEST_DIR

default_out='./output.txt'
OUTPUT_FILENAME=${2:-$default_out}

export CUDA_VISIBLE_DEVICES=""

python3 model_seq2seq.py --load_saver=1 --test_dir=$TEST_DIR \
    --test_mode=1 --output_filename=$OUTPUT_FILENAME \
    --batch_size=100 --save_dir=save_best #--with_attention=0
python3 bleu_eval.py $OUTPUT_FILENAME
