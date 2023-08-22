CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/js/lip2speech-unit/fairseq \
python -B inference.py \
--config-dir conf \
--config-name decode \
common.user_dir=`pwd` \
common_eval.path=`pwd`/checkpoints/lip2speech_lrs3_avhubert_multi.pt \
common_eval.results_path=`pwd`/results