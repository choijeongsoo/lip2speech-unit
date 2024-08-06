CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=`pwd`/../fairseq \
python -B inference.py \
--config-dir conf \
--config-name decode \
hydra.run.dir=`pwd` \
common.user_dir=`pwd` \
common_eval.path=`pwd`/../checkpoints/lip2speech_lrs3_avhubert_multi.pt \
common_eval.results_path=`pwd`/../results/lrs3 \
override.w2v_path=`pwd`/../checkpoints/large_vox_iter5.pt \
override.label_dir=`pwd`/../datasets/lrs3/label \
override.data=`pwd`/../datasets/lrs3/label