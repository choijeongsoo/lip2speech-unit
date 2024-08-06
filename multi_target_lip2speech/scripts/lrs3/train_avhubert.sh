PYTHONPATH=`pwd`/../fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name multi_target_avhubert \
hydra.run.dir=`pwd`/exp/lrs3/multi_target_avhubert \
common.user_dir=`pwd` \
model.w2v_path=`pwd`/../checkpoints/large_vox_iter5.pt \
task.label_dir=`pwd`/../datasets/lrs3/label \
task.data=`pwd`/../datasets/lrs3/label