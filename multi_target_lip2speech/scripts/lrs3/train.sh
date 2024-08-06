PYTHONPATH=`pwd`/../fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name multi_target \
hydra.run.dir=`pwd`/exp/lrs3/multi_target \
common.user_dir=`pwd` \
task.label_dir=`pwd`/../datasets/lrs3/label \
task.data=`pwd`/../datasets/lrs3/label