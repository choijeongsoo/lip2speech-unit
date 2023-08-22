PYTHONPATH=/home/js/lip2speech-unit/fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name multi_target \
hydra.run.dir=`pwd`/exp/lrs3/multi_target \
common.user_dir=`pwd` \
task.label_dir=/home/js/lip2speech-unit/datasets/lrs3/label \
task.data=/home/js/lip2speech-unit/datasets/lrs3/label