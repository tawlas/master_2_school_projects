#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ner.py --config configs/ner/conll03.json --num_epochs 20 --batch_size 16 \
 --loss_type sentence --optim sgd --learning_rate 0.01 --lr_decay 0.99999 --grad_clip 0.0 --warmup_steps 100 --weight_decay 0.0 --unk_replace 0.0 \
 --embedding word2vec --embedding_dict "models/we_models/w2v_cbow_100D" --model_path "models/ner/conll03" \
 --train "data/TP_ISD2020/QUAERO_FrenchMed/EMEA/EMEAtrain_layer1_ID.conll" --dev "data/TP_ISD2020/QUAERO_FrenchMed/EMEA/EMEAdev_layer1_ID.conll" --test "data/TP_ISD2020/QUAERO_FrenchMed/EMEA/EMEAtest_layer1_ID.conll"
