#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ner.py --config configs/ner/quaero-100-demi.json --num_epochs 20 --batch_size 16 \
 --loss_type sentence --optim sgd --learning_rate 0.01 --lr_decay 0.99999 --grad_clip 0.0 --warmup_steps 10 --weight_decay 0.0 --unk_replace 0.0 \
 --embedding sskip --embedding_dict "models/we_models/w2v_sg_p_100D.kv.gz" --model_path "models/ner/ner_fra4_100D_w2v_sg_conll17" \
 --train "data/QUAERO_FrenchPress/fra4_ID.train" --dev "data/QUAERO_FrenchPress/fra4_ID.dev" --test "data/QUAERO_FrenchPress/fra4_ID.test"