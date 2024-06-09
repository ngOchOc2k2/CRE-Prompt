cd CRL-Shaw-48

python run.py \
    --max_length 256 \
    --dataname TACRED \
    --encoder_epochs 30 \
    --encoder_lr 2e-5 \
    --prompt_pool_epochs 25 \
    --prompt_pool_lr 1e-4 \
    --classifier_epochs 200 \
    --seed 2421 \
    --bert_path datasets/bert-base-uncased \
    --data_path datasets \
    --prompt_length 80 \
    --prompt_top_k 1 \
    --batch_size 16 \
    --prompt_pool_size 1 \
    --replay_s_e_e 200 \
    --replay_epochs 50 \
    --classifier_lr 3e-5 \
    --prompt_type only_prompt \
    --pull_constraint_coeff 0.0

