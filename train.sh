for seed in 2021 2121 2221
do
    python main.py \
    --seed $seed \
    --dataname TACRED \
    --output_dir ./output/tacred_tii_seed_$seed \
    --train_inference_task_only \
    --batch_size 16 \
    --classifier_epochs 200 \
    --classifier_lr 1e-3 \
    --encoder_epochs 20 \
    --encoder_lr 2e-3
done


for seed in 2021 2121 2221
do
    python main.py \
    --seed 2421 \
    --dataname TACRED \
    --output_dir ./output/tacred_seed_$seed \
    --trained_original_model ./output/tacred_tii_seed_$seed \
    --batch_size 16 \
    --classifier_epochs 100 --classifier_lr 0.05 \
    --prompt_pool_epochs 30 --prompt_pool_lr 0.03 --larger_prompt_lr \
    --prompt_length 1 \
    --prompt_top_k 8 \
    --prompt_pool_size 80 \
    --temp 0.3 --reg 0.1
done