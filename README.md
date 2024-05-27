### Requirements

```bash
pip install transformers
pip install gypytorch
```


### Run code

```bash
python run.py \
    --max_length 256 \
    --dataname TACRED \
    --encoder_epochs 30 \
    --encoder_lr 2e-5 \
    --prompt_pool_epochs 25 \
    --prompt_pool_lr 1e-4 \
    --seed 2021 \
    --bert_path bert-base-uncased \
    --data_path datasets \
    --prompt_length 8 \
    --prompt_top_k 4 \
    --batch_size 16 \
    --prompt_pool_size 20 \
    --replay_s_e_e 100 \
    --replay_epochs 200 \
    --classifier_lr 5e-5 \
    --classifier_epochs 250 \
    --prompt_type only_prompt  
```


python main.py --train_inference_task_only --seed 42 --dataname TACRED --output_dir ./output/tacred_tii_seed42


Có 5 seed: 2021, 2121, 2221, 2321, 2421
Dataname: TACRED, FewRel


# TODO:

Learnining rate
Batch size
Architecture


python run.py --max_length 256 --dataname TACRED --encoder_epochs 30 --encoder_lr 2e-5 --prompt_pool_epochs 25 --prompt_pool_lr 1e-4 --seed 2021 --bert_path bert-base-uncased --data_path datasets --prompt_length 8 --prompt_top_k 4 --batch_size 16 --prompt_pool_size 20 --replay_s_e_e 100 --replay_epochs 200 --classifier_lr 5e-5 --classifier_epochs 250 --prompt_type only_prompt  