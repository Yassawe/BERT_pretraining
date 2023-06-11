export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
torchrun --nnodes=1 --nproc_per_node 4 pretrain.py --model_type bert --dataset_name wikipedia\
 --dataset_config_name 20220301.simple --tokenizer_name bert-base-uncased --max_seq_length 512 --num_train_epochs 40 --output_dir ./checkpoints_bsln\
  --do_train true --do_eval true --evaluation_strategy steps --logging_strategy steps --logging_first_step true > output/baseline.txt

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
torchrun --nnodes=1 --nproc_per_node 4 pretrain.py --model_type bert --dataset_name wikipedia\
 --dataset_config_name 20220301.simple --tokenizer_name bert-base-uncased --max_seq_length 512 --num_train_epochs 40 --output_dir ./checkpoints_x2\
  --do_train true --do_eval true --evaluation_strategy steps --logging_strategy steps --logging_first_step true > output/x2.txt