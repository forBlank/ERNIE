

nnodes=2
nproc_per_node=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29501 \
NPROC_PER_NODE=$nproc_per_node \
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_ocr_sft_8k.yaml \
                model_name_or_path=/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-PT-Model-25026-pd \
                train_dataset_path="/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_all/add_slash_ocr_formula_fix0904_final.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_all/ppocrvl_lhe_exp2_7_60w.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_all/ocr_sft_0909.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_all/table_filtered.jsonl" \
                train_dataset_prob="0.02671954,0.27253576,0.46163083,0.23911386" \
                max_seq_len=16384 \
                overwrite_output_dir=1 \
                gradient_accumulation_steps=$(expr 128 / $nproc_per_node / $nnodes) \
                max_steps=2042 \
                warmup_steps=20 \
                save_steps=200 \
                prefetch_factor=32 \
                learning_rate=5.0e-6 \
                min_lr=5.0e-7 \
                output_dir="./ocr/all_4packing_acc8_2042step_lr"


# erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_ocr_sft_8k.yaml \
#                 model_name_or_path=/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-PT-Model-25026-pd \
#                 train_dataset_path="/root/paddlejob/workspace/env_run/laipeiwen/dataformat_transform/OCR/ocrv5_train_ernie_choose_sft_0808_5120.jsonl" \
#                 train_dataset_prob="1.0" \
#                 max_seq_len=16384 \
#                 overwrite_output_dir=1 \
#                 gradient_accumulation_steps=32 \
#                 max_steps=20 \
#                 warmup_steps=0 \
#                 warmup_ratio=0 \
#                 save_steps=100 \
#                 prefetch_factor=32 \
#                 lr_scheduler_type="constant" \
#                 lr_scheduler="constant" \
#                 learning_rate=5.0e-6 \
#                 min_lr=5.0e-7 \
#                 output_dir="./ocr/5120sample_4packing_32acc_20step_constantlr"