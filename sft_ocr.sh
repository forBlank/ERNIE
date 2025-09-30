
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_ocr_sft_8k.yaml \
                model_name_or_path=/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-SFT-Model-0913-pd \
                train_dataset_path="/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_middle/add_slash_ocr_formula_fix0904_final.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_middle/ppocrvl_lhe_exp2_7_60w.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_middle/ocr_sft_0909.jsonl,/root/paddlejob/workspace/env_run/laipeiwen/sft0908_for_ernie_style_middle/table_filtered.jsonl" \
                train_dataset_prob="0.02671954,0.27253576,0.46163083,0.23911386" \
                max_seq_len=16384 \
                overwrite_output_dir=1 \
                gradient_accumulation_steps=8 \
                max_steps=165 \
                warmup_steps=10 \
                save_steps=10 \
                prefetch_factor=32 \
                learning_rate=5.0e-6 \
                min_lr=5.0e-7 \
                output_dir="./ocr/10%_165_20lr"