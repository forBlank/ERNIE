
export CUDA_VISIBLE_DEVICES="6,7"


erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_ocr_sft_8k.yaml \
                model_name_or_path=/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-SFT-Model-0913 \
                train_dataset_path=/root/paddlejob/workspace/env_run/laipeiwen/dataformat_transform/OCR/ocrv5_train_ernie_choose_sft_0808.jsonl \
                train_dataset_prob=1.0 \
                max_seq_len=8192 \
                overwrite_output_dir=1