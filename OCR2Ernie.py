import os
import os.path as osp

import paddle

from ernie.tokenizer import Ernie4_5_Tokenizer

from ernie.configuration_ocr import PPOCRVLConfig
from ernie.modeling_ocr import PPOCRVLForConditionalGeneration


root_dir = "/root/paddlejob/workspace/env_run/laipeiwen"

keye_vl_path = ""
ernie_llm_path = osp.join(root_dir, "ERNIE-4.5-0.3B-Paddle")
final_config_path = ""
# output_dir = osp.join(root_dir, "PaddleOCR-Pretrain-Model-25026")
output_dir = osp.join(root_dir, "PaddleOCR-SFT-Model-0913")
os.makedirs(output_dir, exist_ok=True)

# 加载预训练的 Tokenizer
tokenizer = Ernie4_5_Tokenizer.from_pretrained(ernie_llm_path)

print(tokenizer.special_tokens_map['additional_special_tokens'])
print(f"len(tokenizer): {len(tokenizer)}")
print(f"tokenizer vocab size: {len(tokenizer.get_vocab())}")

vision_control_tokens = [
    "<|image_pad|>",
    "<|IMAGE_START|>",
    "<|IMAGE_END|>",
    "<|video_pad|>",
]
# 新增的结构化Token
structural_tokens = [
    '<ecel>', '<fcel>', '<xcel>', '<lcel>', '<ucel>', '<nl>'
]

# 将它们合并到一个列表中
new_special_tokens_list = vision_control_tokens + structural_tokens

special_tokens_dict = {'additional_special_tokens': new_special_tokens_list}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

print(tokenizer.special_tokens_map['additional_special_tokens'])
print(f"len(tokenizer): {len(tokenizer)}")
print(f"tokenizer vocab size: {len(tokenizer.get_vocab())}")

tokenizer.save_pretrained(output_dir)