import os
import os.path as osp

import paddle
paddle.set_printoptions(linewidth=1000)
# import torch
import numpy as np


from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.configuration_ocr import PPOCRVLConfig
from ernie.modeling_ocr import PPOCRVLForConditionalGeneration


root_dir = "/root/paddlejob/workspace/env_run/laipeiwen"

keye_vl_path = ""
ernie_llm_path = osp.join(root_dir, "ERNIE-4.5-0.3B-Paddle")
final_config_path = ""
output_dir = osp.join(root_dir, "PaddleOCR-PT-Model-25026-pd")
# output_dir = osp.join(root_dir, "PaddleOCR-SFT-Model-0913-pd")
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

print(tokenizer.special_tokens_map)

tokenizer.image_token = "<|IMAGE_PLACEHOLDER|>"

tokenizer.save_pretrained(output_dir)


# data_path = "/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE/alignment"

# def load_np(sub_dir, filename):
#     return np.load(osp.join(data_path, sub_dir, filename))

# def crossentropy(logits, labels, backend):
#     logits = logits[labels > 0, :]
#     labels = labels[labels > 0]

#     if backend == "paddle":
#         logits_gt = paddle.gather(logits, labels.unsqueeze(-1), axis=-1)
#         logits_gt = logits_gt.squeeze(0)

#         loss = -paddle.log( paddle.exp(logits_gt) / paddle.sum(paddle.exp(logits), axis=-1) )
#         loss = loss / len(labels)
#     elif backend == "torch":
#         # 1. 计算softmax
#         max_logits = logits.max(dim=1, keepdim=True)[0]  # 防止数值溢出
#         exp_logits = torch.exp(logits - max_logits)
#         probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
#         # 2. 计算log概率
#         log_probs = torch.log(probs + 1e-10)  # 添加小量防止log(0)
        
#         # 3. 按标签索引取值
#         batch_size = logits.size(0)
#         selected_log_probs = log_probs[torch.arange(batch_size), labels]
        
#         # 4. 求平均负对数似然
#         loss = -selected_log_probs.mean()
#     return loss

# def compute_loss(logits, labels, backend):
    
#     if backend == "paddle":
#         logits = paddle.to_tensor(logits)
#         labels = paddle.to_tensor(labels)
#     elif backend == "torch":
#         logits = torch.tensor(logits).cuda()
#         labels = torch.tensor(labels).cuda()
    
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
    
#     if backend == "paddle":
#         shift_logits = shift_logits.reshape((-1, 103424))
#         shift_labels = shift_labels.reshape((-1,))
#     elif backend == "torch":
#         shift_logits = shift_logits.view(-1, 103424)
#         shift_labels = shift_labels.view(-1)

#     if backend == "paddle":
#         loss_fct = paddle.nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits, shift_labels)
        
#     elif backend == "torch":
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits, shift_labels)

#     return loss, loss

# swift_logits = load_np("swift", "PT_4samples_output_logits.npy")
# ernie_logits = load_np("ernie", "PT_4samples_output_logits.npy")

# print("swift_logits")
# print(swift_logits)
# print("ernie_logits")
# print(ernie_logits)

# swift_labels = load_np("swift", "PT_4samples_output_labels.npy")
# ernie_labels = load_np("ernie", "PT_4samples_output_labels.npy")

# swift_loss_paddle, swift_loss_paddle_by_hand = compute_loss(swift_logits, swift_labels, backend="paddle")
# ernie_loss_paddle, ernie_loss_paddle_by_hand = compute_loss(ernie_logits, ernie_labels, backend="paddle")

# print(f"swift_loss_paddle: {swift_loss_paddle.item()}")
# # print(f"swift_loss_paddle_by_hand: {swift_loss_paddle_by_hand.item()}")
# print(f"ernie_loss_paddle: {ernie_loss_paddle.item()}")
# # print(f"ernie_loss_paddle_by_hand: {ernie_loss_paddle_by_hand.item()}")

# swift_loss_torch, swift_loss_torch_by_hand = compute_loss(swift_logits, swift_labels, backend="torch")
# ernie_loss_torch, ernie_loss_torch_by_hand = compute_loss(ernie_logits, ernie_labels, backend="torch")

# print(f"swift_loss_torch: {swift_loss_torch.item()}")
# # print(f"swift_loss_torch_by_hand: {swift_loss_torch_by_hand.item()}")
# print(f"ernie_loss_torch: {ernie_loss_torch.item()}")
# # print(f"ernie_loss_torch_by_hand: {ernie_loss_torch_by_hand.item()}")

# swift_ids = load_np("swift", "PT_4samples_output_ids.npy")
# ernie_ids = load_np("ernie", "PT_4samples_output_ids.npy")

# swift_ids = paddle.to_tensor(swift_ids)
# ernie_ids = paddle.to_tensor(ernie_ids)

# print(swift_ids.tolist())
# print(tokenizer.decode(swift_ids.tolist()))
# print(ernie_ids.tolist())
# print(tokenizer.decode(ernie_ids.tolist()))