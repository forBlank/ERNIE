import os
import os.path as osp
import paddle
from safetensors.paddle import load_file
import numpy as np


def key_torch_to_paddle(key):
    mappings = [
        ("running_mean", "_mean"),
        ("running_var", "_variance"),
    ]
    for pattern, replacement in mappings:
        if pattern in key:
            key = key.replace(pattern, replacement)
    return key


def split_attention_weights(weight=None, bias=None):
    # weight / bias 都是 numpy 数组
    if weight is not None:
        split_size = weight.shape[0] // 3
        q_weight = weight[:split_size]
        k_weight = weight[split_size: 2 * split_size]
        v_weight = weight[2 * split_size:]
        return q_weight.T, k_weight.T, v_weight.T  # 注意这里已经转置
    elif bias is not None:
        split_size = bias.shape[0] // 3
        q_bias = bias[:split_size]
        k_bias = bias[split_size: 2 * split_size]
        v_bias = bias[2 * split_size:]
        return q_bias, k_bias, v_bias


# def to_numpy_and_target_dtype(t: torch.Tensor):
#     """
#     返回 (np_array, target_paddle_dtype)
#     - bfloat16: 临时转 fp32 以便 numpy 处理，最终写回 paddle.bfloat16
#     - float16: 直接 numpy，最终写回 paddle.float16
#     - 其他: 直接 numpy，保持默认
#     """
#     if t.dtype == torch.bfloat16:
#         return t.cpu().to(torch.float32).numpy(), paddle.bfloat16
#     elif t.dtype == torch.float16:
#         return t.cpu().numpy(), paddle.float16
#     else:
#         return t.cpu().numpy(), None


def to_paddle_tensor(np_arr: np.ndarray, target_dtype):
    if target_dtype is None:
        return paddle.to_tensor(np_arr)
    else:
        # 这里一次性用目标 dtype 构建，避免额外拷贝
        return paddle.to_tensor(np_arr, dtype=target_dtype)


def convert_weights(torch_weight_path, paddle_weight_path):
    torch_weights = load_file(torch_weight_path, device="cpu")
    paddle_weights = {}

    for key in torch_weights.keys():
        print(key)

        np_weight, target_dtype = to_numpy_and_target_dtype(torch_weights[key])

        # Special handling for BatchNorm2d layers
        if "channelwise" in key and ("gamma" in key or "beta" in key):
            np_weight = np_weight.reshape((-1, 1, 1, 1))

        # 需要转置的层名片段
        t_layers = [
            "fc",
            "channelwise",
            "mapper_crp",
            "mapper_sca",
            ".mapper.",
            "txt_mapper",
            "txt_pooled_mapper",
            "clip_img_mapper",
            "kv_mapper",
            "clip_mapper",
            "out_proj",
            # "patch_embedding",
            "q_proj",
            "k_proj",
            "v_proj",
            "lm_head",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "lm_head",
            "linear_1",
            "linear_2",
        ]
        if any(t_layer in key and "bias" not in key for t_layer in t_layers):
            np_weight = np_weight.transpose()

        paddle_key = key_torch_to_paddle(key)

        # siglip的特殊处理
        # 处理 in_proj 拆分（注意不要在后面再覆盖）
        if "attention.in_proj" in key or "attention.attn.in_proj" in key:
            if "weight" in key:
                q_w, k_w, v_w = split_attention_weights(weight=np_weight)
                paddle_weights[paddle_key.replace("in_proj_weight", "q_proj.weight")] = to_paddle_tensor(q_w, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_weight", "k_proj.weight")] = to_paddle_tensor(k_w, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_weight", "v_proj.weight")] = to_paddle_tensor(v_w, target_dtype)
            elif "bias" in key:
                q_b, k_b, v_b = split_attention_weights(bias=np_weight)
                paddle_weights[paddle_key.replace("in_proj_bias", "q_proj.bias")] = to_paddle_tensor(q_b, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_bias", "k_proj.bias")] = to_paddle_tensor(k_b, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_bias", "v_proj.bias")] = to_paddle_tensor(v_b, target_dtype)
            print("#(split) ", key, "->", paddle_key)
            continue  # ⬅️ 防止后续又把整块覆盖

        # ernie的特殊处理
        if "mlp.up_proj." in key:
            if "weight" in key:
                q_w, k_w, v_w = split_attention_weights(weight=np_weight)
                paddle_weights[paddle_key.replace("in_proj_weight", "q_proj.weight")] = to_paddle_tensor(q_w, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_weight", "k_proj.weight")] = to_paddle_tensor(k_w, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_weight", "v_proj.weight")] = to_paddle_tensor(v_w, target_dtype)
            elif "bias" in key:
                q_b, k_b, v_b = split_attention_weights(bias=np_weight)
                paddle_weights[paddle_key.replace("in_proj_bias", "q_proj.bias")] = to_paddle_tensor(q_b, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_bias", "k_proj.bias")] = to_paddle_tensor(k_b, target_dtype)
                paddle_weights[paddle_key.replace("in_proj_bias", "v_proj.bias")] = to_paddle_tensor(v_b, target_dtype)
            print("#(split) ", key, "->", paddle_key)

        # 常规写入
        paddle_weights[paddle_key] = to_paddle_tensor(np_weight, target_dtype)
        print("#", key, "->", paddle_key, np_weight.shape, target_dtype)

    for key in list(paddle_weights.keys()):
        if key.startswith("model."):
            if "mlp.gate_proj." in key:
                gate_proj = paddle_weights.pop(key)
                up_proj = paddle_weights.pop(key.replace("gate_proj", "up_proj"))
                paddle_weights[key.replace("gate_proj", "up_gate_proj")] = paddle.concat([gate_proj, up_proj], axis=-1)

            if "self_attn.q_proj" in key:
                q_proj = paddle_weights.pop(key)
                k_proj = paddle_weights.pop(key.replace("q_proj", "k_proj"))
                v_proj = paddle_weights.pop(key.replace("q_proj", "v_proj"))
                paddle_weights[key.replace("q_proj", "qkv_proj")] = paddle.concat([q_proj, k_proj, v_proj], axis=-1)

    # 保存（保持原键 & 去掉 vision_model. 前缀各一份）
    paddle.save(paddle_weights, paddle_weight_path)

    print("Saved keys (full):", len(paddle_weights))


def convert_weights2(safetensors_weight_path, paddle_weight_path):
    paddle_safttensors_weights = load_file(safetensors_weight_path, device="cpu")

    # 保存（保持原键 & 去掉 vision_model. 前缀各一份）
    paddle.save(paddle_safttensors_weights, paddle_weight_path)

    print("Saved keys (full):", len(paddle_safttensors_weights))

# Example usage
# convert_weights(
#     "/ssd3/sunting/infer_vlm/20250830-174657-1drope-checkpoint-5774/model.safetensors",
#     "/ssd3/sunting/infer_vlm/checkpoint-5774-1drope-pdformers/model_state.pdparams",
# )

root_dir = "/root/paddlejob/workspace/env_run/laipeiwen"

output_dir = osp.join(root_dir, "PaddleOCR-Ernie", "model_state.pdparams")
print(f"Output directory: {output_dir}")

# mdoel_path = osp.join("PaddleOCR-VL-PT-SFT", "checkpoint-pretrain_model-3DRoPE-25026")
# mdoel_path = osp.join("PaddleOCR-VL-PT-SFT", "checkpoint-pt-3DRoPE-25026")
mdoel_path = osp.join("code4git/ERNIE/ocr", "5120sample_4packing_8acc_20step_lr")
print(f"Loading model state dict from {mdoel_path}")
mdoel_path = osp.join(root_dir, mdoel_path, "model-00001-of-00001.safetensors")

convert_weights2(
    mdoel_path,
    output_dir,
)