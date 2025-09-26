import paddle
import paddle.nn.functional as F
import numpy as np
import sys
import os.path as osp

paddle.set_printoptions(linewidth=1000)

def inbatch_pack_offset_to_attn_mask_start_row_indices(inbatch_pack_offset):
    inbatch_pack_offset = inbatch_pack_offset.numpy()
    attn_mask_row_start_indices = []
    min_start_row = np.inf
    for bidx in range(inbatch_pack_offset.shape[0]):
        item = inbatch_pack_offset[bidx]
        cumsum_item = item[item != -1]
        record_lens = cumsum_item[1:] - cumsum_item[0:-1]
        min_start_row = min(cumsum_item[1], min_start_row)
        row_start_indices = np.repeat(cumsum_item[1:], record_lens)
        attn_mask_row_start_indices.append(row_start_indices[None, None, ...])
    attn_mask_row_start_indices = np.concatenate(attn_mask_row_start_indices, axis=0)
    return paddle.to_tensor(attn_mask_row_start_indices, dtype=paddle.int32)

def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m

# with open('attention_mask.txt', 'w') as f:
#     sys.stdout = f

#     image_grid_thw = [[1 , 2, 2], [1 , 3, 3], [1 , 4, 4]]

#     cu_seqlens = [0]

#     pro = 0
#     for idx, thw in enumerate(image_grid_thw):
#         thw_tuple = tuple(thw)
#         numel = np.prod(thw_tuple)
#         cu_seqlens.append(cu_seqlens[-1] + numel)

#     print(cu_seqlens)

#     grid_thw = paddle.to_tensor(image_grid_thw)
#     cu_seqlens = paddle.repeat_interleave(
#         grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
#     )
#     cu_seqlens = cu_seqlens.cumsum(axis=0, dtype="int32")
#     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

#     print(f"cu_seqlens = {cu_seqlens}")

#     # FlashAttentionVarlen cu_seqlens to FlashMask mask
#     cu_seqlens_rm_first = cu_seqlens[1:]
#     cu_seqlens_rm_last = cu_seqlens[:-1]
#     repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

#     print(repeats)

#     startend_row_indices_lts = paddle.repeat_interleave(
#         cu_seqlens_rm_first, repeats
#     ).reshape([1, 1, -1, 1])
#     startend_row_indices_ute = paddle.repeat_interleave(
#         cu_seqlens_rm_last, repeats
#     ).reshape([1, 1, -1, 1])
#     startend_row_indices = paddle.concat(
#         [startend_row_indices_lts, startend_row_indices_ute], axis=-1
#     )
#     print(startend_row_indices)

#     m = flashmask_to_densemask(startend_row_indices, dtype="float32", causal=True)

#     print(m)

#     m = flashmask_to_densemask(startend_row_indices, dtype="float32", causal=False)

#     print(m)

#     cu_seqlens = cu_seqlens.unsqueeze(0)
#     startend_row_indices = inbatch_pack_offset_to_attn_mask_start_row_indices(cu_seqlens)
#     startend_row_indices = startend_row_indices.unsqueeze(-1)

#     print(startend_row_indices)

#     m = flashmask_to_densemask(startend_row_indices, dtype="float32", causal=True)

#     print(m)

#     cu_seqlens = paddle.to_tensor([0, 5])
#     cu_seqlens = cu_seqlens.unsqueeze(0)
#     startend_row_indices = inbatch_pack_offset_to_attn_mask_start_row_indices(cu_seqlens)
#     startend_row_indices = startend_row_indices.unsqueeze(-1)

#     print(startend_row_indices)

#     m = flashmask_to_densemask(startend_row_indices, dtype="float32", causal=True)

#     print(m)

idx = None

def tensor_md5sum(tensor):
    """
    计算张量(tensor)的MD5哈希值
    
    参数:
        tensor: numpy.ndarray 或 torch.Tensor
            输入的张量数据
    
    返回:
        str: 输入张量的MD5哈希值(十六进制字符串)
    
    异常:
        TypeError: 如果输入不是numpy.ndarray或torch.Tensor
    """
    # 参数校验
    if not isinstance(tensor, (np.ndarray, paddle.Tensor)):
        raise TypeError("输入必须是numpy.ndarray或torch.Tensor")
    
    # 如果是PyTorch张量，先转换为numpy数组
    if isinstance(tensor, paddle.Tensor):
        if tensor.dtype == paddle.bfloat16:
            tensor = tensor.astype("float32")
    
    return tensor._md5sum()

def test_md5():

    print("float32")
    print(paddle.to_tensor([1.23, 4.56])._md5sum())

    print("bfloat16")
    print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16')._md5sum())
    print(tensor_md5sum(paddle.to_tensor([1.23, 4.56]).astype('bfloat16')))

    print("bfloat16 -> float32")
    print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32")._md5sum())
    print(tensor_md5sum(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32")))

    print("bfloat16 -> float32 -> bfloat16")
    print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32").astype("bfloat16")._md5sum())
    print(tensor_md5sum(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32").astype("bfloat16")))

    bf16_torch_tensor = np.load("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT/bf16_tensor.npy")
    bf16_torch_tensor = paddle.to_tensor(bf16_torch_tensor)
    print(bf16_torch_tensor._md5sum())
    bf16_torch_tensor = bf16_torch_tensor.astype("bfloat16")
    print(bf16_torch_tensor._md5sum())

    bf16_torch_emebdding = np.load("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT/bf16_embedding.npy")
    bf16_torch_emebdding = paddle.to_tensor(bf16_torch_emebdding)
    print(bf16_torch_emebdding._md5sum())
    bf16_torch_emebdding = bf16_torch_emebdding.astype("bfloat16")
    print(bf16_torch_emebdding._md5sum())
    bf16_torch_emebdding = bf16_torch_emebdding.astype("bfloat16").astype("float32")
    print(bf16_torch_emebdding._md5sum())


def evaluate_tensor(swift_input, ernie_input):

    torch_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT", swift_input+".npy"))

    paddle_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE", ernie_input+".npy"))
    
    print(f"\n\n{swift_input} Diff")

    torch_input = paddle.to_tensor(torch_input)
    print("torch tensor md5 = ", tensor_md5sum(torch_input))

    paddle_input = paddle.to_tensor(paddle_input)
    print("paddle tensor md5 = ", tensor_md5sum(paddle_input))

    torch_mean = paddle.mean(torch_input)
    paddle_mean = paddle.mean(paddle_input)
    mean_diff = (torch_mean - paddle_mean) / torch_mean
    print(f"mean diff = {mean_diff.item()*100} % ")
    mean_diff = (torch_mean - paddle_mean) / paddle_mean
    print(f"mean diff = {mean_diff.item()*100} % ")

    torch_std = paddle.std(torch_input)
    paddle_std = paddle.std(paddle_input)
    std_diff = (torch_std - paddle_std) / torch_std
    print(f"std diff = {std_diff.item()*100} % ")
    std_diff = (torch_std - paddle_std) / paddle_std
    print(f"std diff = {std_diff.item()*100} % ")

    denominator = paddle.maximum(paddle.abs(torch_input) + 1e-8, paddle.abs(paddle_input) + 1e-8)

    zero_torch = ~(paddle.abs(torch_input) > 0)
    zero_paddle = ~(paddle.abs(paddle_input) > 0)
    zero_mask = paddle.logical_or(zero_torch, zero_paddle)
    zero_num = paddle.sum(zero_mask)

    diff = paddle.abs(torch_input - paddle_input)
    relative_zero_diff = diff[zero_mask]
    relative_diff = diff[~zero_mask] / denominator[~zero_mask]
    
    topk_num=1
    max_diff, max_diff_idx = paddle.topk(relative_diff.flatten(), k=topk_num)
    if zero_num:
        max_zero_diff, max_zero_diff_idx = paddle.topk(relative_zero_diff.flatten(), k=topk_num)

    for i in range(topk_num):
        print(f"max relative element diff = {max_diff[i].item()*100} % ")
        print(f"corresponding torch element = {torch_input[~zero_mask].flatten()[max_diff_idx[i]].item()}")
        print(f"corresponding paddle element = {paddle_input[~zero_mask].flatten()[max_diff_idx[i]].item()}")

        if idx = None:
            idx = 

        if zero_num:
            print(f"max element diff with zero = {max_zero_diff[i].item()}")
            print(f"corresponding torch element = {torch_input[zero_mask].flatten()[max_zero_diff_idx[i]].item()}")
            print(f"corresponding paddle element = {paddle_input[zero_mask].flatten()[max_zero_diff_idx[i]].item()}")

    # max_diff = paddle.max(diff)
    # max_diff_idx = paddle.argmax(diff)
    # print(f"max relative element diff = {max_diff.item()*100} % ")
    # print(f"corresponding torch element = {torch_input.flatten()[max_diff_idx].item()}")
    # print(f"corresponding paddle element = {paddle_input.flatten()[max_diff_idx].item()}")

    
    # relative_torch_diff = paddle.abs(diff / (torch_input+1e-8))
    # max_diff = paddle.max(relative_torch_diff)
    # max_diff_idx = paddle.argmax(relative_torch_diff)
    # print(f"max relative element diff with torch = {max_diff.item()*100} % ")
    # print(f"corresponding diff element = {relative_torch_diff.flatten()[max_diff_idx].item()}")
    # print(f"corresponding torch element = {torch_input.flatten()[max_diff_idx].item()}")
    # print(f"corresponding paddle element = {paddle_input.flatten()[max_diff_idx].item()}")

    # relative_paddle_diff = paddle.abs(diff / (paddle_input+1e-8))
    # max_diff = paddle.max(relative_paddle_diff)
    # max_diff_idx = paddle.argmax(relative_paddle_diff)
    # print(f"max relative element diff with paddle = {max_diff.item()*100} % ")
    # print(f"corresponding diff element = {relative_paddle_diff.flatten()[max_diff_idx].item()}")
    # print(f"corresponding torch element = {torch_input.flatten()[max_diff_idx].item()}")
    # print(f"corresponding paddle element = {paddle_input.flatten()[max_diff_idx].item()}")


def compare_tensor(swift_input, ernie_input, path="both"):

    if path == "both":
        torch_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT", swift_input+".npy"))

        paddle_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE", ernie_input+".npy"))
    elif path == "ernie":
        torch_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE", swift_input+".npy"))

        paddle_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE", ernie_input+".npy"))
    elif path == "swift":
        torch_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT", swift_input+".npy"))

        paddle_input = np.load(osp.join("/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT", ernie_input+".npy"))
    
    if path == "both":
        print(f"\n\nSwift:{swift_input} - Ernie:{ernie_input} Diff")
    else:
        print(f"\n\n{path}:{swift_input} - {path}:{ernie_input} Diff")

    torch_input = paddle.to_tensor(torch_input)
    print("tensor md5 = ", tensor_md5sum(torch_input))

    paddle_input = paddle.to_tensor(paddle_input)
    print("tensor md5 = ", tensor_md5sum(paddle_input))

    # torch_mean = paddle.mean(torch_input)
    # paddle_mean = paddle.mean(paddle_input)
    # mean_diff = (torch_mean - paddle_mean) / torch_mean
    # print(f"mean diff = {mean_diff.item()*100} % ")
    # mean_diff = (torch_mean - paddle_mean) / paddle_mean
    # print(f"mean diff = {mean_diff.item()*100} % ")

    # torch_std = paddle.std(torch_input)
    # paddle_std = paddle.std(paddle_input)
    # std_diff = (torch_std - paddle_std) / torch_std
    # print(f"std diff = {std_diff.item()*100} % ")
    # std_diff = (torch_std - paddle_std) / paddle_std
    # print(f"std diff = {std_diff.item()*100} % ")

    denominator = paddle.maximum(paddle.abs(torch_input) + 1e-8, paddle.abs(paddle_input) + 1e-8)

    zero_torch = ~(paddle.abs(torch_input) > 0)
    zero_paddle = ~(paddle.abs(paddle_input) > 0)
    zero_mask = paddle.logical_or(zero_torch, zero_paddle)
    zero_num = paddle.sum(zero_mask)

    diff = paddle.abs(torch_input - paddle_input)
    relative_zero_diff = diff[zero_mask]
    relative_diff = diff[~zero_mask] / denominator[~zero_mask]
    
    topk_num=1
    max_diff, max_diff_idx = paddle.topk(relative_diff.flatten(), k=topk_num)
    if zero_num:
        max_zero_diff, max_zero_diff_idx = paddle.topk(relative_zero_diff.flatten(), k=topk_num)

    for i in range(topk_num):
        print(f"max relative element diff = {max_diff[i].item()*100} % ")
        print(f"corresponding element = {torch_input[~zero_mask].flatten()[max_diff_idx[i]].item()}")
        print(f"corresponding element = {paddle_input[~zero_mask].flatten()[max_diff_idx[i]].item()}")

        if zero_num:
            print(f"max element diff with zero = {max_zero_diff[i].item()}")
            print(f"corresponding element = {torch_input[zero_mask].flatten()[max_zero_diff_idx[i]].item()}")
            print(f"corresponding element = {paddle_input[zero_mask].flatten()[max_zero_diff_idx[i]].item()}")


def test_interpolate():

    # image = paddle.to_tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).astype("float32")
    # print(image)

    # image = F.interpolate(
    #     image,
    #     size=(2, 4),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # print(image)

    img_hw = [(10, 86), (10, 82), (10, 88), (10, 84)]
    siglip_pos_embedding = [f"position_embeddings_h{h}_w{w}" for h, w in img_hw]
    siglip_image_embedding = [f"image_embeddings_h{h}_w{w}" for h, w in img_hw]
    siglip_image_embedding_add_pos = [f"image_embeddings_add_pos_h{h}_w{w}" for h, w in img_hw]

    siglip_embedding = siglip_pos_embedding + siglip_image_embedding + siglip_image_embedding_add_pos

    tensor_name_list = ["vit_embedding", "siglip_rope_emb_cos", "siglip_rope_emb_sin", "siglip_attn_q", "siglip_attn_k", "siglip_attn_v"]

    # for tensor_name in siglip_embedding:
    #     evaluate_tensor(tensor_name, tensor_name)

    compare_tensor(siglip_pos_embedding[-1], siglip_image_embedding[-1], "ernie")
    compare_tensor(siglip_pos_embedding[-1], siglip_image_embedding[-1], "swift")
    compare_tensor(siglip_pos_embedding[-1], siglip_pos_embedding[-1], "both")
    compare_tensor(siglip_image_embedding[-1], siglip_image_embedding[-1], "both")
    compare_tensor(siglip_image_embedding_add_pos[-1], siglip_image_embedding_add_pos[-1], "both")

    

def add_diff_reprodcution():

    swift_path = "/root/paddlejob/workspace/env_run/laipeiwen/PaddleOCR-VL-PT-SFT"
    ernie_path = "/root/paddlejob/workspace/env_run/laipeiwen/code4git/ERNIE"

    img_hw = [(10, 86), (10, 82), (10, 88), (10, 84)]
    siglip_pos_embedding = [f"position_embeddings_h{h}_w{w}" for h, w in img_hw]
    siglip_image_embedding = [f"image_embeddings_h{h}_w{w}" for h, w in img_hw]
    siglip_image_embedding_add_pos = [f"image_embeddings_add_pos_h{h}_w{w}" for h, w in img_hw]




if __name__ == "__main__":
    test_interpolate()