import paddle
import paddle.nn.functional as F
import numpy as np
import sys

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

print("float32")
print(paddle.to_tensor([1.23, 4.56])._md5sum())

print("bfloat16")
print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16')._md5sum())

print("bfloat16 -> float32")
print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32")._md5sum())

print("bfloat16 -> float32 -> bfloat16")
print(paddle.to_tensor([1.23, 4.56]).astype('bfloat16').astype("float32").astype("bfloat16")._md5sum())