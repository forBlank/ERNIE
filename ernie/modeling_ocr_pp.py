# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle Ernie model for ocr with Pipeline Parallelism"""

import ast
import copy
import math
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.nn import functional as F
from paddle.utils.layers_utilsdownload import flatten, map_structure, pack_sequence_as
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddleformers.transformers.model_outputs import ModelOutput
from paddleformers.transformers.model_utils import PipelinePretrainedModel
from paddleformers.utils.log import logger

from .comm_utils import (
    all_gather_varlen,
    gather_varlen,
)
from .configuration_ocr import PPOCRVLConfig
from .modeling_moe import _parse_moe_group
from .modeling_moe_pp import (
    EmptyLayer,
    create_skip_config_for_refined_recompute,
    get_pp_vp_split_layers,
)
from .modeling_moe_vl_pp import (
    multimodal_data_provider,
    gather_tensors_list_in_pp_group,
    get_len_and_offset,
    exchange_pp_imgs_with_thw,
    shard_data_in_pp_group,
)
from .siglip.modeling import SiglipVisionConfig
from .siglip.modeling_pp import SiglipVisionModelPipe
from .modeling_ocr_ernie_pp import (
    RMSNormPipe,
    LayerNormPipe,
    Ernie4_5VLEmbeddingPipe,
    Ernie4_5_DecoderLayerPipe,
    Ernie4_5LMHeadPipe,
    ErniePretrainingCriterionPipe,
)
from .modeling_ocr import PPOCRVLForConditionalGeneration
from .sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)



class PPOCRVLForConditionalGenerationPipe(PipelinePretrainedModel, PipelineLayer):
    """Pipeline-parallel OCR model with conditional generation"""
    
    config_class = PPOCRVLConfig
    _get_tensor_parallel_mappings = (
        PPOCRVLForConditionalGeneration._get_tensor_parallel_mappings
    )
    _resolve_prefix_keys = PPOCRVLForConditionalGeneration._resolve_prefix_keys
    
    def _prepare_pipeline_inputs_func(self, data: Union[List, Dict]):
        """
        Convert input data into a format acceptable by the model, including image processing, text processing, etc.

        Args:
            data (Union[List, Dict]): Input data, which can be a list or a dictionary.
            If it is a list, each element should be a dictionary containing all the inputs required by the model.
            The keys in the dictionary include:
            'images', 'grid_thw', 'input_ids', 'audio_ids',
            'token_type_ids', 'image_type_ids', 'labels', 'audio_labels', 'position_ids'.
            'images' represents image data, 'grid_thw'
            represents size and position information of the image,
            'input_ids' represents text ID, 'audio_ids' represents audio ID,
            'token_type_ids' represents the text type ID,
            'image_type_ids' represents the image type ID,
            'labels' represents labels, 'audio_labels' represents audio labels,
            'position_ids' represents position ID.

        Returns:
            Tuple[Dict, Dict]: Returns two dictionaries.
            The first dictionary contains all the input information for the model,
            including 'token_type_ids', 'input_ids', 'image_fea',
            'image_type_ids', 'global_grid_thw', 'position_ids', 'audio_ids';
            the second dictionary contains label information,
            including 'token_type_ids_shifted', 'labels', 'audio_labels'.

        Raises:
            AssertionError: If data is not a list or a dictionary, an AssertionError will be raised .
        """
        assert isinstance(data, list), type(data)
        if getattr(self.config.vision_config, "variable_resolution", False):
            assert (
                not self.balanced_image_preprocess
            ), "balanced_image_preprocess is not supported in variable_resolution"
        all_keys = [
            "images",
            "grid_thw",
            "input_ids",
            "inbatch_pack_offset",
            "audio_ids",
            "token_type_ids",
            "image_type_ids",
            "labels",
            "audio_labels",
            "position_ids",
        ]
        inputs = []
        for k in all_keys:
            temp = []
            for d in data:
                if k not in d:
                    temp.append(None)
                else:
                    temp.append(d[k])
            inputs.append(temp)

        hcg = get_hcg()
        dp_group = hcg.get_pipe_parallel_group()
        dp_worldsize = hcg.get_pipe_parallel_world_size()
        dp_src_rank = dp_group.ranks[0]
        dp_rank = hcg._get_pipe_parallel_id()
        this_rank = dist.get_rank()

        images, grid_thw, *other_inputs = inputs

        if self.pp_need_data_ranks:
            send_args = [
                grid_thw,
            ] + other_inputs
            recv_args = gather_tensors_list_in_pp_group(send_args, merge_output=False)
            if recv_args is not None:
                recv_args = list(zip(*recv_args))
                (
                    global_grid_thw,
                    ids,
                    inbatch_pack_offset,
                    audio_ids,
                    token_type_ids,
                    image_type_ids,
                    labels,
                    audio_labels,
                    position_ids,
                ) = (sum(args_from_all_pp, []) for args_from_all_pp in recv_args)
            else:
                # middle pp
                global_grid_thw = ids = audio_ids = token_type_ids = image_type_ids = (
                    labels
                ) = audio_labels = position_ids = inbatch_pack_offset = None
        else:
            (
                ids,
                inbatch_pack_offset,
                audio_ids,
                token_type_ids,
                image_type_ids,
                labels,
                audio_labels,
                position_ids,
            ) = other_inputs
            global_grid_thw = grid_thw
        if ids is not None:  # pp0, pp, -1
            token_type_ids = [t.astype("int32") for t in token_type_ids]
            token_type_ids_shifted = [t[:, 1:] for t in token_type_ids]
        else:
            ids = audio_ids = token_type_ids = image_type_ids = (
                token_type_ids_shifted
            ) = labels = audio_labels = inbatch_pack_offset = None

        if self.vision_model is None:
            images = None
            global_grid_thw = None
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    inbatch_pack_offset,
                    images,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
            )

        if (self.pp_need_data_ranks and dp_rank not in self.pp_need_data_ranks) or (
            not self.pp_need_data_ranks and dp_rank != 0
        ):
            images = []

        image_len_before_concat = paddle.to_tensor(
            [len(n) if n is not None else 0 for i, n in enumerate(images)],
            dtype="int32",
        )

        images_is_all_none = paddle.to_tensor(
            all(i is None for i in images), dtype="int32"
        )
        dist.broadcast(images_is_all_none, src=dp_src_rank, group=dp_group)
        if images_is_all_none.item():
            images = None  # no images
            global_grid_thw = None
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    inbatch_pack_offset,
                    images,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
            )

        images = [i for i in images if i is not None]
        images = paddle.concat(images) if len(images) else None  # list -> tensor
        grid_thw = [i for i in grid_thw if i is not None]
        grid_thw = paddle.concat(grid_thw) if len(grid_thw) else None  # list -> tensor

        # start pp data balance
        pp_data_balance = getattr(self.vision_model, "pp_data_balance", False)

        if (
            self.balanced_image_preprocess
            or self.config.offload_pp_data_chunk_size > 0
            or pp_data_balance
        ):
            # to initial group of batch send recv, early do alltoall
            if not hasattr(get_hcg(), "pp_sd_group"):
                pp_sd_group = get_hcg().get_pipe_parallel_group()
                # alltoall to make p2p eager
                fake_data = paddle.ones([pp_sd_group.nranks, 1])
                fake_out = paddle.empty([pp_sd_group.nranks, 1])
                dist.alltoall(fake_out, fake_data, pp_sd_group)
                get_hcg().pp_sd_group = pp_sd_group

        if pp_data_balance:
            # step1: get some infos, like seqlen, grid_thw, for current sort and later restore
            seq_list, seq_idx_list = get_len_and_offset(images.shape[0], dp_group)
            self.vision_model.seq_list = seq_idx_list

            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [0, 0, 1, 0],
                value=1,
            )

            # get offset
            img_idx = paddle.cumsum(grid_thw[:, 1] * grid_thw[:, 2])
            thwsum = img_idx[-1]
            assert (
                thwsum == images.shape[0]
            ), f"thwsum {thwsum}, images.shape {images.shape}"
            img_idx = img_idx[:-1]
            img_idx = F.pad(img_idx, [1, 0], value=0)
            assert (
                img_idx.shape[0] == grid_thw.shape[0]
            ), f"img_idx.shape {img_idx.shape} , grid_thw.shape {grid_thw.shape}"

            # add rank for thw
            rank_column = paddle.full(
                shape=[grid_thw.shape[0], 1], fill_value=dp_rank, dtype=grid_thw.dtype
            )
            gridthw_withid = paddle.concat([grid_thw, rank_column], axis=-1)

            # get offset for thw and img of all pp
            thw_len = paddle.to_tensor(gridthw_withid.shape[0], dtype=paddle.int32)
            thw_len_list = []
            dist.stream.all_gather(thw_len_list, thw_len, group=dp_group)
            gathered_gridthw_withid = all_gather_varlen(
                gridthw_withid, thw_len_list, dp_group
            )
            gathered_img_idx = all_gather_varlen(img_idx, thw_len_list, dp_group)
            gridthw_withid = gathered_gridthw_withid
            img_idx = gathered_img_idx

            # sort by image size
            gridthw_withid = np.array(gridthw_withid, dtype=np.int64)
            img_idx = np.array(img_idx, dtype=np.int64)
            # products = gridthw_withid[:, 1] * gridthw_withid[:, 2]
            # sorted_indices = np.argsort(products)
            sorted_indices = sorted(
                range(gridthw_withid.shape[0]),
                key=lambda i: gridthw_withid[i, 1] * gridthw_withid[i, 2],
            )
            sorted_thw = gridthw_withid[sorted_indices]
            sorted_idx = img_idx[sorted_indices]

            indices = np.arange(sorted_thw.shape[0]) % dp_worldsize
            indices = np.expand_dims(indices, axis=-1)
            sorted_thw = np.concatenate((sorted_thw, indices), axis=-1)
            sorted_thw = paddle.to_tensor(sorted_thw, dtype=gridthw_withid.dtype)
            sorted_idx = paddle.to_tensor(sorted_idx, dtype=img_idx.dtype)

            assert sorted_thw.shape[1] == 5, f"{sorted_thw.shape}"
            self.vision_model.sorted_thw = sorted_thw.clone()
            self.vision_model.sorted_idx = sorted_idx.clone()
            # data exchange
            new_images, new_thw, new_idx, old_idx = exchange_pp_imgs_with_thw(
                images,
                sorted_thw[sorted_thw[:, -2] == dp_rank],
                sorted_idx[sorted_thw[:, -2] == dp_rank],
                sorted_thw[sorted_thw[:, -1] == dp_rank],
                sorted_idx[sorted_thw[:, -1] == dp_rank],
                dp_rank,
                src_rank_index=-2,
                dst_rank_index=-1,
                group=dp_group,
            )

            # data for vit
            images = new_images
            # record old rank and sort rank
            grid_thw_5column = paddle.stack(new_thw, axis=0)
            new_idxes = paddle.to_tensor(new_idx, dtype=img_idx.dtype)
            old_idxes = paddle.to_tensor(old_idx, dtype=img_idx.dtype)
            grid_thw = grid_thw_5column[:, :-2]

        # I dont know why can not release GPU memory, so I using `_clear_data` to clear underlaying GPU memory
        if self.config.offload_pp_data_chunk_size > 0:
            for img in inputs[0]:
                if img is not None:
                    img._clear_data()
        image_len_before_concat_gathered = gather_varlen(
            image_len_before_concat, dst=dp_src_rank, group=dp_group
        )

        @partial(
            shard_data_in_pp_group,
            fwd_batch_size=getattr(self.config.vision_config, "vit_first_fwd_bsz", 128),
            input_is_parallel=len(self.pp_need_data_ranks) > 1,
            is_balanced=self.balanced_image_preprocess,
            offload_pp_data_chunk_size=self.config.offload_pp_data_chunk_size,
        )
        def fwd_image(images, grid_thw):
            # logger.info(f"# image inside shard : {images.shape}")
            if self.image_preprocess is not None:
                assert images.dtype == paddle.uint8, images.dtype
                images = self.image_preprocess.rescale_factor * images.astype("float32")
                images = (
                    images - self.image_preprocess.image_mean_tensor
                ) / self.image_preprocess.image_std_tensor
                images = images.astype("bfloat16")
            else:
                assert images.dtype == paddle.bfloat16, images.dtype
            
            images = images.unsqueeze(0)
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]

            for idx, thw in enumerate(grid_thw):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = paddle.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(
                    paddle.full((numel,), idx, dtype=paddle.int64)
                )
                cu_seqlens.append(cu_seqlens[-1] + numel)

            siglip_position_ids = paddle.concat(siglip_position_ids, axis=0)
            cu_seqlens = paddle.to_tensor(cu_seqlens, dtype=paddle.int32)
            sample_indices = paddle.concat(sample_indices, axis=0)

            image_fea = self.vision_model(
                pixel_values=images,
                image_grid_thw=image_grid_hws,
                position_ids=siglip_position_ids,
                vision_return_embed_list=True,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )

            if self.config.tensor_parallel_degree > 1:
                if getattr(self.config.vision_config, "variable_resolution", False):
                    S, C = image_fea.shape
                    image_fea = image_fea.reshape(
                        [-1, C * self.config.spatial_conv_size**2]
                    )
                image_fea = ScatterOp.apply(image_fea, axis=-1)  # mp 切 Fea
                if getattr(self.config.vision_config, "variable_resolution", False):
                    image_fea = image_fea.reshape([S, -1])
            # logger.info(f"# image-fea inside shard : {image_fea.shape}")

            return image_fea

        if self.balanced_image_preprocess:
            # broadcast image shape if needed
            if len(self.pp_need_data_ranks) < dp_worldsize:
                if self.balanced_image_shape is None:
                    pp_sd_group = get_hcg().pp_sd_group
                    src_rank = pp_sd_group.ranks[0]
                    this_rank = dist.get_rank()
                    if src_rank == this_rank:
                        assert images.ndim == 4, images.shape
                        full_image_shape = paddle.shape(images).cuda().astype("int32")
                    else:
                        full_image_shape = paddle.empty([4], dtype="int32")
                    dist.broadcast(full_image_shape, src_rank, group=pp_sd_group)
                    full_image_shape = full_image_shape.tolist()
                    self.balanced_image_shape = full_image_shape
                if dp_rank not in self.pp_need_data_ranks:
                    assert (
                        images is None
                    ), "pp rank exceed partial pp_need_data must be None"
                    full_image_shape = self.balanced_image_shape
                    full_image_shape[0] = 0
                    images = paddle.empty(full_image_shape, dtype=paddle.uint8)
                else:
                    # check [c, h, w] must be equal
                    assert (
                        images.shape[1:] == self.balanced_image_shape[1:]
                    ), "image shape is not equal to the previous cache shape"

        image_fea = fwd_image(images, grid_thw)

        if pp_data_balance:
            new_seq_list, new_seq_idx_list = get_len_and_offset(
                images.shape[0], dp_group
            )
            new_thw_len_list, new_thw_idx_list = get_len_and_offset(
                grid_thw_5column.shape[0], dp_group
            )

            new_gathered_gridthw_withid = all_gather_varlen(
                grid_thw_5column, new_thw_len_list, dp_group
            )
            new_gathered_img_idx = all_gather_varlen(
                new_idxes, new_thw_len_list, dp_group
            )
            new_gathered_old_idx = all_gather_varlen(
                old_idxes, new_thw_len_list, dp_group
            )
            assert (
                new_gathered_gridthw_withid.shape[0] == new_gathered_img_idx.shape[0]
            ), f"{new_gathered_gridthw_withid.shape[0]} != {new_gathered_img_idx.shape[0]}"
            # gather each pp img seq
            if image_fea is not None:
                new_gathered_gridthw_withid = np.array(
                    new_gathered_gridthw_withid, dtype=np.int64
                )
                new_gathered_img_idx = np.array(new_gathered_img_idx, dtype=np.int64)
                new_gathered_old_idx = np.array(new_gathered_old_idx, dtype=np.int64)
                new_seq_idx_list = np.array(new_seq_idx_list, dtype=np.int64)

                new_fea = []
                for rank in range(dp_group.nranks):
                    # get thw and offset
                    cur_thw = new_gathered_gridthw_withid[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]
                    cur_idx = new_gathered_img_idx[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]
                    old_idx = new_gathered_old_idx[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]

                    sorted_indices = np.argsort(old_idx)
                    sorted_fea_idx = cur_idx[sorted_indices]
                    sorted_fea_thw = cur_thw[sorted_indices]

                    # according to the original offset, restore the order of fea
                    start_offset = (
                        new_seq_idx_list[sorted_fea_thw[:, -1]] + sorted_fea_idx
                    )
                    end_offset = (
                        new_seq_idx_list[sorted_fea_thw[:, -1]]
                        + sorted_fea_idx
                        + sorted_fea_thw[:, 1] * sorted_fea_thw[:, 2]
                    )
                    index_list = [
                        np.arange(start_offset[i], end_offset[i])
                        for i in range(len(start_offset))
                    ]
                    index_list = paddle.to_tensor(
                        np.concatenate(index_list, axis=-1), dtype=paddle.int64
                    )
                    fea = paddle.gather(image_fea, index_list)
                    new_fea.append(fea)
                new_fea = paddle.concat(new_fea, axis=0)
                image_fea = new_fea

        if image_fea is not None:  # pp 0 or LM batch
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    inbatch_pack_offset,
                    image_fea,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
                split_image=image_len_before_concat_gathered.tolist(),
                image_fea_concated=isinstance(image_fea, paddle.Tensor),
            )

        image_fea = None
        return multimodal_data_provider(
            (
                token_type_ids,
                ids,
                inbatch_pack_offset,
                image_fea,
                image_type_ids,
                global_grid_thw,
                position_ids,
                audio_ids,
            ),
            (token_type_ids_shifted, labels, audio_labels),
        )

    def __init__(self, config, recompute=False):
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(
                f"disable FFN tensor model parallel, moe-group={config.moe_group}"
            )
            config.disable_ffn_model_parallel = True

        # add
        config.moe_group_origin = config.moe_group
        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        hcg = get_hcg()

        self.config = config
        self.image_preprocess = None
        self.pp_need_data_ranks = []  # default to all need data
        self.balanced_image_preprocess = (
            config.balanced_image_preprocess
            if hasattr(config, "balanced_image_preprocess")
            else False
        )
        self.balanced_image_shape = None

        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)
        logger.info(f"using vpp={config.virtual_pp_degree}")
        if config.sequence_parallel:
            logger.info(
                f"using sequence_parallel, input seqlen={config.max_sequence_length}"
            )
            assert config.max_sequence_length is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        if isinstance(config.vision_config, SiglipVisionConfig):
            logger.info("variable resolution vision model")
            config.vision_config.variable_resolution = True
        else:
            raise RuntimeError(f"unknown vision_config: {config.vision_config}")

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=Ernie4_5VLEmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    use_full_recompute=config.recompute,
                    config=config,
                ),
                "model",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(
                    Ernie4_5VLEmbeddingPipe,
                    config=config,
                    use_full_recompute=config.recompute,
                ),
                "model",
            )

        no_recompute_layers = get_pp_vp_split_layers(config)

        def _need_full_recompute(layer_idx):
            return layer_idx not in no_recompute_layers and config.recompute

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    Ernie4_5_DecoderLayerPipe,
                    config=create_skip_config_for_refined_recompute(i, config),
                    layer_idx=i,
                    use_full_recompute=_need_full_recompute(i),
                ),
                f"ernie.layers.{i}",
            )

        for i in range(config.add_tail_layers):
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"empty.layers.{i+config.num_hidden_layers}",
            )

        self.add_sequential_layer(
            LayerDesc(
                RMSNormPipe if config.use_rmsnorm else LayerNormPipe, config=config
            ),
            "model.norm",
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=Ernie4_5LMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(Ernie4_5LMHeadPipe, config=config), "lm_head"
            )
        recompute_interval = 0

        seg_method = (
            config.pp_seg_method
            if hasattr(config, "pp_seg_method")
            else "layer:Ernie4_5DecoderLayer|EmptyLayer"
        )
        try:
            result = ast.literal_eval(seg_method)
            if isinstance(result, list):
                seg_method = result
        except Exception:
            pass
        if (
            seg_method == "layer:Ernie4_5DecoderLayer|EmptyLayer"
            and (config.num_hidden_layers + config.add_tail_layers)
            % get_hcg().topology().get_dim_size("pipe")
            != 0
        ):
            seg_method = "uniform"
        logger.info(
            f"using recompute_interval={recompute_interval}, seg_method={seg_method}"
        )

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=ErniePretrainingCriterionPipe(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=config.virtual_pp_degree,
        )
        self._modality_param_mapping = None
        vision_model = SiglipVisionModelPipe(self.config)
        self.add_vision_model(encoder=vision_model)
    
    def add_vision_model(
        self,
        encoder: nn.Layer,
    ):
        """add_vision_model"""
        self.vision_model = encoder

    def add_image_preprocess(self, preprocess):
        """add image_preprocess"""
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess