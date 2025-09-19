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
import math
from collections import OrderedDict
from copy import deepcopy

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.layers.mpu.mp_layers import VocabParallelEmbedding
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute
from paddleformers.transformers.model_utils import PipelinePretrainedModel
from paddleformers.utils.log import logger

from .configuration_ocr import PPOCRVLConfig
from .distributed import ScatterOp, mark_as_sequence_parallel_parameter
from .modeling_moe_vl import TokenType
from .modeling_ocr import Projector
from .modeling_ocr_ernie import (
    Ernie4_5DecoderLayer,
    Ernie4_5LMHead,
    ErniePretrainingCriterion,
)
# from .modeling_ocr_ernie import (
#     Ernie4_5Attention,
#     Ernie4_5DecoderLayer,
#     Ernie4_5LMHead,
#     Ernie4_5MLP,
#     Ernie4_5PretrainedModel,
#     Ernie4_5RotaryEmbedding,
#     ErniePretrainingCriterion,
#     LayerNorm,
#     RMSNorm,
# )
from .sequence_parallel_utils import GatherOp


def get_attr(layer, name):
    """Return attribute from layer's inner layers recursively until found."""
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args, mtp_enable=False):
    """
    Parses input arguments and converts them into model-ready format.

    Processes different input argument patterns into standardized hidden states,
    attention masks and position IDs tensors. All output tensors will have
    stop_gradient=True flag set.

    Args:
        args (Union[tuple, paddle.Tensor]): Input arguments which can be either:
            - Tuple containing 3 elements: (hidden_states, attention_mask, position_ids)
            - Tuple containing 2 elements: (hidden_states, attention_mask)
            - Tuple containing 1 element: (hidden_states)
            - Single tensor: hidden_states
            If rope_embeddings are provided, they should be included in the tuple.

    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            Returns a tuple containing:
            - hidden_states (paddle.Tensor): Processed hidden states
            - attention_mask (Optional[paddle.Tensor]): Attention mask if provided
            - position_ids (Optional[paddle.Tensor]): Position IDs if provided
            All returned tensors have stop_gradient=True.
    """
    if isinstance(args, tuple):
        if not mtp_enable:
            nbatch_pack_offset = None

        if len(args) == 4:
            hidden_states, attention_mask, position_ids, nbatch_pack_offset = args
        elif len(args) == 3:
            if mtp_enable:
                hidden_states, attention_mask, nbatch_pack_offset = args
                position_ids = None
            else:
                hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            if mtp_enable:
                hidden_states, nbatch_pack_offset = args
                attention_mask = None
            else:
                hidden_states, attention_mask = args
            position_ids = None
        elif len(args) == 1:
            (hidden_states,) = args
            attention_mask = None
            position_ids = None
            nbatch_pack_offset = None
    else:
        hidden_states = args
        attention_mask, position_ids, nbatch_pack_offset = None, None, None
    # need position_ids to compute value for PPO.
    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if nbatch_pack_offset is not None:
        nbatch_pack_offset.stop_gradient = True

    return hidden_states, attention_mask, position_ids, nbatch_pack_offset


def return_args(hidden_states, attention_mask=None, position_ids=None):
    """
    Packages model outputs into a standardized return format.

    Returns either a single tensor or a tuple containing hidden states and
    optional attention masks/position IDs. All returned tensors are cloned
    to prevent modification of original inputs.

    Args:
        hidden_states (paddle.Tensor): Model output tensor with shape
            (batch_size, seq_len, hidden_size).
        attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            with shape (batch_size, seq_len). Defaults to None.
        position_ids (Optional[paddle.Tensor]): Position IDs tensor
            with shape (batch_size, seq_len). Defaults to None.

    Returns:
        Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
            Returns either:
            - Single tensor if only hidden_states provided
            - Tuple containing (hidden_states, attention_mask, position_ids)
              based on provided arguments
            All returned tensors are cloned copies.
    """
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]
    return ret

class Ernie4_5EmbeddingPipe(nn.Layer):
    """Pipeline-compatible embedding layer"""
    
    def __init__(self, config):
        """
        Initializes the embedding layer with model configuration.

        Args:
            config (Config): Model configuration.
        """
        super().__init__()
        self.sequence_parallel = config.sequence_parallel
        self.config = config
        
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    
    @property
    def embedding_weight(self):
        """
        Provides access to the underlying embedding weights.

        Returns:
            paddle.Tensor: The weight matrix of shape [vocab_size, hidden_size]
        """
        return self.embed_tokens.weight

    def forward(self, args):
        """
        Performs embedding lookup and attention mask preprocessing.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input arguments which can be:
                - Tuple containing (input_ids, attention_mask, position_ids)
                - Single tensor containing input_ids

        Returns:
            Union[Tuple, paddle.Tensor]: Returns either:
                - Tuple containing (embeddings, processed_attention_mask, position_ids)
                - Single tensor of embeddings if no masks/positions provided

        Note:
            - Automatically generates position_ids if not provided
            - Supports sequence parallel redistribution of embeddings
        """
        input_ids, attention_mask, position_ids, nbatch_pack_offset = parse_args(
            args, self.config.num_nextn_predict_layers > 0
        )
        input_ids.stop_gradient = True
        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        if self.config.num_nextn_predict_layers > 0:
            if self.config.enable_mtp_magic_send:
                emb = emb[:, : -self.config.num_nextn_predict_layers, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                inputs_embeds_extra = emb[
                    :, -self.config.num_nextn_predict_layers :, :
                ]  # [B, S, D]
                inputs_embeds = emb[:, : -self.config.num_nextn_predict_layers, :]
                inputs_embeds_ori = inputs_embeds

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                mtp_emb_res = [inputs_embeds]
                for depth in range(self.config.num_nextn_predict_layers):
                    inputs_embeds_mtp = paddle.concat(
                        [
                            inputs_embeds_ori[:, (depth + 1) :, :],
                            inputs_embeds_extra[:, : (depth + 1), :],
                        ],
                        axis=1,
                    )
                    if self.sequence_parallel:
                        inputs_embeds_mtp = inputs_embeds_mtp.reshape(
                            [-1, inputs_embeds_mtp.shape[-1]]
                        )
                        inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)

                    mtp_emb_res.append(inputs_embeds_mtp)
                res = paddle.concat(mtp_emb_res)
                ret = (res,)
        else:
            if self.sequence_parallel:
                emb = emb.reshape([-1, emb.shape[-1]])
                emb = ScatterOp.apply(emb)

            ret = (emb,)

        if attention_mask is not None:
            if attention_mask.dtype != paddle.int32:
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]

                attention_mask = paddle.scale(
                    x=attention_mask.astype(emb.dtype),
                    scale=1000000.0,
                    bias=-1.0,
                    bias_after_scale=False,
                )

        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if nbatch_pack_offset is not None:
            ret += (nbatch_pack_offset.clone(),)
        if len(ret) == 1:
            ret = ret[0]
        return ret


class Ernie4_5VLEmbeddingPipe(Ernie4_5EmbeddingPipe):
    """Embedding + Resampler"""

    def __init__(self, config, use_full_recompute=False):
        config = deepcopy(config)
        sequence_parallel = config.sequence_parallel
        config.sequence_parallel = False  # disable inner`ScatterOp`
        self.use_full_recompute = use_full_recompute
        self.offload_resamler = False  # config.pp_recompute_offload_resampler
        # out_dim = config.hidden_size
        super().__init__(config)
        if config.mm_vocab_size > 0:
            self.mm_embed_tokens = VocabParallelEmbedding(
                config.mm_vocab_size, config.hidden_size
            )
        else:
            self.mm_embed_tokens = None
        self.mlp_AR = Projector(config, config.vision_config)
        self.config = config
        self.scatter_output = sequence_parallel  # outer `ScatterOp`
        self.use_mem_eff_attn = config.use_mem_eff_attn

    def forward(self, args):
        """forward lm embedding + mm embedding + resampler"""
        # assert len(args) == 4, args
        super_forward = super().forward
        token_type_ids, input_ids, *args = args

        def get_args(args, need_inbatch, need_image, need_varres, need_pos):
            """
            get args: inbatch, position-id, image, image_type_ids, grid_thw
            """
            assert isinstance(args, (tuple, list)), type(args)
            keys = [
                i
                for i, j in zip(
                    [
                        "inbatch",
                        "images",
                        "image_type_ids",
                        "grid_thw",
                        "position_ids",
                    ],  # args 的出现顺序
                    [need_inbatch, need_image, need_image, need_varres, need_pos],
                )
                if j
            ]
            args = dict(zip(keys, args))
            return (
                args.get("inbatch"),
                args.get("images"),
                args.get("image_type_ids"),
                args.get("grid_thw"),
                args.get("position_ids"),
                # args.get("audio"),
            )

        # inbatch_pack_offset, image_features, image_type_ids, grid_thw, position_ids, audio_ids = get_args(
        inbatch_pack_offset, image_features, image_type_ids, grid_thw, position_ids = (
            get_args(
                args,
                self.use_mem_eff_attn,  # inbatch, False
                self.config.vision_config is not None,  # image-type-ids
                getattr(
                    self.config.vision_config, "variable_resolution", False
                ),  # varres
                self.config.rope_3d,  # position-ids
            )
        )

        if inbatch_pack_offset is not None:
            inbatch_pack_offset.stop_gradient = True

        if position_ids is not None:
            position_ids.stop_gradient = True

        token_type_ids_input = token_type_ids[..., :-1]
        token_type_ids_input_ori = token_type_ids_input.clone()
        image_mask = input_ids == self.config.im_patch_id

        token_type_ids_input = token_type_ids_input.flatten()
        input_ids = input_ids.flatten()

        token_type_ids_input[token_type_ids_input == TokenType.video] = TokenType.image
        input_ids.stop_gradient = False  # make recompute happy
        if image_features is not None:
            image_features.stop_gradient = False

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()
        if self.mm_embed_tokens is not None:
            lm_input_ids[token_type_ids_input == TokenType.image] = 0
            mm_input_ids[token_type_ids_input == TokenType.text] = (
                self.config.max_text_id
            )

        def fwd(image_features, _):
            nonlocal input_ids, lm_input_ids, mm_input_ids, token_type_ids_input, image_type_ids, image_mask
            """recompute"""
            assert lm_input_ids.max() < self.config.vocab_size, lm_input_ids.tolist()

            inputs_embeds = super_forward(lm_input_ids)
            if isinstance(inputs_embeds, tuple):
                inputs_embeds = inputs_embeds[0]
            if image_features is not None:  # text sample will pass through vit
                # mapping_forward
                if self.use_full_recompute and self.training:
                    image_features = recompute(
                        self.mlp_AR,
                        image_features,
                        grid_thw,
                        # offload_indices=[0, 1] if self.offload_resamler else [],
                    )
                else:
                    image_features = self.mlp_AR(
                        image_features,
                        grid_thw,
                    )
                # B, N, C = image_features.shape
                # image_features = image_features.reshape([B * N, C])

                if self.mm_embed_tokens is not None:
                    mm_ids_features = self.mm_embed_tokens(
                        mm_input_ids - self.config.max_text_id
                    )
                    mm_ids_features = mm_ids_features.astype(inputs_embeds.dtype)
                    image_indices = paddle.nonzero(
                        token_type_ids_input == TokenType.image
                    ).flatten()
                    inputs_embeds = paddle.scatter_(
                        inputs_embeds,
                        image_indices,
                        paddle.gather(mm_ids_features, image_indices, axis=0),
                        overwrite=True,
                    )
                # else:
                # assert (mm_input_ids <= self.config.max_text_id).all().item(), (
                #     f"found vistual token in ids, but `mm_vocab_size` == 0, "
                #     f"ids:{input_ids}, max_text_id={self.config.max_text_id} "
                # )
                image_indices = paddle.nonzero(image_mask.flatten()).flatten()
                image_features = image_features.reshape([-1, image_features.shape[-1]])
                inputs_embeds = paddle.scatter_(
                    inputs_embeds,
                    image_indices,
                    image_features.astype(inputs_embeds.dtype),
                    overwrite=True,
                )

            if self.scatter_output:
                inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            else:
                inputs_embeds = inputs_embeds.reshape(
                    token_type_ids_input_ori.shape + [inputs_embeds.shape[-1]]
                )

            return inputs_embeds

        # `image_features` could be none, add fake tensor to make recompute happy
        fake_tensor = paddle.zeros([])
        fake_tensor.stop_gradient = False

        inputs_embeds = fwd(image_features, fake_tensor)

        # modify video token type to image token type for expert gating
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image
        ret = (token_type_ids, inputs_embeds)
        if position_ids is not None:
            ret += (position_ids,)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset,)
        return ret



class Ernie4_5_DecoderLayerPipe(Ernie4_5DecoderLayer):
    """Pipeline-compatible decoder layer"""
    
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx, use_full_recompute=False)
        self.config = config
        self.layer_idx = layer_idx
        self.use_full_recompute = use_full_recompute
        self.use_meme_eff_attn = config.use_mem_eff_attn  # fix by liaojincheng
        self.sequence_parallel = config.sequence_parallel
        self.rope_3d = config.rope_3d
    
    def forward(self, args):
        """forward"""

        if len(args) == 2:
            token_type_ids, hidden_states = args
            inbatch_pack_offset = None
            position_ids = None
        elif len(args) == 3:
            if self.rope_3d:
                token_type_ids, hidden_states, position_ids = args
                inbatch_pack_offset = None
            else:
                token_type_ids, hidden_states, inbatch_pack_offset = args
                position_ids = None
                inbatch_pack_offset.stop_gradient = True
        elif len(args) == 4:
            token_type_ids, hidden_states, position_ids, inbatch_pack_offset = args

        token_type_ids = token_type_ids.clone()
        if inbatch_pack_offset is not None:
            attn_mask_start_row_indices = (
                inbatch_pack_offset_to_attn_mask_start_row_indices(inbatch_pack_offset)
            )
        else:
            attn_mask_start_row_indices = None

        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute
            and self.config.recompute_granularity == "full"
            and has_gradient
        ):
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx % v1 == v2 else []
                )
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx in offload_value else []
                )

            hidden_states = recompute(
                super().forward,
                hidden_states,
                None,  # attention_mask,
                attn_mask_start_row_indices,  # attn_mask_start_row_indices
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past key_value
                False,  # use-cache
                False,  # output_gate_logits
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                None,  # attention_mask,
                attn_mask_start_row_indices,  # attn_mask_start_row_indices
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past key_value
                False,  # use-cache
                False,  # output_gate_logits
            )
        ret = (token_type_ids, hidden_states)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        return ret
        
    

class RMSNormPipe(RMSNorm):
    """Pipeline-compatible RMSNorm"""
    
    def __init__(self, config):
        super().__init__(config)
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            
    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return return_args(hidden_states, attention_mask, position_ids)

class LayerNormPipe(LayerNorm):
    """Pipeline-compatible LayerNorm"""
    
    def __init__(self, config):
        super().__init__(config)
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            mark_as_sequence_parallel_parameter(self.bias)
            
    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return return_args(hidden_states, attention_mask, position_ids)

class Ernie4_5LMHeadPipe(Ernie4_5LMHead):
    """
    Pipeline-compatible Language Model Head for ERNIE MoE models.
    """

    def forward(self, args):
        """
        Computes language model logits from hidden states in pipeline-compatible manner.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input which can be:
                - Tuple containing (hidden_states, attention_mask, position_ids)
                - Single tensor of hidden_states
                Note: Attention mask and position IDs are ignored in processing

        Returns:
            paddle.Tensor: Output logits tensor with shape:
                [batch_size, sequence_length, vocab_size]
                representing unnormalized log probabilities for each token
        """
        if self.config.num_nextn_predict_layers > 0:
            logits = list()
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits
        else:
            hidden_states, _, _, _ = parse_args(args)
            logits = super().forward(hidden_states)  # 返回tensor
            return logits

    @property
    def embedding_weight(self):
        """Return the LM head embedding weights"""
        return get_attr(self, "weight")

class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    """
    Pipeline-compatible pretraining criterion for ERNIE models.
    """

    def __init__(self, config):
        """
        Initializes the pretraining criterion with model configuration.

        Args:
            config (Config): Model configuration.
        """
        super().__init__(config)

    def forward(self, logits, labels):
        """
        Computes pretraining loss with optional loss masking.

        Args:
            logits (Union[paddle.Tensor, Tuple[paddle.Tensor]]): Model predictions which can be:
                - Single tensor of shape [batch_size, seq_len, vocab_size]
                - Tuple of tensors for multiple prediction heads
            labels (Union[paddle.Tensor, Tuple[paddle.Tensor]]): Ground truth which can be:
                - Single tensor of shape [batch_size, seq_len]
                - Tuple containing (labels_tensor, loss_mask_tensor)

        Returns:
            Union[paddle.Tensor, Tuple]:
                During training:
                    - Single loss tensor for backpropagation
                During evaluation:
                    - Tuple containing (summed_loss, loss_components) for detailed monitoring
        """
        if isinstance(labels, tuple):
            labels, loss_mask = labels
        else:
            labels, loss_mask = labels, None
        if self.config.num_nextn_predict_layers > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss, loss_sum = super().forward(
                logits, labels, loss_mask, mtp_logits=mtp_logits
            )
            if not self.training:
                return loss_sum
            return loss
        else:
            loss, loss_sum = super().forward(logits, labels, loss_mask)
            if not self.training:
                return loss_sum
            return loss