# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class HGRNBitConfig(PretrainedConfig):

    model_type = 'hgrn_bit'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        attn_mode: str = "fused_recurrent",
        num_heads: Optional[int] = 1,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        share_conv_kernel: bool = True,
        use_lower_bound: bool = True,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_num_experts_per_tok: int = 2,
        moe_router_aux_loss_coef: float = 1e-2,
        moe_router_jitter_noise: float = 0.0,
        moe_router_bias: bool = False,
        moe_normalize_topk_prob: bool = True,
        moe_output_router_logits: bool = False,
        moe_use_quantized_experts: bool = False,
        moe_layer_indices: Optional[list] = None,
        moe_upcycling_noise_scale: float = 0.05,
        moe_noise_alpha: Optional[float] = None,
        moe_noise_mode: str = "legacy_global_std",
        moe_expert_intermediate_factor: float = 1.0,
        moe_expert_intermediate_size: Optional[int] = None,
        moe_init_method: str = "copy_noise",
        moe_grouped_topk: bool = False,
        moe_num_virtual_groups: int = 1,
        moe_topk_per_group: int = 1,
        moe_routing_mode: str = "standard",
        moe_pair_weights: str = "router",
        moe_complement_pairs: Optional[list] = None,
        moe_output_scale: float = 1.0,
        moe_enable_learnable_output_scale: bool = False,
        moe_output_scale_granularity: str = "global",
        moe_initial_output_scale: float = 1.0,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.use_lower_bound = use_lower_bound
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_num_experts_per_tok = moe_num_experts_per_tok
        self.moe_router_aux_loss_coef = moe_router_aux_loss_coef
        self.moe_router_jitter_noise = moe_router_jitter_noise
        self.moe_router_bias = moe_router_bias
        self.moe_normalize_topk_prob = moe_normalize_topk_prob
        self.moe_output_router_logits = moe_output_router_logits
        self.moe_use_quantized_experts = moe_use_quantized_experts
        self.moe_layer_indices = [] if moe_layer_indices is None else list(moe_layer_indices)
        self.moe_upcycling_noise_scale = moe_upcycling_noise_scale
        self.moe_noise_alpha = moe_noise_alpha
        self.moe_noise_mode = moe_noise_mode
        self.moe_expert_intermediate_factor = moe_expert_intermediate_factor
        self.moe_expert_intermediate_size = moe_expert_intermediate_size
        self.moe_init_method = moe_init_method
        self.moe_grouped_topk = moe_grouped_topk
        self.moe_num_virtual_groups = moe_num_virtual_groups
        self.moe_topk_per_group = moe_topk_per_group
        self.moe_routing_mode = moe_routing_mode
        self.moe_pair_weights = moe_pair_weights
        self.moe_complement_pairs = [] if moe_complement_pairs is None else list(moe_complement_pairs)
        self.moe_output_scale = moe_output_scale
        self.moe_enable_learnable_output_scale = moe_enable_learnable_output_scale
        self.moe_output_scale_granularity = moe_output_scale_granularity
        self.moe_initial_output_scale = moe_initial_output_scale
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
