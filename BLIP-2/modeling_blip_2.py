# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BLIP-2 model."""
# 导入模块
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入其他模块中的类和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的检查点
_CHECKPOINT_FOR_DOC = "Salesforce/blip2-opt-2.7b"

# BLIP-2 预训练模型存档列表
BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-opt-2.7b",
    # 查看所有 BLIP-2 模型: https://huggingface.co/models?filter=blip
]

# 定义一个数据类, 用于存储 `Blip2ForConditionalGeneration` 的输出
@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
            语言模型的语言建模损失, 当提供 `labels` 参数时返回
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
            语言模型的语言建模头的预测得分, 形状为 `(batch_size, sequence_length, config.vocab_size)`
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
            视觉编码器的输出
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
            Q-Former (查询 Transformer) 的输出
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
            语言模型的输出
    """
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    # 将对象转换为元组, 返回包含对象所有键值的元组
    def to_tuple(self) -> Tuple[Any]:
        # 遍历对象的所有键
        # 如果键不是特定的键, 则直接将对应的值添加到元组中
        # 如果键是特定的键之一, 则调用相应属性的 to_tuple() 方法, 并将结果添加到元组中
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->Blip2
class Blip2VisionEmbeddings(nn.Module):
    # 初始化 Blip2VisionEmbeddings 类
    def __init__(self, config: Blip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # = 1408
        self.image_size = config.image_size # = 224
        self.patch_size = config.patch_size # = 14

        # 创建表示类别嵌入的可学习参数, 用于表示图像的类别信息
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # shape = [1, 1, vision_config.hidden_size]

        # 创建用于将像素值转换为嵌入向量的卷积层, 用于提取图像的局部特征
        """
        torch.nn.Conv2d: 对由多个输入平面组成的输入信号应用 2D 卷积。
        
        输入 shape = (N, C_in, H, W), 输出 shape = (N, C_out, H_out, W_out)
        N 是批量大小, C 表示通道数, H 是输入平面的高度(以像素为单位), W 是宽度(以像素为单位)

        kernel_size: 卷积核的大小
        stride: 控制交叉相关性的步幅
        """
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # 计算图像中 patch 的数量和位置嵌入的维度
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1 # = 257

        # 创建位置嵌入的可学习参数, 用于表示图像中不同位置的信息
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim)) # shape = [1, 257, vision_config.hidden_size]

    # 前向传播函数, 输入为像素值张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # input(pixel_values) shape = [batch_size, 3, 224, 224]
        
        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]
        # 获取目标数据类型
        target_dtype = self.patch_embedding.weight.dtype
        
        # 将输入像素值转换为 patch 嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [batch_size, 1408, 16, 16]
        # 将 patch 嵌入扁平化并转置, 以便与类别嵌入进行拼接
        # patch_embeds.flatten(2) shape = [batch_size, 1408, 256], 即从第2维开始展平
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # shape = [batch_size, 256, 1408]

        # 扩展类别嵌入以匹配批量大小, 并转换为目标数据类型
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype) # shape = [batch_size, 1, 1408]
        
        # 拼接 类别嵌入 和 patch嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) # shape = [batch_size, 257, 1408]
        # 将 位置嵌入 加到 最终输出 中
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype) # shape = [batch_size, 257, 1408]
        
        return embeddings # 包含图像类型和分块(patch)信息的嵌入


class Blip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化 Blip2Attention 类
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # = 1408
        self.num_heads = config.num_attention_heads # = 16
        self.head_dim = self.embed_dim // self.num_heads # = 88
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout) # config.attention_dropout = 0.0

        # 创建一个线性层, 用于计算查询、键、值的线性变换
        # Linear(in_features=1408, out_features=4224, bias=False)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None
        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)) # shape = 4224
            self.qkv.bias = nn.Parameter(qkv_bias)

        # 创建一个线性层, 用于将多头注意力的输出映射回原始维度
        # Linear(in_features=1408, out_features=1408, bias=True)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状以便用于注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        计算自注意力机制的输出
        Input(hidden_states) shape = batch_size x Time x vision_config.hidden_size
        """
        bsz, tgt_len, embed_dim = hidden_states.size() # shape = [batch_size, 257, vision_config.hidden_size]

        # 使用 QKV 线性层对 隐藏状态 进行线性变换
        mixed_qkv = self.qkv(hidden_states) # shape = [batch_size, 257, 4224]

        # 重塑成多头格式, 以便分离 Q、K、V
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        ) # shape = [batch_size, 257, 4224] --reshape--> [batch_size, 257, 3, num_heads, head_dim] --permute--> [3, batch_size, num_heads, 257, head_dim]
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算原始的注意力分数, 即 query 与 key 的点积
        # [batch_size, num_heads, 257, head_dim] * [batch_size, num_heads, head_dim, 257] = [batch_size, num_heads, 257, 257]
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # shape = [batch_size, num_heads, 257, 257]
        # 执行 dropout 操作
        attention_probs = self.dropout(attention_probs) # shape = [batch_size, num_heads, 257, 257]

        # 如果指定了头掩码, 则执行头掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量, 即先计算 注意力概率 * value, 然后变形
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3) # [batch_size, num_heads, 257, 257] * [batch_size, num_heads, 257, head_dim] ----> [batch_size, num_heads, 257, head_dim] --permute--> [batch_size, 257, num_heads, head_dim]

        # 重塑上下文张量以匹配输入shape
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape) # shape = [batch_size, 257, vision_config.hidden_size]

        # 通过投影层将上下文层映射到输出空间
        output = self.projection(context_layer) # shape = [batch_size, 257, vision_config.hidden_size]

        # 根据是否需要输出注意力权重, 构建返回的输出元组
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# 源自 transformers.models.blip.modeling_blip.BlipMLP
class Blip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act] # gelu 激活函数
        # Linear(in_features=1408, out_features=6144, bias=True)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Linear(in_features=6144, out_features=1408, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # input(hidden_states) shape = [batch_size, 257, vision_config.hidden_size]
        hidden_states = self.fc1(hidden_states)           # shape = [batch_size, 257, vision_config.intermediate_size]
        hidden_states = self.activation_fn(hidden_states) # shape = [batch_size, 257, vision_config.intermediate_size]
        hidden_states = self.fc2(hidden_states)           # shape = [batch_size, 257, vision_config.hidden_size]
        return hidden_states


# 源自 transformers.models.blip.modeling_blip.BlipEncoderLayer
class Blip2EncoderLayer(nn.Module):
    def __init__(self, config: Blip2Config):
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size # = 1408
        # 创建 注意力机制 模块
        self.self_attn = Blip2Attention(config)
        # 创建第一个规范化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建 MLP 模块
        self.mlp = Blip2MLP(config)
        # 创建第二个规范化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                是否返回注意力权重
        """
        # 将输入(隐藏状态)作为残差连接的一部分
        residual = hidden_states # shape = [batch_size, 257, vision_config.hidden_size]
        # 应用第一个规范化层
        hidden_states = self.layer_norm1(hidden_states) # shape = [batch_size, 257, vision_config.hidden_size]
        # 应用注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask, # None
            output_attentions=output_attentions, # False
        ) # shape 不变
        # 将注意力机制的输出与残差连接
        hidden_states = hidden_states + residual # shape 不变
        
        # 更新残差以备后续使用
        residual = hidden_states
        # 应用第二个规范化层
        hidden_states = self.layer_norm2(hidden_states) # shape 不变
        # 应用 MLP 模块
        hidden_states = self.mlp(hidden_states) # shape 不变
        # 将 MLP 模块的输出与残差连接
        hidden_states = hidden_states + residual # shape 不变

        # 将输出打包为元组
        outputs = (hidden_states,)
        # 如果需要返回注意力权重, 则加入输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义一个名为 Blip2PreTrainedModel 的类, 继承自 PreTrainedModel
class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    一个抽象类, 初始化权重, 下载并加载预训练模型
    """
    # 配置类为 Blip2Config
    config_class = Blip2Config
    # 模型前缀为 "blip"
    base_model_prefix = "blip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要跳过的模块名称列表
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    # 在设备中跳过的键名
    _skip_keys_device_placement = "past_key_values"
    # 定义需要保持在 FP32 模块中的模块名称列表
    _keep_in_fp32_modules = ["wo"]

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化范围, 用于初始化标准差
        factor = self.config.initializer_range
        # 如果模块是 卷积层、嵌入层或全连接层
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            # 使用正态分布, 初始化权重
            module.weight.data.normal_(mean=0.0, std=factor)
            # 如果模块有偏置项且不为None, 则将偏置项初始化为0
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        # 如果模块是 Blip2VisionEmbeddings 类型
        if isinstance(module, Blip2VisionEmbeddings):
            # 如果配置中有 vision_config, 获取视觉配置的初始化范围, 用于初始化标准差
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            # 使用截断正态分布, 初始化位置嵌入和类别嵌入
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        # 如果模块是 LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果模块是全连接层且有偏置项
        elif isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项初始化为0
            module.bias.data.zero_()
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Blip2Encoder):
            module.gradient_checkpointing = value


BLIP_2_START_DOCSTRING = r"""
    # 启动文档字符串
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 此模型继承自 [`PreTrainedModel`]. 请查看超类文档, 了解该库为其所有模型实现的通用方法(例如下载或保存、调整输入嵌入的大小、修剪头等)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    # 此模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类. 
    # 将其用作常规 PyTorch 模块, 并参阅 PyTorch 文档, 以了解与通用用法和行为相关的所有事项.
    Parameters:
        config ([`Blip2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        # config ([`Blip2Config`]): 模型配置类, 及模型所有参数.
            # 使用配置文件初始化, 该文件不会加载与模型相关的权重, 只有配置. 查看 [`~PreTrainedModel.from_pretrained`] 方法, 以加载模型权重.
"""

BLIP_2_VISION_INPUTS_DOCSTRING = r"""
    # 视觉输入的文档字符串
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.
            # 像素值. 可以使用 [`Blip2Processor`] 获取像素值. 有关详细信息, 请参阅 [`Blip2Processor.__call__`].
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参阅返回张量中的 `attentions`.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态. 有关更多详细信息, 请参阅返回张量中的 `hidden_states`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组.
"""

BLIP_2_TEXT_INPUTS_DOCSTRING = r""" 
    # 文本输入的文档字符串
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
            # 输入序列词元在词汇表中的索引. 如果您提供填充, 否则默认情况下, 将忽略填充. 可以使用 AutoTokenizer 获取这些索引.
            # 请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情. [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            # 掩码, 避免对填充词元索引执行注意. 掩码值在`[0, 1]`中选取:
            # - 1 表示**未被掩码**的词元, 
            # - 0 表示**被掩码**的词元.
            # [什么是注意掩码?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            # 解码器输入序列词元在词汇表中的索引.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            # 可以使用 [`AutoTokenizer`] 获取索引. 有关详细信息, 请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`].
            [What are decoder input IDs?](../glossary#decoder-input-ids)
            # [什么是解码器输入 ID?](../glossary#decoder-input-ids)
            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
            # T5 使用 `pad_token_id` 作为起始词元用于 `decoder_input_ids` 的生成. 
            # 如果使用 `past_key_values`, 则可选地只有最后一个 `decoder_input_ids` 必须输入 (参见 `past_key_values`).
            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
            # 要了解如何准备 `decoder_input_ids` 进行预训练, 请查看[T5 训练](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
            # 默认行为：生成一个张量, 它忽略 `decoder_input_ids` 中的填充词元. 默认情况下也将使用因果掩码.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参阅返回张量中的 `attentions`.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态. 有关更多详细信息, 请参阅返回张量中的 `hidden_states`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组.
"""

BLIP_2_INPUTS_DOCSTRING = r"""
    # 输入的文档字符串
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.
            # 像素值. 可以使用 [`Blip2Processor`] 获取像素值. 有关详细信息, 请参阅 [`Blip2Processor.__call__`].
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.
            # 输入序列词元在语言模型词汇表中的索引. 可以选择性地提供输入词元作为文本提示, 语言模型可以继续执行该提示.
            Indices can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for details.
            # 可以使用 [`Blip2Processor`] 获取索引. 有关详细信息, 请参阅 [`Blip2Processor.__call__`].
            [What are input IDs?](../glossary#input-ids)
            # [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            # 掩码, 避免对填充词元索引执行注意. 掩码值在`[0, 1]`中选取:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            # - 1 表示**未被掩码**的词元, 
            # - 0 表示**被掩码**的词元.
            [What are attention masks?](../glossary#attention-mask)
            # [什么是注意掩码?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.
            # 解码器输入序列词元在语言模型词汇表中的索引. 仅在使用编码器-解码器语言模型(如 T5)时应用.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)
            # 可以使用 [`AutoTokenizer`] 获取索引. 有关详细信息, 请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]. 
            # [什么是解码器输入 ID?](../glossary#decoder-input-ids)
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
            # 默认行为：生成一个张量, 它忽略 `decoder_input_ids` 中的填充词元. 默认情况下也将使用因果掩码.
            Only relevant in case an encoder-decoder language model (like T5) is used.
            # 仅在使用编码器-解码器语言模型(如 T5)时应用.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参阅返回张量中的 `attentions`.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态. 有关更多详细信息, 请参阅返回张量中的 `hidden_states`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组.
"""


# 源自 transformers.models.blip.modeling_blip.BlipEncoder 
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].
    # Transformer 编码器由 `config.num_hidden_​​layers` 个自注意力层组成.每层是一个 [`Blip2EncoderLayer`].
    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
            # `Blip2Encoder` 对应的视觉配置
    """
    def __init__(self, config: Blip2Config):
        super().__init__()
        self.config = config
        # 创建包含 config.num_hidden_layers(=39) 个 BlipEncoderLayer 实例的 ModuleList
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
                # 输入的嵌入表示. 应为浮点数, 而不是整数词元。
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                # 掩码, 避免对填充词元索引执行注意. 掩码值在`[0, 1]`中选取:                
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                # - 1 表示**未被掩码**的词元, 
                # - 0 表示**被掩码**的词元.
                [What are attention masks?](../glossary#attention-mask)
                # [什么是注意掩码?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                # 是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参阅返回张量中的 `attentions`.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
                # 是否返回所有层的隐藏状态. 有关更多详细信息, 请参阅返回张量中的 `hidden_states`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                # 是否返回 [`~utils.ModelOutput`] 而不是普通元组.
        """
        # 设置是否返回所有注意力层的注意力张量
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # self.config.output_attentions = False
        # 设置是否返回所有层的隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states # self.config.output_hidden_states = False
        )
        # 设置是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不输出隐藏状态, 则将 encoder_states 初始化为空
        encoder_states = () if output_hidden_states else None
        # 如果不输出注意力张量, 则将 all_attentions 初始化为空
        all_attentions = () if output_attentions else None

        # 初始隐藏状态为输入的嵌入表示
        hidden_states = inputs_embeds # shape = [batch_size, 257, vision_config.hidden_size]
        # 遍历编码器的每层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果输出隐藏状态, 则将当前隐藏状态添加到 encoder_states 中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            if (self.gradient_checkpointing # = False
                and self.training):         # = True
                # 如果启用激活检查点且处于训练模式, 则使激活检查点技术
                """
                create_custom_forward 函数——描述在模型或模型的一部分的前向传递中运行的内容. 它还应该知道如何处理(作为元组传递的)输入. 例如, 在 LSTM 中, 如果用户传递 (activation, hidden), 函数应该正确地将第一个输入用作 activation, 将第二个输入用作 hidden
                """
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                """
                torch.utils.checkpoint.checkpoint
                https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint

                激活检查点是一种用计算换取内存的技术. 检查点区域中的前向计算省略保存用于反向传播的张量, 并在反向传递期间重新计算它们, 而不是保持反向所需的张量, 直到在后向传播过程中的梯度计算使用它们.

                目前有两种可用的检查点实现, 由 use_reentrant 参数决定. 建议您使用 use_reentrant=False。
                """
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                # 否则直接调用 Blip2EncoderLayer 层的前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 用 Blip2EncoderLayer 输出的第一个元素更新隐藏状态
            hidden_states = layer_outputs[0] # shape = [batch_size, 257, vision_config.hidden_size]

            # 如果输出注意力张量, 则将当前层的注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态, 则将最终隐藏状态添加到 encoder_states 中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果无需返回字典形式的输出, 则返回元组形式的输出, 包含: 1) 最终隐藏状态, 2) 所有隐藏状态, 3) 所有注意力层的注意力张量
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回字典形式的输出
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# 源自 transformers.models.blip.modeling_blip.BlipVisionModel
class Blip2VisionModel(Blip2PreTrainedModel):
    # 主要输入名称
    main_input_name = "pixel_values"
    # 配置类
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config
        
        # 嵌入维度
        embed_dim = config.hidden_size # = 1408

        # 创建视觉嵌入层 Blip2VisionEmbeddings 
        self.embeddings = Blip2VisionEmbeddings(config)
        # 创建编码器 Blip2Encoder
        self.encoder = Blip2Encoder(config)
        # 创建 LayerNorm 层, 用于后处理
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        """
        PreTrainedModel.post_init()
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1329
        在每个 Transformer 模型初始化结束时执行的方法, 用于执行(需要正确初始化模型的模块的)代码, 例如权重初始化
        """
        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 设置是否返回所有注意力层的注意力张量
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否返回所有层的隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值(输入)为 None, 则引发 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # input(pixel_values) shape = [batch_size, 3, 224, 224]
        # 将像素值传入视觉嵌入层, 并得到隐藏状态
        hidden_states = self.embeddings(pixel_values) # shape = [batch_size, 257, vision_config.hidden_size]

        # 将隐藏状态传入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 编码器输出中的第一个元素为最终隐藏状态
        last_hidden_state = encoder_outputs[0] # shape = [batch_size, 257, vision_config.hidden_size]
        # 后层归一化
        last_hidden_state = self.post_layernorm(last_hidden_state) # shape = [batch_size, 257, vision_config.hidden_size]

        # 从最后隐藏状态中提取池化输出
        pooled_output = last_hidden_state[:, 0, :] # shape = [batch_size, vision_config.hidden_size]
        # todo: 只保留第二维的第一个, 其对应图像的类型嵌入, 舍弃 patch 嵌入, 这叫池化?
        # 后层归一化
        pooled_output = self.post_layernorm(pooled_output) # shape = [batch_size, vision_config.hidden_size]

        # 如果无需返回字典形式的输出, 则返回元组形式的输出, 包含: 1) 最终隐藏状态, 2) 最终隐藏状态的池化输出, 3) encoder_outputs[1:], 其中 encoder_outputs[1:] 包含所有隐藏状态和所有注意力层的注意力张量
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        # 否则返回字典形式的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        # 获取视觉嵌入层
        return self.embeddings

# 源自 transformers.models.bert.modeling_bert.BertSelfAttention
class Blip2QFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        """
        inputs:
            config: 配置
            is_cross_attention: 是否是交叉注意力的 bool 变量
        """
        super().__init__()
        self.config = config
        # 检查嵌入维度(config.hidden_size)能否被注意力头数(config.num_attention_heads)整除, 如果不能则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 注意力头数
        self.num_attention_heads = config.num_attention_heads # = 12
        # 每个注意力头数的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # = 768 / 12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # = 12 * 64 = 768 = qformer_config.hidden_size

        # 创建查询、键和值的线性层
        #   Linear(in_features=768, out_features=768, bias=True)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            #   Linear(in_features=1408, out_features=768, bias=True)
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            #   Linear(in_features=1408, out_features=768, bias=True)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            # Linear(in_features=768, out_features=768, bias=True)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            # Linear(in_features=768, out_features=768, bias=True)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # 0.1
        
        # 设置位置嵌入类型, 默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型是相对位置嵌入, 则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        # 是否保存注意力
        self.save_attention = False

    # 保存注意力梯度
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    # 获取注意力梯度
    def get_attn_gradients(self):
        return self.attn_gradients

    # 保存注意力概率
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    # 获取注意力概率
    def get_attention_map(self):
        return self.attention_map

    # 调整形状以便计算注意力分数
    def transpose_for_scores(self, x):
        # input(x) shape = [batch_size, *, self.all_head_size] -> output shape = [batch_size, self.num_attention_heads, *, self.attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # shape = [batch_size, *, self.num_attention_heads, self.attention_head_size]
        return x.permute(0, 2, 1, 3) # shape = [batch_size, self.num_attention_heads, *, self.attention_head_size]

    def forward(
        self,
        hidden_states,               # 隐藏状态
        attention_mask=None,         # 注意力掩码
        head_mask=None,
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None, # 编码器的注意力掩码
        past_key_value=None,         # 之前记录的(key, value)元组
        output_attentions=False,     # 是否输出注意力概率
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 如果实例化为交叉注意模块, 则键和值来自编码器; 注意力掩码需要使编码器的填充词元不被注意.

        # 如果编码器的隐藏状态非空, 表示为交叉注意力, 则 is_cross_attention = True, 否则 = False
        is_cross_attention = encoder_hidden_states is not None

        # 构造 key & value
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states)) # encoder_hidden_states shape = [batch_size, 257, config.encoder_hidden_size] --self.key-->[batch_size, 257, self.all_head_size] --self.transpose_for_scores--> [batch_size, self.num_attention_heads, 257, self.attention_head_size]
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states)) # 同上
            # 如果是交叉注意力, 用编码器的注意力掩码替换 attention_mask, 保证其 shape 与 attention_scores 一致
            attention_mask = encoder_attention_mask # shape = [batch_size, 1, 1, 257]
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # hidden_states shape = [batch_size, 32, qformer_config.hidden_size] --self.key-->[batch_size, 32, self.all_head_size] --self.transpose_for_scores--> [batch_size, self.num_attention_heads, 32, self.attention_head_size]
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # 同上

        # 构造 query
        # hidden_states shape = [batch_size, 32, qformer_config.hidden_size]
        mixed_query_layer = self.query(hidden_states) # shape = [batch_size, 32, self.all_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer) # shape = [batch_size, self.num_attention_heads, 32, self.attention_head_size]

        # 记录(key, value)
        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算"query"和"key"之间的点积, 得到原始`注意力分数`
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch_size, self.num_attention_heads, 32, self.attention_head_size] * [batch_size, self.num_attention_heads, self.attention_head_size, *] -> [batch_size, self.num_attention_heads, 32, *], 其中当自注意力时, 表示维度的 * = 32; 当交叉注意力时, 表示维度的 * = 257

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # todo
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # shape = [batch_size, self.num_attention_heads, 32, *]
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # [batch_size, self.num_attention_heads, 32, *] + [batch_size, 1, 1, *] -> [batch_size, self.num_attention_heads, 32, *]
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # 将注意力分数归一化为`注意力概率`
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # shape = [batch_size, self.num_attention_heads, 32, *]

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs) # 保存注意力概率
            """
            torch.Tensor.register_hook:
            https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
            https://blog.csdn.net/leviopku/article/details/124630642
            todo
            """
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 这实际上是丢弃了需要注意的全部词元, 这可能看起来有点不寻常, 但取自原始 Transformer 论文.
        attention_probs_dropped = self.dropout(attention_probs) # shape = [batch_size, self.num_attention_heads, 32, *]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        
        # 计算上下文张量, 即注意力概率 * value
        # [batch_size, self.num_attention_heads, 32, *] * [batch_size, self.num_attention_heads, *, self.attention_head_size] -> [batch_size, self.num_attention_heads, 32, self.attention_head_size]
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # shape = [batch_size, 32, self.num_attention_heads, self.attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # shape = [batch_size, 32, self.all_head_size], 保持与第一个输入 hidden_states 的形状相同
        
        # 将上下文张量记录于 outputs 的第一个元素, 将(key, value)元组记录于 outputs 的最后一个元素
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


# 源自 transformers.models.bert.modeling_bert.BertSelfOutput 
class Blip2QFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建线性层: Linear(in_features=768, out_features=768, bias=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层, 对输出进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层, 防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 0.1

    def forward(self, hidden_states: torch.Tensor, 
        input_tensor: torch.Tensor) -> torch.Tensor:
        """
        inputs:
            hidden_states shape = [batch_size, 32, qformer_config.hidden_size]
            input_tensor shape  = [batch_size, 32, qformer_config.hidden_size]
        """
        # 线性层
        hidden_states = self.dense(hidden_states) # [batch_size, 32, qformer_config.hidden_size]
        # Dropout 层
        hidden_states = self.dropout(hidden_states) # [batch_size, 32, qformer_config.hidden_size]
        # 隐藏状态与输入张量相加后, 再进行 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # [batch_size, 32, qformer_config.hidden_size]
        return hidden_states


# 源自 transformers.models.bert.modeling_bert.BertAttention
class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 创建 Blip2QFormerMultiHeadAttention 层
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)
        # 创建 Blip2QFormerSelfOutput 层
        self.output = Blip2QFormerSelfOutput(config)
        # 初始化一个集合, 用于存储被剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # todo: 暂时未被调用, 待测试数据流
        
        # 查找可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        inputs:
            hidden_states,          # 隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
            attention_mask,         # 注意力掩码, shape = [batch_size, 1, 1, 32]
            head_mask,              # None
            encoder_hidden_states,  # 编码器的隐藏状态: 当被用于自注意力层时, 无输入, 采用默认的 None; 当被用于交叉注意力层时, 输入 shape = [batch_size, 257, config.encoder_hidden_size]
            encoder_attention_mask, # 编码器的注意力掩码: 当被用于自注意力层时, 无输入, 采用默认的 None; 当被用于交叉注意力层时, 输入 shape = [batch_size, 1, 1, 257]
            past_key_value,         # 之前记录的(key, value)元组, None
            output_attentions,      # 是否输出注意力概率, False
        """
        # Blip2QFormerMultiHeadAttention 层
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # self_outputs 的第一个元素为上下文张量, 其 shape = [batch_size, 32, qformer_config.hidden_size], 最后一个元素为(key, value)元组
        
        # 通过 Blip2QFormerSelfOutput 层, 构建残差网络
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将 self_outputs 中的其他信息添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 源自 transformers.models.bert.modeling_bert.BertIntermediate
class Blip2QFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建线性层, Linear(in_features=768, out_features=3072, bias=True)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否是字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串类型, 则使用预定义的激活函数字典 ACT2FN 中对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串类型, 则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # inputs(hidden_states) shape = [batch_size, 32, qformer_config.hidden_size]

        # 线性层
        hidden_states = self.dense(hidden_states) # shape = [batch_size, 32, qformer_config.intermediate_size]
        # 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states) # shape = [batch_size, 32, qformer_config.intermediate_size]
        return hidden_states


# 源自 transformers.models.bert.modeling_bert.BertOutput
class Blip2QFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层, Linear(in_features=3072, out_features=768, bias=True)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 0.1

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        inputs:
            hidden_states shape = [batch_size, 32, qformer_config.intermediate_size]
            input_tensor shape  = [batch_size, 32, qformer_config.hidden_size]
        """
        # 线性层
        hidden_states = self.dense(hidden_states) # shape = [batch_size, 32, qformer_config.hidden_size]
        # 使用 Dropout 进行正则化, 减少过拟合现象
        hidden_states = self.dropout(hidden_states) # shape = [batch_size, 32, qformer_config.hidden_size]
        # 使用 LayerNorm 对两者之和进行归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # shape = [batch_size, 32, qformer_config.hidden_size]
        return hidden_states


# 源自 transformers.models.bert.modeling_bert.BertLayer
class Blip2QFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 设置 apply_chunking_to_forward 的输出参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward # = 0
        self.seq_len_dim = 1
        
        # 利用 Blip2QFormerAttention, 构建 自注意力 层
        self.attention = Blip2QFormerAttention(config)

        # 当前层的索引
        self.layer_idx = layer_idx

        # 如果当前层的 layer_idx 被 config.cross_attention_frequency(=2) 整除
        if layer_idx % config.cross_attention_frequency == 0:
            # 则利用 Blip2QFormerAttention, 在自注意力层(self.attention)后, 构建额外的 交叉注意力 层
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        # 创建 Blip2QFormerIntermediate 层, qformer_config.hidden_size -> qformer_config.intermediate_size
        self.intermediate_query = Blip2QFormerIntermediate(config)
        # 创建 Blip2QFormerOutput 层, qformer_config.intermediate_size -> qformer_config.hidden_size
        self.output_query = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        """
        inputs:
            hidden_states           隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
            attention_mask          注意力掩码, shape = [batch_size, 1, 1, 32]
            head_mask               = None
            encoder_hidden_states   编码器的隐藏状态, 每个 Blip2QFormerLayer 层的该输入, 均为 视觉模型的最终隐藏状态, shape = [batch_size, 257, config.encoder_hidden_size]
            encoder_attention_mask  编码器的注意力掩码, shape = [batch_size, 1, 1, 257]
            past_key_value          之前记录的(key, value)元组 = None
            output_attentions       是否输出注意力概率 = False
            query_length = 32       查询长度
        """
        # 解码器单向自注意力缓存的(键,值)位于元组的前2个位置, 如果 past_key_value 不等于 None, 则取其前两个元素作为自注意力的过去(key, value)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 利用 Blip2QFormerAttention, 实现自注意力
        self_attention_outputs = self.attention(
            hidden_states,                           # 隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
            attention_mask,                          # 注意力掩码, shape = [batch_size, 1, 1, 32]
            head_mask,                               # = None
            output_attentions=output_attentions,     # 是否输出注意力概率 = False
            past_key_value=self_attn_past_key_value, # 之前记录的(key, value)元组 = None
        )
        # 获取自注意力输出中的上下文张量
        attention_output = self_attention_outputs[0] # shape = [batch_size, 32, qformer_config.hidden_size]
        # 获取自注意力输出中的`注意力概率`, 如果选择将其输出时
        outputs = self_attention_outputs[1:-1]
        # 获取自注意力输出中的(key, value)缓存
        present_key_value = self_attention_outputs[-1] # key and value shape = [batch_size, self.num_attention_heads, 32, self.attention_head_size]

        # 如果查询长度大于0
        if query_length > 0:
            # 截取(自注意力输出中的)上下文张量的部分
            query_attention_output = attention_output[:, :query_length, :] # shape = [batch_size, query_length, qformer_config.hidden_size]
            # 如果存在交叉注意力
            if self.has_cross_attention:
                # 如果编码器隐藏状态为空, 表示不是交叉注意力, 则抛出异常
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                
                # 利用 Blip2QFormerAttention, 实现交叉注意力
                cross_attention_outputs = self.crossattention(
                    query_attention_output,              # 隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
                    attention_mask,                      # 注意力掩码, shape = [batch_size, 1, 1, 32]
                    head_mask,                           # = None
                    encoder_hidden_states,               # 编码器的隐藏状态, 每个 Blip2QFormerLayer 层的该输入, 均为 视觉模型的最终隐藏状态, shape = [batch_size, 257, config.encoder_hidden_size]
                    encoder_attention_mask,              # 编码器的注意力掩码, shape = [batch_size, 1, 1, 257]
                    output_attentions=output_attentions, # 是否输出注意力概率 = False
                )
                # 获取交叉注意力输出中的上下文张量
                query_attention_output = cross_attention_outputs[0] # shape = [batch_size, query_length, qformer_config.hidden_size]
                # 将交叉注意力输出中的`注意力概率`添加到 outputs 中, 如果选择将其输出时
                outputs = outputs + cross_attention_outputs[1:-1]

            # 对自注意力或交叉注意力输出中的上下文张量 应用 分块前向传播
            """
            pytorch_utils.apply_chunking_to_forward:
            将分块应用于前向传播
            refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L166
            This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
            此函数在`chunk_dim`维上, 将`input_tensors`分块为较小输入张量部分, 大小为`chunk_size`. 然后, 它将`forward_fn`层独立应用于每个块, 以节省内存.
            If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly applying `forward_fn` to `input_tensors`.
            如果`forward_fn`独立于`chunk_dim`, 则该函数将产生与直接将`forward_fn`应用于`input_tensors`相同的结果.
            Args:
                forward_fn (`Callable[..., torch.Tensor]`):
                    The forward function of the model.
                    模型的前向函数
                chunk_size (`int`):
                    The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
                    块大小
                chunk_dim (`int`):
                    The dimension over which the `input_tensors` should be chunked.
                    维度, 在该维度上, `input_tensors`被分块
                input_tensors (`Tuple[torch.Tensor]`):
                    The input tensors of `forward_fn` which will be chunked
                    `forward_fn`的输入张量, 它将被分块
            Returns:
                `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
                如果应用, 则会产生与 `forward_fn` 相同形状的张量
            """
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward, # = 0, 未施行分块!!
                self.seq_len_dim,             # = 1
                query_attention_output, # shape = [batch_size, query_length, qformer_config.hidden_size]
            ) # layer_output shape = [batch_size, query_length, qformer_config.hidden_size]

        """
            # 如果自注意力输出中的上下文张量(attention_output)的第2维长度大于查询长度
            if attention_output.shape[1] > query_length:
                # 对注意力输出中剩余部分应用分块前向传播
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk, # 函数内有未定义的
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                # 拼接查询部分和剩余部分的输出
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 对注意力输出应用分块前向传播
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, # 函数内有未定义的
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        """

        # 将当前层最终输出(layer_output)添加到 outputs 中的第一个位置
        outputs = (layer_output,) + outputs
        # 将自注意力输出中的(key, value)缓存添加到 outputs 中的最后一个位置
        outputs = outputs + (present_key_value,)
        # outputs 包含: 1) 当前层最终输出(layer_output); 2) 自注意力输出中的`注意力概率`(如果 output_attentions = True); 3)交叉注意力输出中的`注意力概率`(如果 output_attentions = True); 4) 自注意力输出中的(key, value)缓存
        return outputs

    """
    # 前向传播分块函数, 用于处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 中间层计算
        intermediate_output = self.intermediate(attention_output) # self.intermediate 未定义
        # 输出层计算
        layer_output = self.output(intermediate_output, attention_output) # self.output 未定义
        return layer_output
    """

    # 应用分块技术的前向传播函数
    def feed_forward_chunk_query(self, attention_output):
        # input(attention_output) shape = [batch_size, query_length, qformer_config.hidden_size]
        # Blip2QFormerIntermediate 层
        intermediate_output = self.intermediate_query(attention_output) # shape = [batch_size, query_length, qformer_config.intermediate_size]
        # 利用 Blip2QFormerOutput, 构建残差网络
        layer_output = self.output_query(intermediate_output, attention_output) # shape = [batch_size, query_length, qformer_config.hidden_size]
        return layer_output


# 源自 transformers.models.bert.modeling_bert.BertEncoder
class Blip2QFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个由多个 Blip2QFormerLayer 组成的层列表, 层数由 config.num_hidden_layers(=12) 决定
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        """
        inputs:
            hidden_states           # 隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
            attention_mask          # 注意力掩码, shape = [batch_size, 1, 1, 32]
            head_mask               # = [None, None, None, None, None, None, None, None, None, None, None, None]
            encoder_hidden_states   # 编码器的隐藏状态, shape = [batch_size, 257, config.encoder_hidden_size]
            encoder_attention_mask  # 编码器的注意力掩码, shape = [batch_size, 1, 1, 257]
            past_key_values         # 之前记录的(key, value)元组 = None
            use_cache               # 是否使用缓存 = None
            output_attentions       # 是否输出注意力概率 = False
            output_hidden_states    # 是否输出隐藏状态 = False
            return_dict             # 是否返回字典形式的输出 = True
            query_length            # 查询长度 = 32
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 遍历编码器的每层, 即 Blip2QFormerLayer 层
        for i in range(self.config.num_hidden_layers):
            # 获取当前层的模块
            layer_module = self.layer[i]

            # 如果需要输出隐藏状态, 则将当前层输入的隐藏状态(即上一层输出的隐藏状态)添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 获取当前层的 head_mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取之前记录的(key, value)元组
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点且处于训练模式, 则使激活检查点技术
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 如果使用缓存, 警告不兼容, 并将 use_cache 设置为 False
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                
                # 激活检查点
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 否则直接调用当前 Blip2QFormerLayer 层的前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states, # 每个 Blip2QFormerLayer 层的该输入, 均为 视觉模型的最终隐藏状态
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            # 将隐藏状态更新为当前层输出中的第一个元素(即上下文张量)
            hidden_states = layer_outputs[0]

            # 如果使用缓存, 则将当前层输出中的最后一个元素(即自注意力输出中的(key, value)缓存)添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力, 则将当前层输出中的第二个元素(即自注意力输出中的`注意力概率`)添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果当前层有交叉注意力, 则将当前层输出中的第三个元素(即交叉注意力输出中的`注意力概率`)添加到 all_cross_attentions 中
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        # 如果需要输出隐藏状态, 则将最后一层输出的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果无需返回字典形式的输出, 则返回元组形式的输出
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则返回字典形式的输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,        # 最后一层输出中的 隐藏状态(即上下文张量)
            past_key_values=next_decoder_cache,     # 所有层输出中的 自注意力输出中的(key, value)缓存
            hidden_states=all_hidden_states,        # 输入的隐藏状态 + 所有层输出中的 隐藏状态(即上下文张量)
            attentions=all_self_attentions,         # 所有层输出中的 自注意力输出中的(key, value)缓存
            cross_attentions=all_cross_attentions,  # 所有(包含交叉注意力)层输出中的 交叉注意力输出中的(key, value)缓存
        )


# 源自 transformers.models.bert.modeling_bert.BertModel
class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.

    The BertModel can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    BertModel 既可以充当编码器(仅具有自注意力), 也可以充当解码器, 其中在自注意力层之间添加了一个交叉注意力层, 遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的架构.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    要充当解码器, 需要将配置的 `is_decoder` 参数设置为 `True` 来初始化模型. 要在 Seq2Seq 模型中使用, 需要将 `is_decoder` 参数和 `add_cross_attention` 设置为 `True` 来初始化模型；然后预期 `encoder_hidden_​​states` 作为前向传播的输入.
    """

    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config)
        self.config = config

        # 创建 LayerNorm 层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建 Q-Former 编码器
        self.encoder = Blip2QFormerEncoder(config)

        """
        PreTrainedModel.post_init():
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1329
        在每个 Transformer 模型初始化结束时执行的方法, 用于执行(需要正确初始化模型的模块的)代码, 例如权重初始化
        """
        self.post_init()

    """
    def get_input_embeddings(self):
        # 获取输入嵌入层的单词嵌入
        return self.embeddings.word_embeddings # self.embeddings未定义

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的单词嵌入
        self.embeddings.word_embeddings = value # self.embeddings未定义
    """

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        剪枝模型的头. heads_to_prune 是 {layer_num: 此层中要剪枝的头列表} 字典, 请参阅基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        ModuleUtilsMixin.get_extended_attention_mask:
        refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1039
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        制作可广播的注意力和因果掩码, 以便忽略未来和掩码词元.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
                掩码, 其中 1 表示需要注意的词元, 0 表示需要忽略的词元
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
                模型输入的形状
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
            扩展的注意力掩码, 其 dtype 与 `attention_mask.dtype` 相同.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力掩码, 在这种情况下, 我们只需使其可广播到所有头.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # 提供了一个填充掩码, 其维度为 [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # - 模型是一个编码器, 因此使掩码可广播到 [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :] # [batch_size, 32] -> [batch_size, 1, 1, 32]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 因为对于我们想注意的位置, attention_mask 为 1.0, 而对于被掩码的位置, attention_mask 为 0.0, 所以该操作将创建一个张量, 对于我们想注意的位置, 其值为 0.0, 而对于被掩码的位置, 其值为 -10000.0.因为我们在 softmax 之前将其添加到原始分数中, 所以这实际上与完全删除它们相同.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        query_embeds   # 随机初始化的可学习参数, shape = [batch_size, 32, qformer_config.hidden_size]
        attention_mask # 注意力掩码, = None
        head_mask      # = None
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
            编码器最后一层输出的隐藏状态. 如果模型配置为解码器, 则用于交叉注意力
            视觉模型的最终隐藏状态 shape = [batch_size, 257, config.encoder_hidden_size]
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            掩码, 避免对编码器输入的填充词元索引执行注意. 如果模型配置为解码器, 则此掩码用于交叉注意力. 掩码值在`[0, 1]`中选取: 
            - 1 表示**未被掩码**的词元, 
            - 0 表示**被掩码**的词元.
            全1张量, shape = [batch_size, 257]
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
            包含注意力块的预计算的键和值隐藏状态. 可用于加速解码. 如果使用 `past_key_values`, 用户可以选择仅输入形状为 `(batch_size, 1)` 的最后一个 `decoder_input_ids`(那些没有将其过去键值状态提供给此模型), 而不是形状为 `(batch_size,sequence_length)` 的所有 `decoder_input_ids`.
            = None
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
            如果设置为`True`, 则返回`past_key_values` 的键值状态, 可用于加速解码(参见`past_key_values`).
            # 是否使用缓存 = None
        output_attentions
            # 是否输出注意力概率 = None
        output_hidden_states
            # 是否输出隐藏状态 = None
        return_dict
            # 是否返回字典形式的输出 = True
        """
        # # 为 Blip2QFormerEncoder 前向传播准备 输入8
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # self.config.output_attentions = False
        
        # # 为 Blip2QFormerEncoder 前向传播准备 输入9
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states # self.config.output_hidden_states = False
        
        # # 为 Blip2QFormerEncoder 前向传播准备 输入10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # self.config.use_return_dict = True

        # # 为 Blip2QFormerEncoder 前向传播准备 输入11
        query_length = query_embeds.shape[1] if query_embeds is not None else 0 # = 32

        # # 为 Blip2QFormerEncoder 前向传播准备 输入1
        embedding_output = self.layernorm(query_embeds)   # shape = [batch_size, 32, qformer_config.hidden_size]
        embedding_output = self.dropout(embedding_output) # shape = [batch_size, 32, qformer_config.hidden_size]

        past_key_values_length = past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0 # = 0
        input_shape = embedding_output.size()[:-1] # = [batch_size, 32]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        # # 为 Blip2QFormerEncoder 前向传播准备 输入2
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device) # 全1张量, shape = [batch_size, 32]

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力掩码, 在这种情况下, 我们只需使其可广播到所有头.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device) # shape = [batch_size, 1, 1, 32]

        # # 为 Blip2QFormerEncoder 前向传播准备 输入5
        # If a 2D or 3D attention mask is provided for the cross-attention,
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 如果为交叉注意力提供 2D 或 3D 注意力掩码, 我们需要使其可广播到 [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size() # [batch_size, 257, config.encoder_hidden_size]
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length) # = (batch_size, 257)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device) # shape = [batch_size, 257]
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask) # shape = [batch_size, 1, 1, 257]
            else:
                """
                ModuleUtilsMixin.invert_attention_mask:
                https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L987
                todo
                """
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask) # [batch_size, 257] -> [batch_size, 1, 1, 257]
        else:
            encoder_extended_attention_mask = None

        # # 为 Blip2QFormerEncoder 前向传播准备 输入3
        # Prepare head mask if needed
        # 如果需要, 准备头掩码
        # 1.0 in head_mask indicate we keep the head
        # head_mask 中的 1.0 表示我们保留头
        # attention_probs has shape bsz x n_heads x N x N
        # attention_probs 的形状为 bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 输入 head_mask 的形状为 [num_heads] 或 [num_hidden_​​layers x num_heads], 并且 head_mask 被转换为形状 [num_hidden_​​layers x batch x num_heads x seq_length x seq_length]
        """
        ModuleUtilsMixin.get_head_mask:
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1091
        todo
        """
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) # None -> [None, None, None, None, None, None, None, None, None, None, None, None]

        # Blip2QFormerEncoder 前向传播
        encoder_outputs = self.encoder(
            embedding_output,                                       # 1.隐藏状态, shape = [batch_size, 32, qformer_config.hidden_size]
            attention_mask=extended_attention_mask,                 # 2. 注意力掩码, shape = [batch_size, 1, 1, 32]
            head_mask=head_mask, # 3. = [None, None, None, None, None, None, None, None, None, None, None, None]
            encoder_hidden_states=encoder_hidden_states,            # 4. 编码器的隐藏状态(即视觉模型的最终隐藏状态), shape = [batch_size, 257, config.encoder_hidden_size]
            encoder_attention_mask=encoder_extended_attention_mask, # 5. 编码器的注意力掩码, shape = [batch_size, 1, 1, 257]
            past_key_values=past_key_values,                        # 6. 之前记录的(key, value)元组 = None
            use_cache=use_cache,                                    # 7. 是否使用缓存 = None
            output_attentions=output_attentions,                    # 8. 是否输出注意力概率 = False
            output_hidden_states=output_hidden_states,              # 9. 是否输出隐藏状态 = False
            return_dict=return_dict,                                # 10. 是否返回字典形式的输出 = True
            query_length=query_length,                              # 11. 查询长度 = 32
        )
        sequence_output = encoder_outputs[0] # 最终隐藏状态(即上下文张量), shape = [batch_size, 32, qformer_config.hidden_size]
        pooled_output = sequence_output[:, 0, :] # 最终隐藏状态的池化输出, shape = [batch_size, qformer_config.hidden_size]

        # 如果无需返回字典形式的输出, 则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        # 否则返回字典形式的输出
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output, # 最终隐藏状态(即上下文张量)
            pooler_output=pooled_output, # 最终隐藏状态的池化输出
            past_key_values=encoder_outputs.past_key_values, # 所有层输出中的 自注意力输出中的(key, value)缓存
            hidden_states=encoder_outputs.hidden_states, # embedding_output + 所有层输出中的 隐藏状态(即上下文张量)
            attentions=encoder_outputs.attentions, # 所有层输出中的 自注意力输出中的(key, value)缓存
            cross_attentions=encoder_outputs.cross_attentions, # 所有(包含交叉注意力)层输出中的 交叉注意力输出中的(key, value)缓存
        )


# 使用装饰器为类添加文档字符串
@add_start_docstrings(
    """
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    BLIP-2 模型, 用于生成文本和图像特征. 该模型由视觉编码器、Querying Transformer (Q-Former) 和语言模型组成.
    """,
    BLIP_2_START_DOCSTRING,
)
# 定义 Blip2Model 类，继承自 Blip2PreTrainedModel
class Blip2Model(Blip2PreTrainedModel):
    # 配置类
    config_class = Blip2Config
    # 主要输入名称
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        # 创建视觉模型, 主要由视觉嵌入(Blip2VisionEmbeddings)和 Transformer 编码器(Blip2Encoder)组成
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 创建可学习的查询令牌参数
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        
        # 创建 Q-Former 模型
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # 创建语言投影层，将查询 Transformer 的隐藏状态映射到文本配置的隐藏大小
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        # 根据配置选择加载语言模型，支持从 config.text_config 创建自回归语言模型或序列到序列语言模型
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 根据基础模型更新 _tied_weights_keys
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # 设置语言模型
        self.language_model = language_model

        """
        PreTrainedModel.post_init()
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1329
        在每个 Transformer 模型初始化结束时执行的方法, 用于执行(需要正确初始化模型的模块的)代码, 例如权重初始化
        """
        self.post_init()

    # 获取语言模型的输入嵌入层
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置语言模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 设置语言模型的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # 获取语言模型的输出嵌入层
    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    # 获取语言模型的编码器
    def get_encoder(self):
        return self.language_model.get_encoder()

    # 获取语言模型的解码器
    def get_decoder(self):
        return self.language_model.get_decoder()

    # 绑定权重
    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(BLIP_2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```py"""
        # 如果未提供值，则使用配置中的参数值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果配置为仅使用解码器，只使用语言模型
        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # 获取输入嵌入
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # 使用语言模型
            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        # 返回文本输出
        return text_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        返回：
            vision_outputs (`BaseModelOutputWithPooling` 或者 `torch.FloatTensor` 的元组):
                视觉模型的输出。如果 `return_dict=True`，输出是一个包含图像特征、汇聚图像特征和隐藏状态（如果`output_hidden_states=True`）的 [`BaseModelOutputWithPooling`]。
        示例:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_outputs = model.get_image_features(**inputs)
        ```py"""
        # 如果未提供output_attentions参数，则使用配置中的output_attentions参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供output_hidden_states参数，则使用配置中的output_hidden_states参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供return_dict参数，则使用配置中的use_return_dict参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用视觉模型获取特征
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回视觉模型的输出
        return vision_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    def get_qformer_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```py"""
        # 设置输出注意力权重，默认为模型配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取图像嵌入
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # 创建图像注意力掩码，全为1，大小与图像嵌入的形状相同
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # 扩展查询令牌以匹配图像嵌入的形状
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 通过 QFormer 前向传播查询令牌，使用图像嵌入进行跨注意力
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回查询输出
        return query_outputs

    # 添加模型前向传播的注释文档字符串
    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    # 定义一个前向传播函数，用于模型推断
    def forward(
        # 输入像素值的张量，通常是图像数据，类型为浮点张量
        self,
        pixel_values: torch.FloatTensor,
        # 输入序列的标识符张量，通常是输入文本的编码，类型为浮点张量
        input_ids: torch.FloatTensor,
        # 注意力掩码张量，用于指示哪些标记是填充的，哪些是真实的，类型为可选的长整型张量
        attention_mask: Optional[torch.LongTensor] = None,
        # 解码器输入序列的标识符张量，类型为可选的长整型张量
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器注意力掩码张量，用于指示哪些标记是填充的，哪些是真实的，类型为可选的长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 是否返回注意力权重张量的标志，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏状态张量的标志，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 标签张量，通常用于计算损失，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,
        # 是否返回字典格式的输出，类型为可选的布尔值
        return_dict: Optional[bool] = None,
# 导入所需模块或函数
@add_start_docstrings(
    """
    # BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision
    # encoder, Querying Transformer (Q-Former) and a language model.
    # BLIP-2 模型, 用于根据图像和可选的文本提示生成文本. 该模型由视觉编码器、Querying Transformer (Q-Former) 和语言模型组成.

    # One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    # the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
    # 您可以选择将`input_ids`传递给模型, 作为文本提示, 使语言模型继续提示. 否则, 语言模型将从 [BOS] (beginning-of-sequence)词元开始生成文本.

    # <Tip>

    # Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.
    # 注意, Flan-T5 检查点不能转换为 float16. 它们是使用 bfloat16 进行预训练的.

    # </Tip>
    """,
    BLIP_2_START_DOCSTRING,
)
class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    # 配置类
    config_class = Blip2Config
    # 主要输入名称
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        # 创建视觉模型, 主要由视觉嵌入(Blip2VisionEmbeddings)和 Transformer 编码器(Blip2Encoder)组成
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 创建可学习的查询词元参数
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)) # shape = [1, 32, qformer_config.hidden_size]
        
        # 创建 Q-Former 模型
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # 创建语言投影(线性)层, 将 Q-Former 输出中的最终隐藏状态, 从 qformer_config.hidden_size 维映射到 text_config.hidden_size 维
        # Linear(in_features=768, out_features=2560, bias=True)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # 根据配置创建语言模型
        # 根据配置选择加载语言模型
        if config.use_decoder_only_language_model:
            # 创建自回归语言模型
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            # 创建序列到序列语言模型
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 根据已选的基础语言模型, 更新 _tied_weights_keys
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys] # ['language_model.lm_head.weight']

        # 设置语言模型
        self.language_model = language_model

        """
        PreTrainedModel.post_init()
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1329
        在每个 Transformer 模型初始化结束时执行的方法, 用于执行(需要正确初始化模型的模块的)代码, 例如权重初始化
        """
        self.post_init()

    # 获取语言模型的输入嵌入层
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置语言模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 设置语言模型的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # 获取语言模型的输出嵌入层
    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    # 获取语言模型的编码器
    def get_encoder(self):
        return self.language_model.get_encoder()

    # 获取语言模型的解码器
    def get_decoder(self):
        return self.language_model.get_decoder()

    # 绑定权重
    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
    
    # 定义一个私有方法
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        一些预处理技巧使得模型`加速`兼容.
        查看 https://github.com/huggingface/transformers/pull/21707 了解更多详细信息.
        """
        # 获取 HF 设备映射
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # 当使用多 GPU + BLIP-2 + `accelerate` 时, 警告用户可能会出现异常行为
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # 对 `generate` 兼容

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:
        1. 图像字幕(不提供文本提示)
        Image captioning (without providing a text prompt):

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ... )
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```

        2. 视觉问答(提示 = 问题)
        Visual question answering (prompt = question):

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        ... )  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```

        3. 通过 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 支持 int8 推理. 在保持相同性能的同时, 大大降低了模型占用的内存量.
        Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
        This greatly reduces the amount of memory used by the model while maintaining the same performance.

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
        ... )  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```

        inputs:
            pixel_values            # shape = [batch_size, 3, 224, 224]
            input_ids               # 提示索引, shape = [batch_size, 13], todo: 外部输入, 待从主函数中确认
            attention_mask          # 注意力掩码 = None
            decoder_input_ids       # = None
            decoder_attention_mask  # = None
            output_attentions       # 是否输出注意力概率 = None
            output_hidden_states    # 是否输出隐藏状态 = None
            labels                  # = input_ids, 用于计算loss, shape = [batch_size, 13]
            return_dict             # 是否返回字典形式的输出 = None
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # self.config.use_return_dict = True

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        # step 1: 通过视觉编码器前向传播图像, 以获得图像嵌入, 其形状为 (batch_size, seq_len, hidden_​​size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,                  # shape = [batch_size, 3, 224, 224]
            output_attentions=output_attentions,        # = None
            output_hidden_states=output_hidden_states,  # = None
            return_dict=return_dict,                    # = True
        )
        image_embeds = vision_outputs[0] # 视觉模型的最终隐藏状态, shape = [batch_size, 257, vision_config.hidden_size]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # step 2: 通过 Q-Former 前向传播查询词元, 将图像嵌入用于交叉注意力
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device) # 全1张量, shape = [batch_size, 257]

        # self.query_tokens shape = [1, 32, qformer_config.hidden_size]
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # shape = [batch_size, 32, qformer_config.hidden_size]
        query_outputs = self.qformer(
            query_embeds=query_tokens,                  # 随机初始化的可学习参数, shape = [batch_size, 32, qformer_config.hidden_size]
            encoder_hidden_states=image_embeds,         # 视觉模型的最终隐藏状态, shape = [batch_size, 257, vision_config.hidden_size]
            encoder_attention_mask=image_attention_mask,# 全1张量, shape = [batch_size, 257]
            output_attentions=output_attentions,        # = None
            output_hidden_states=output_hidden_states,  # = None
            return_dict=return_dict,                    # = True
        )
        query_output = query_outputs[0] # 最终隐藏状态(即上下文张量), shape = [batch_size, 32, qformer_config.hidden_size]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # step 3: 使用语言模型, 以查询输出和提示为输入
        language_model_inputs = self.language_projection(query_output) # 语言模型输入(即查询输出投影), shape = [batch_size, 32, text_config.hidden_size]
        
        # 通过语言模型的输入嵌入层(Embedding(50272, 2560, padding_idx=1)), 提示索引(input_ids) shape = [batch_size, 13] -> 提示输入嵌入(inputs_embeds) shape = [batch_size, 13, text_config.hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # 提示输入嵌入, shape = [batch_size, 13, text_config.hidden_size]
        # 在第2维上, 将 语言模型输入(查询输出投影, language_model_inputs) 和 提示输入嵌入(inputs_embeds) 拼接为 输入嵌入
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1) # 输入嵌入, shape = [batch_size, 45, text_config.hidden_size]

        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        ) # 语言模型注意力掩码(即查询输出投影注意力掩码), 全1张量, shape = [batch_size, 32]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) # 提示注意力掩码, 全1张量, shape = [batch_size, 13]
        expected_device = language_model_attention_mask.device
        # 在第2维上, 将 语言模型注意力掩码(查询输出投影注意力掩码, language_model_attention_mask) 和 提示注意力掩码(attention_mask) 拼接为 注意力掩码
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1) # shape = [batch_size, 45]

        if self.config.use_decoder_only_language_model:
            # 自回归语言模型
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,        # = None
                output_hidden_states=output_hidden_states,  # = None
                return_dict=return_dict,                    # = True
            )
            logits = outputs.logits if return_dict else outputs[0] # shape = [batch_size, 45, text_config.vocab_size]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            # 我们在这里计算损失, 因为我们需要考虑查询嵌入的序列长度
            if labels is not None:
                labels = labels.to(logits.device)
                # 从 logits 中取第2维的后 13(即labels.size(1)) 项, 即 logits 中与文本提示相关的部分
                logits = logits[:, -labels.size(1) :, :] # shape = [batch_size, 13, text_config.vocab_size]
                # Shift so that tokens < n predict n
                # logits[..., :-1, :] <=> logits[:, :-1, :]
                shift_logits = logits[..., :-1, :].contiguous() # shape = [batch_size, 12, text_config.vocab_size]
                # labels[..., 1:] <=> labels[:, 1:]
                shift_labels = labels[..., 1:].contiguous().to(logits.device) # shape = [batch_size, 12]

                # 定义损失函数
                loss_fct = CrossEntropyLoss(reduction="mean")
                # Flatten the tokens
                # loss_fct([36, text_config.vocab_size], [36])
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            # 序列到序列语言模型
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        # 如果无需返回字典形式的输出, 则返回元组形式的输出
        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output
        # 否则返回字典形式的输出
        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,                      # 根据 logits 和 labels, 计算得到的 loss
            logits=logits,                  # 语言模型输出中的 logits
            vision_outputs=vision_outputs,  # 视觉模型的输出
            qformer_outputs=query_outputs,  # Q-Former 的输出
            language_model_outputs=outputs, # 语言模型的输出
        )
    # 禁用梯度追踪的上下文管理器
    @torch.no_grad()
    # 定义生成方法，接受像素值、输入 ID、注意力掩码等参数，以及其他生成方法的关键字参数
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        覆盖`generate`函数, 以便能够将模型用作条件生成器.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
                将被处理的输入图像
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
                用作提示的序列, 用于生成
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
                掩码, 避免对填充词元索引执行注意力

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
            字幕 (list): 字符串列表, 长度为 batch_size * num_captions
        """
        if hasattr(self, "hf_device_map"):
            # 预处理用于`加速`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]  # 获取批处理大小
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state  # 通过视觉模型处理像素值，获取图像嵌入表示
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)  # 创建图像注意力掩码，默认全1

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 通过复制扩展查询标记，以匹配图像嵌入维度
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )  # 使用查询标记和图像嵌入作为输入，执行查询转换操作
        query_output = query_outputs.last_hidden_state  # 获取查询转换的输出

        language_model_inputs = self.language_projection(query_output)  # 使用查询转换的输出作为输入，执行语言模型投影操作
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )  # 创建语言注意力掩码，默认全1
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])  # 如果未提供输入标记，则使用特殊的起始标记
                .repeat(batch_size, 1)  # 在批次维度上复制起始标记
                .to(image_embeds.device)  # 将起始标记移动到与图像嵌入相同的设备上
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # 如果未提供注意力掩码，则创建一个与输入标记相同形状的全1张量
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)  # 将语言注意力掩码与输入标记的注意力掩码连接在一起

        # 将查询嵌入与提示嵌入连接起来
        inputs_embeds = self.get_input_embeddings()(input_ids)  # 获取输入标记的嵌入表示
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)  # 将语言模型输入嵌入与输入标记嵌入连接在一起

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )  # 使用语言模型生成序列

        return outputs  # 返回生成的序列
