# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
# `.\transformers\models\blip_2\processing_blip_2.py`
"""
BLIP-2 的处理器类。
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

从 transformers.models.blip.processing_blip.BlipProcessor 修改
class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.
    构建一个 BLIP-2 处理器, 将 BLIP 图像处理器和 OPT/T5 词元分析器封装成单个处理器.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.
    [`BlipProcessor`] 提供 [`BlipImageProcessor`] 和 [`AutoTokenizer`] 的所有功能. 更多信息请参阅 [`~BlipProcessor.__call__`] 和 [`~BlipProcessor.decode`] 的文档字符串.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
            [`BlipImageProcessor`] 的一个实例. 图像处理器是一个必需的输入.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
            [`PreTrainedTokenizer`] 的一个实例. 词元分析器是一个必需的输入.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer" # BLIP-2
    # tokenizer_class = ("BertTokenizer", "BertTokenizerFast") # BLIP-1

    def __init__(self, image_processor, tokenizer):
        # 禁用词元分析器返回 token_type_ids
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为图像处理器
        self.current_processor = self.image_processor

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.
        使用 [`BlipImageProcessor.__call__`] 方法为模型准备图像,
        使用 [`BertTokenizerFast.__call__`] 方法为模型准备文本.

        Please refer to the docstring of the above two methods for more information.
        请参阅上述两种方法的文档字符串以获取更多信息.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # 输入仅文本时
        if images is None:
            self.current_processor = self.tokenizer
            # 使用 AutoTokenizer 处理文本
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            ) # text_encoding 包含: 1) 本文输入在词表中的索引(input_ids), 2) 其注意力掩码(attention_mask)
            return text_encoding

        # 输入包含图像时
        # 使用 BlipImageProcessor 处理图像
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)
        # encoding_image_processor 包含: 1) 像素值(pixel_values)

        # 如果输入同时包含图像和文本时
        if text is not None:
            # 使用 AutoTokenizer 处理文本
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            ) # text_encoding 包含: 1) 本文输入在词表中的索引(input_ids), 2) 其注意力掩码(attention_mask)
        else:
            # 当输入仅图像时
            text_encoding = None

        # 如果 text_encoding 非空
        if text_encoding is not None:
            # 将 text_encoding 中的信息添加至 encoding_image_processor 中,
            encoding_image_processor.update(text_encoding)
            # encoding_image_processor 包含: 1) 像素值(pixel_values), 2) 本文输入在词表中的索引(input_ids), 3) 其注意力掩码(attention_mask)

        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. 
        Please refer to the docstring of this method for more information.
        此方法将其所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.batch_decode`].
        请参阅该方法的文档字符串以获取更多信息.
        根据文本索引返回本文字符, 适用于批量处理
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. 
        Please refer to the docstring of this method for more information.
        此方法将其所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.decode`].
        请参阅该方法的文档字符串以获取更多信息.
        根据文本索引返回本文字符, 适用于单个处理
        """
        # 调用PreTrainedTokenizer的decode方法进行解码，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # 返回 tokenizer 输出中的 keys
        tokenizer_input_names = self.tokenizer.model_input_names
        # 返回 image_processor 输出中的 keys
        image_processor_input_names = self.image_processor.model_input_names
        # 返回全部 keys
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))