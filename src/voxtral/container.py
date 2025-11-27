import torch
from peft import LoraConfig, get_peft_model
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizerFast
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from typing import Optional, Union
from src.models.containers import supports_flash_attention
from src.models.dicow.encoder import DiCoWEncoder
from src.models.dicow.config import DiCoWConfig
from src.models.dicow.modeling_dicow import DiCoWForConditionalGeneration
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


class VoxtralForConditionalGenerationCustom(VoxtralForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        dicow_config = DiCoWConfig.from_pretrained("/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/dicow_large_v3")
        dicow_config.ctc_weight=0.0
        dicow_config.additional_self_attention_layer = False
        dicow_config.pre_ctc_sub_sample = False
        self.audio_tower = DiCoWEncoder(dicow_config)
        self.post_init()

    def get_audio_embeds(self, input_features: torch.FloatTensor, stno_mask: torch.FloatTensor):
        """
        This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector.
        Args:
            input_features (`torch.FloatTensor`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

        Returns:
            `torch.FloatTensor`:
                The audio embeddings.
        """
        audio_outputs = self.audio_tower(input_features, stno_mask=stno_mask)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            stno_mask=None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> repo_id = "mistralai/Voxtral-Mini-3B-2507"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

        >>> conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(conversation)
        >>> inputs = inputs.to(device, dtype=torch.bfloat16)

        >>> outputs = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_embeds = self.get_audio_embeds(input_features, stno_mask)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = input_ids == self.config.audio_token_id
            inputs_embeds[audio_token_mask] = audio_embeds

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

def get_dixtral(repo_id, device):
    dicow = DiCoWForConditionalGeneration.from_pretrained(
        "/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/dicow_large_v3", device_map=device)
    model = VoxtralForConditionalGenerationCustom.from_pretrained(repo_id,
                                                                  device_map=device)
    model.audio_tower.load_state_dict(dicow.get_encoder().state_dict(), strict=False)
    del dicow
    return model


class VoxtralContainer:
    def __init__(self, use_flash_attention=False, params_to_keep_frozen_keywords=None, remove_timestamps_from_ctc=False,
                 model_args=None, data_args=None, use_fddt=False, use_lora=False, repo_id="mistralai/Voxtral-Mini-3B-2507"):
        self.model_type = model_args.whisper_model

        self.model = get_dixtral(repo_id, device='cuda')
        self.processor = AutoProcessor.from_pretrained(repo_id)

        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.model.set_tokenizer(self.processor.tokenizer)
        self.model.config.forced_decoder_ids = None

        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|down_proj|up_proj).*",
                lora_dropout=0.0,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)

        if params_to_keep_frozen_keywords is not None:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    continue
                for keyword in params_to_keep_frozen_keywords:
                    if keyword in name:
                        param.requires_grad = False
                        break
                else:
                    param.requires_grad = True

    def freeze_except(self, prefixes_to_preheat):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for prefix in prefixes_to_preheat:
                if name.startswith(prefix):
                    param.requires_grad = True
