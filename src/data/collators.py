import base64
import io
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import soundfile as sf
import torch
from transformers.utils import logging

from data.augmentations import SpecAug

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _insert_soft_prompts(
    p_ids: list, p_att: list, soft_ids: list, soft_att: list
) -> tuple:
    """Insert soft prompt tokens just before the final trigger token."""
    return p_ids[:-1] + soft_ids + p_ids[-1:], p_att[:-1] + soft_att + p_att[-1:]


def _build_sequence(
    p_ids: list, p_att: list, t_ids: list, eos_id: int,
    in_longform: bool, prep_for_generate: bool,
) -> tuple:
    """Concatenate prompt and answer tokens; in generate mode keep prompt-only input."""
    if not in_longform or not prep_for_generate:
        ids = p_ids + t_ids + [eos_id]
        attn = p_att + [1] * (len(t_ids) + 1)
    else:
        ids = p_ids
        attn = p_att
    lab = [-100] * len(p_ids) + t_ids + [eos_id]
    return ids, attn, lab


def _pad_and_tensorize(seqs: list, fill: int) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    return torch.tensor(
        [x + [fill] * (max_len - len(x)) for x in seqs], dtype=torch.long
    )


def _assemble_stno_mask(inputs: list) -> torch.Tensor:
    return torch.stack(
        [chunk for s in inputs for chunk in s["stno_mask"].split(1500, dim=0)]
    ).transpose(1, 2)


def numpy_audio_to_base64_wav(audio: np.ndarray, sampling_rate: int = 16_000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Collators ──────────────────────────────────────────────────────────────────

@dataclass
class DataCollator:
    processor: object
    max_length: int
    model_id: str
    conv_subsample_factor: int = 2
    prep_for_generate: bool = True
    num_soft_prompts: int = 8
    soft_prompt_token_id: int = 23
    stno_gaussian_noise_var: float = None
    stno_gaussian_noise_prob: float = None
    stno_segment_augment_prob: float = 0.3
    stno_segment_change_prob: float = 0.1
    stno_min_segment_length: int = 5
    stno_max_segment_length: int = 50
    spec_aug_prob: float = 0.3

    def __post_init__(self):
        self.spec_aug = SpecAug(
            apply_time_warp=True, time_warp_window=5, time_warp_mode="bicubic",
            apply_freq_mask=True, freq_mask_width_range=[0, 27], num_freq_mask=2,
            apply_time_mask=True, time_mask_width_ratio_range=[0, 0.05], num_time_mask=5,
        )

    @staticmethod
    def add_gaussian_noise_and_rescale(prob_mask, variance=0.05, fraction=0.5):
        B, C, T = prob_mask.shape
        num_noisy = int(B * fraction)
        if num_noisy == 0:
            return prob_mask
        idx = torch.randperm(B)[:num_noisy]
        noisy = prob_mask.clone()
        noisy[idx] += torch.randn((num_noisy, C, T), device=prob_mask.device) * (variance ** 0.5)
        noisy[idx] -= torch.clamp(noisy[idx].amin(dim=1, keepdim=True), max=0)
        noisy[idx] /= noisy[idx].sum(dim=1, keepdim=True)
        return noisy

    @staticmethod
    def soft_segment_augmentation(stno_mask, change_prob=0.2, min_seg_len=5, max_seg_len=20):
        B, C, T = stno_mask.shape
        augmented = stno_mask.clone()
        for b in range(B):
            pos = 0
            while pos < T:
                seg_len = torch.randint(min_seg_len, max_seg_len + 1, (1,)).item()
                end = min(pos + seg_len, T)
                if torch.rand(1).item() < change_prob:
                    seg = augmented[b, :, pos:end]
                    dominant = seg.mean(dim=1).argmax().item()
                    others = [c for c in range(C) if c != dominant]
                    if others:
                        tgt = others[torch.randint(0, len(others), (1,)).item()]
                        target = torch.zeros_like(seg)
                        target[tgt] = 1.0
                        alpha = torch.rand(1).item()
                        mixed = (1 - alpha) * seg + alpha * target
                        augmented[b, :, pos:end] = mixed / mixed.sum(dim=0, keepdim=True)
                pos = end
        return augmented

    def __call__(
        self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]], nested=False
    ) -> Dict[str, torch.Tensor]:
        longform = [s["is_long_form"] for s in inputs]
        if len(set(longform)) != 1:
            raise ValueError("Mixed longform/shortform batch")
        in_longform = longform[0]

        prompt = self.processor.apply_transcription_request(
            language="en", sampling_rate=16_000,
            audio=[s["input_features"] for s in inputs],
            model_id=self.model_id, format=["WAV"] * len(inputs),
        )
        passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        prompt_ids, prompt_attn = prompt["input_ids"], prompt["attention_mask"]

        tok = self.processor.tokenizer
        text_ids_list = tok(
            [s["transcript"] for s in inputs],
            add_special_tokens=False, padding=False, truncation=True,
            max_length=2048, return_tensors=None,
        )["input_ids"]

        soft_ids = [self.soft_prompt_token_id] * self.num_soft_prompts
        soft_att = [1] * self.num_soft_prompts
        all_ids, all_attn, all_labs = [], [], []
        for i in range(len(inputs)):
            p_ids, p_att = _insert_soft_prompts(
                prompt_ids[i].tolist(), prompt_attn[i].tolist(), soft_ids, soft_att,
            )
            ids, attn, lab = _build_sequence(
                p_ids, p_att, text_ids_list[i], tok.eos_token_id, in_longform, self.prep_for_generate,
            )
            all_ids.append(ids)
            all_attn.append(attn)
            all_labs.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        batch = {
            "input_ids": _pad_and_tensorize(all_ids, pad_id),
            "attention_mask": _pad_and_tensorize(all_attn, 0),
            "labels": _pad_and_tensorize(all_labs, -100),
        }
        if "idx" in inputs[0]:
            batch["idxs"] = tok(
                [s["idx"] for s in inputs], padding="longest",
                max_length=self.max_length, return_tensors="pt",
            )["input_ids"]
        batch.update(passthrough)
        batch["stno_mask"] = _assemble_stno_mask(inputs)

        if not in_longform and not nested:
            if self.stno_segment_augment_prob and torch.rand(1).item() < self.stno_segment_augment_prob:
                batch["stno_mask"] = self.soft_segment_augmentation(
                    batch["stno_mask"],
                    change_prob=self.stno_segment_change_prob,
                    min_seg_len=self.stno_min_segment_length,
                    max_seg_len=self.stno_max_segment_length,
                )
            if self.stno_gaussian_noise_var and self.stno_gaussian_noise_var > 0:
                batch["stno_mask"] = self.add_gaussian_noise_and_rescale(
                    batch["stno_mask"], self.stno_gaussian_noise_var, self.stno_gaussian_noise_prob,
                )
            if torch.rand(1).item() < self.spec_aug_prob:
                n_feat = batch["input_features"].shape[1]
                aug_in = torch.cat(
                    [batch["input_features"],
                     batch["stno_mask"].repeat_interleave(self.conv_subsample_factor, dim=2)],
                    dim=1,
                ).permute(0, 2, 1)
                aug_out = self.spec_aug(aug_in)[0].permute(0, 2, 1)
                batch["input_features"] = aug_out[:, :n_feat, :]
                stno_raw = aug_out[:, n_feat:, :]
                batch["stno_mask"] = torch.stack(
                    stno_raw.split(self.conv_subsample_factor, dim=-1)
                ).mean(dim=-1).permute(1, 2, 0)

        return batch


@dataclass
class DataCollatorQA:
    processor: object
    max_length: int
    model_id: str
    conv_subsample_factor: int = 2
    prep_for_generate: bool = True
    num_soft_prompts: int = 0
    soft_prompt_token_id: int = 23
    sampling_rate: int = 16_000

    def __call__(
        self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        conversations = []
        for s in inputs:
            audio = s["input_features"]
            if not isinstance(audio, list):
                audio = [audio]
            content = [
                {"type": "audio", "base64": numpy_audio_to_base64_wav(a, self.sampling_rate)}
                for a in audio
            ] + [{"type": "text", "text": s["prompt"]}]
            conversations.append([{"role": "user", "content": content}])

        prompt = self.processor.apply_chat_template(conversations)
        passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        prompt_ids, prompt_attn = prompt["input_ids"], prompt["attention_mask"]

        tok = self.processor.tokenizer
        has_answers = "gt_answer" in inputs[0]
        if has_answers:
            text_ids_list = tok(
                [s["gt_answer"] for s in inputs],
                add_special_tokens=False, padding=False, truncation=True,
                max_length=2048, return_tensors=None,
            )["input_ids"]
        else:
            text_ids_list = [[] for _ in range(prompt_ids.size(0))]

        soft_ids = [self.soft_prompt_token_id] * self.num_soft_prompts
        soft_att = [1] * self.num_soft_prompts
        all_ids, all_attn, all_labs = [], [], []
        for i, sample in enumerate(inputs):
            p_ids, p_att = _insert_soft_prompts(
                prompt_ids[i].tolist(), prompt_attn[i].tolist(), soft_ids, soft_att,
            )
            ids, attn, lab = _build_sequence(
                p_ids, p_att, text_ids_list[i], tok.eos_token_id,
                sample.get("is_long_form", True), self.prep_for_generate,
            )
            all_ids.append(ids)
            all_attn.append(attn)
            all_labs.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        batch = {
            "input_ids": _pad_and_tensorize(all_ids, pad_id),
            "attention_mask": _pad_and_tensorize(all_attn, 0),
            "labels": _pad_and_tensorize(all_labs, -100),
        }
        batch.update(passthrough)
        if "stno_mask" in inputs[0]:
            batch["stno_mask"] = _assemble_stno_mask(inputs)

        return batch
