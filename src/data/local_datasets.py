import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path
from typing import Dict, List, Union

import lhotse
import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from torch.utils.data import Dataset
from transformers.utils import logging

from data.augmentations import RandomBackgroundNoise
from utils.general import round_nearest, get_cut_recording_id
from utils.training_args import DataArguments

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def is_ignored_segment(text):
    return bool(re.match(r"^\s*#ignore=", text or ""))


def add_timestamps(transcript, sample_len, sampling_rate=16_000, precision=0.02):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(sample_len / sampling_rate, precision):.2f}|>"}


class TS_ASR_DatasetSuperclass:
    """
        Contains all dataset-related methods that both, random and segmented datasets use.
    """

    def __init__(self,
                 cutsets,
                 text_norm=lambda x: x,
                 use_timestamps=False,
                 max_timestamp_pause=0.0,
                 model_features_subsample_factor=2,
                 dataset_weights=None,
                 feature_extractor=None,
                 global_lang_id=None,
                 load_channel_zero_only=False,
                 load_signal_sum=False,
                 musan_augment_prob=0.0,
                 musan_root=None,
                 *args,
                 **kwargs):

        self.cutsets = cutsets

        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        self.cset = reduce(lambda a, b: a + b, self.cutsets)
        self.max_timestamp_pause = max_timestamp_pause
        self.use_timestamps = use_timestamps
        self.text_norm = text_norm
        self.feature_extractor = feature_extractor
        self.model_features_subsample_factor = model_features_subsample_factor
        self.global_lang_id = global_lang_id
        self.prepare_cuts()
        self.load_channel_zero_only = load_channel_zero_only
        self.load_signal_sum = load_signal_sum
        self.musan_augment_prob = musan_augment_prob
        if self.musan_augment_prob > 0.0:
            self.musan_augment = RandomBackgroundNoise(sample_rate=16_000, noise_dir=musan_root)

    @staticmethod
    def get_number_of_speakers_from_monocut(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return len(spks)

    @staticmethod
    def get_cut_spks(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return sorted(spks)

    def get_segment_text_with_timestamps(self, segment, use_timestamps, text_norm, skip_end_token):
        start = f"<|{round_nearest(segment.start, 0.02):.2f}|>"
        end = f"<|{round_nearest(segment.end_, 0.02):.2f}|>"
        text = text_norm(segment.text_)
        if not text:
            return ""
        if skip_end_token:
            end = ""
        if use_timestamps:
            text = start + text + end
        return text

    def merge_supervisions(self, target_spk_supervision):
        from types import SimpleNamespace
        new_merged_list = []
        for supervision in sorted(target_spk_supervision, key=lambda x: x.start):
            if len(new_merged_list) == 0:
                new_merged_list.append(
                    SimpleNamespace(start=supervision.start, end_=supervision.end, text_=supervision.text))
            else:
                prev = new_merged_list[-1]
                if round(prev.end_, 2) == round(supervision.start,
                                                2) or supervision.start - prev.end_ <= self.max_timestamp_pause:
                    prev.end_ = supervision.end
                    prev.text_ = prev.text_ + " " + supervision.text
                else:
                    new_merged_list.append(
                        SimpleNamespace(start=supervision.start, end_=supervision.end, text_=supervision.text))
        return new_merged_list

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset, weight in zip(self.cutsets, self.dataset_weights):
            with ThreadPoolExecutor() as executor:
                spk_per_cut = list(executor.map(self.get_number_of_speakers_from_monocut, cutset.cuts))
            spk_per_cut = np.array(spk_per_cut) * weight
            self.to_index_mapping.append(spk_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))

    def get_stno_mask(self, cut: Cut, speaker_id: str):
        speakers = self.get_cut_spks(cut)
        speakers_to_idx = {spk: idx for idx, spk in enumerate(speakers)}

        # Build mask directly at model frame resolution to avoid a large sample-rate intermediate array.
        frame_step = self.model_features_subsample_factor * self.feature_extractor.hop_length
        n_samples = cut.num_samples
        n_samples_padded = n_samples + (-n_samples % self.feature_extractor.n_samples)
        n_frames = n_samples_padded // frame_step
        sr = cut.sampling_rate

        spk_mask = np.zeros((len(speakers), n_frames), dtype=np.float32)
        for sup in cut.supervisions:
            if sup.speaker not in speakers_to_idx:
                continue
            start_frame = int(sup.start * sr / frame_step)
            end_frame = min(int(np.ceil(sup.end * sr / frame_step)), n_frames)
            spk_mask[speakers_to_idx[sup.speaker], start_frame:end_frame] = 1.0

        if speaker_id == "-1":
            speaker_index = -1
            spk_mask = np.pad(spk_mask, ((0, 1), (0, 0)), mode='constant')
        else:
            speaker_index = speakers_to_idx[speaker_id]

        return self._create_stno_masks(spk_mask, speaker_index)

    @staticmethod
    def _create_stno_masks(spk_mask: np.ndarray, s_index: int):
        non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(axis=0)
        anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
        target_spk = spk_mask[s_index] * anyone_else
        non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = spk_mask[s_index] - target_spk
        stno_mask = np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0).T
        return stno_mask

    def get_features(self, cut: Cut):
        if self.load_channel_zero_only:
            samples = cut.recording.load_audio(channels=[0], offset=cut.start, duration=cut.duration).squeeze()
        elif self.load_signal_sum:
            samples = cut.recording.load_audio(offset=cut.start, duration=cut.duration)
        else:
            samples = cut.load_audio().squeeze()

        if self.musan_augment_prob > 0.0 and torch.rand(1).item() < self.musan_augment_prob:
            samples = self.musan_augment(samples)

        return samples, None

    @staticmethod
    def downsample_mean(arr, factor=1600):
        arr = np.array(arr, dtype=float)
        n = len(arr) // factor  # full chunks only
        arr = arr[:n * factor]  # trim to multiple of factor
        return arr.reshape(n, factor).mean(axis=1)

    def get_potentionally_parent_recording(self, cut):
        if self.parent_csets is not None:
            if get_cut_recording_id(cut) in self.parent_recording_to_id:
                return self.parent_csets[self.parent_recording_to_id[get_cut_recording_id(cut)]]
        return cut

    @staticmethod
    def mix_two_recordings(len_1, len_2, allowed_pause):
        rec2_offset = np.random.uniform(low=-len_1 - len_2 - allowed_pause, high=allowed_pause)
        # we start with rec1 followed by rec2 -> positive value means rec2 is offset by inserting pause after rec1
        # if -len1 is sampled rec1 is fully overlapped with rec2
        # if -len_1-len_2-allowed_pause is sampled first goes rec2 followed by pause and rec1
        if -rec2_offset <= len_1:
            return 0, len_1 + rec2_offset
        else:
            return -(len_1 + rec2_offset), 0

    @staticmethod
    def sample_offsets(target_duration, durations, overlap_factor, allowed_pause=2.0):
        # first we pair-wise mix other recordings
        N = len(durations)
        duration_to_mix = target_duration * overlap_factor

        shuffle_indexes = np.random.permutation(N)

        prev_rec_dur = durations[shuffle_indexes[0]]
        offsets = np.zeros(N)
        for i in range(1, N):
            other_rec_dur = durations[shuffle_indexes[i]]
            offset_1, offset_2 = TS_ASR_DatasetSuperclass.mix_two_recordings(prev_rec_dur, other_rec_dur, allowed_pause)
            offsets[:] += offset_1
            offsets[shuffle_indexes[i]] = offset_2
            prev_rec_dur = max(offset_1 + prev_rec_dur, offset_2 + other_rec_dur)

        if prev_rec_dur < duration_to_mix:
            # sample offset of others
            offset = np.random.uniform(low=0, high=target_duration - prev_rec_dur)
            return 0, offsets + offset

        mix_direction = np.random.choice([-1, 1])

        if mix_direction == 1:
            return prev_rec_dur - duration_to_mix, offsets
        else:
            return 0, offsets + (target_duration - duration_to_mix)

    def get_transcript(self, target_spk_supervisions, last_segment_unfinished):
        merged_supervisions = self.merge_supervisions(target_spk_supervisions)
        # Build parts with raw text first, track ST boundary flags
        parts = []
        for i, segment in enumerate(merged_supervisions):
            is_last = (i == len(merged_supervisions) - 1)
            raw = segment.text_ or ""
            # Skip ignored segments entirely
            if is_ignored_segment(raw):
                continue
            trailing_st = bool(re.search(r"<ST\s*/>\s*$", raw))
            leading_st = bool(re.match(r"^\s*<ST\s*/>", raw))
            # Strip leading and trailing <ST/>, collapse internal consecutive ones
            raw = re.sub(r"^\s*(<ST\s*/>\s*)+", "", raw)
            raw = re.sub(r"(\s*<ST\s*/>)+\s*$", "", raw)
            parts.append({
                "text": raw,
                "leading_st": leading_st,
                "trailing_st": trailing_st,
                "is_last": is_last,
                "segment": segment,
                "skip_end_token": is_last and last_segment_unfinished,
            })
        # Now stitch with context-aware joining
        parts = [p for p in parts if p["text"].strip()]
        stitched_texts = []
        for i, part in enumerate(parts):
            text = part["text"]
            if not text.strip():
                continue
            if i == 0:
                stitched_texts.append(text)
            else:
                prev_trailing = parts[i - 1]["trailing_st"]
                curr_leading = part["leading_st"]
                if prev_trailing or curr_leading:
                    # Continuation: join without sentence break
                    if stitched_texts:
                        prev = stitched_texts[-1].rstrip()
                        if prev and prev[-1] in '.?!;':
                            # Previous was complete sentence, treat as new sentence
                            stitched_texts[-1] = prev
                            text = text[0].upper() + text[1:] if text else text
                        elif prev and prev[-1] == ',':
                            # Already has comma, just lowercase continuation
                            text = text[0].lower() + text[1:] if text else text
                        else:
                            # Genuine mid-sentence cut, join with space (not comma)
                            stitched_texts[-1] = prev
                            text = text[0].lower() + text[1:] if text else text
                    stitched_texts.append(text)
                else:
                    # Clean sentence boundary
                    if stitched_texts:
                        prev = stitched_texts[-1].rstrip()
                        if prev and prev[-1] not in '.?!;:':
                            stitched_texts[-1] = prev + "."
                    text = text[0].upper() + text[1:] if text else text
                    stitched_texts.append(text)
        # Ensure final sentence is closed
        if stitched_texts:
            prev = stitched_texts[-1].rstrip()
            if prev and prev[-1] not in '.?!;:':
                stitched_texts[-1] = prev + "."
        # Join and apply norm (which no longer needs to handle <ST/>)
        separator = "" if self.use_timestamps else " "
        # Re-attach timestamps if needed
        final_parts = []
        seg_iter = iter([p for p in parts if p["text"].strip()])
        for text in stitched_texts:
            try:
                part = next(seg_iter)
                segment = part["segment"]
                if self.use_timestamps:
                    start = f"<|{round_nearest(segment.start, 0.02):.2f}|>"
                    end = f"<|{round_nearest(segment.end_, 0.02):.2f}|>" if not part["skip_end_token"] else ""
                    text = start + text + end
            except StopIteration:
                pass
            final_parts.append(text)
        transcription = separator.join(final_parts)
        transcription = self.text_norm(transcription)
        return transcription

    def get_transcripts(self):
        transcripts = []
        for idx in range(len(self)):
            cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
            cut = self.cset[cut_index]
            spks = self.get_cut_spks(cut)
            local_sid = (idx - self.to_index_mapping[cut_index]) % len(spks)
            speaker_id = spks[local_sid]
            last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut,
                                                                                          "per_spk_flags") else False
            target_spk_supervisions = filter(lambda x: x.speaker == speaker_id, cut.supervisions)
            transcription = self.get_transcript(target_spk_supervisions, last_segment_unfinished)
            transcripts.append(transcription)

    def cut_to_sample(self, cut: Cut, speaker_id: str, idx: int = -1, is_nested: bool = False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut, "per_spk_flags") else False
        target_spk_supervisions = filter(lambda x: x.speaker == speaker_id, cut.supervisions)
        transcription = self.get_transcript(target_spk_supervisions, last_segment_unfinished)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": transcription, "is_long_form": False}

        outputs["transcript"] = re.sub(r'\s+', ' ', transcription).strip()

        if hasattr(cut, "lang"):
            outputs["language"] = cut.lang
        elif self.global_lang_id:
            outputs["language"] = self.global_lang_id
        else:
            raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


class TS_ASR_Dataset(TS_ASR_DatasetSuperclass, Dataset):
    def __init__(self, *args, **kwargs):
        TS_ASR_DatasetSuperclass.__init__(self, *args, **kwargs)

    def __len__(self):
        return self.to_index_mapping[-1]

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        spks = self.get_cut_spks(cut)
        local_sid = (idx - self.to_index_mapping[cut_index]) % len(spks)
        sid = spks[local_sid]
        return self.cut_to_sample(cut, sid, idx)


class LhotseLongFormDataset(TS_ASR_Dataset):
    def __init__(self, cutset: CutSet,
                 references: CutSet = None, provide_gt_lang: bool = False, break_to_characters=False,
                 use_ids_as_transcripts=True, **kwargs):
        self.break_to_characters = break_to_characters
        cutset = cutset.to_eager()
        if self.break_to_characters:
            cutset = cutset.map(lambda cut: cut.map_supervisions(
                lambda supervision: supervision.transform_text(self.add_space_between_chars)))
            if references is not None:
                references = references.map(lambda cut: cut.map_supervisions(
                    lambda supervision: supervision.transform_text(self.add_space_between_chars)))

        self._references = references
        super().__init__(cutsets=[cutset], **kwargs)

        if self._references is not None:
            rids = set(get_cut_recording_id(cut) for cut in self.references)
            cids = set(get_cut_recording_id(cut) for cut in self.cset)
            if len(rids & cids) == 0:
                raise ValueError("'references' doesn't match inference cuts")  # fail immediately
            if cids != rids:
                logger.warn("'cutset' and 'references' aren't the same sets")

        self.provide_gt_lang = provide_gt_lang
        self.use_ids_as_transcripts = use_ids_as_transcripts

    @staticmethod
    def add_space_between_chars(text):
        pattern = re.compile(
            r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60\u0E00-\u0E7F])"
        )  # CJKT chars
        chars = pattern.split(text)
        chars = [ch for ch in chars if ch.strip()]
        text = " ".join(w for w in chars)
        text = re.sub(r"\s+", " ", text)
        return text

    @property
    def references(self) -> CutSet:
        """Returns the reference CutSet for evaluation.

        This property allows using separate reference and hypothesis CutSets, which is useful
        for evaluation scenarios like diarization where we want to score system outputs
        against ground truth references. If no separate references were provided during
        initialization, falls back to using the input CutSet as references.

        Returns:
            CutSet: The reference CutSet containing ground truth transcripts and speaker labels
        """
        if self._references is not None:
            return self._references
        return self.cset

    def has_reference_lang(self, rec_id):
        cut = self.references.filter(lambda x: get_cut_recording_id(x) == rec_id)[0]
        if hasattr(cut, "lang"):
            return cut.lang
        else:
            return False

    def cut_to_sample(self, cut: Cut, speaker_id, idx: int = -1, is_nested=False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True, "idx": f'{cut.id},{speaker_id}'}

        if not self.use_ids_as_transcripts:
            target_spk_supervisions = filter(lambda x: x.speaker == speaker_id, cut.supervisions)
            last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut,
                                                                                          "per_spk_flags") else False
            merged_supervisions = self.merge_supervisions(target_spk_supervisions)
            transcription = ("" if self.use_timestamps else " ").join(
                [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm,
                                                       (idx == len(
                                                           merged_supervisions) - 1) and last_segment_unfinished)
                 for idx, segment in
                 enumerate(merged_supervisions)])
            outputs["transcript"] = re.sub(r'\s+', ' ', transcription).strip()

        if self.provide_gt_lang and not is_nested:
            if hasattr(cut, "lang"):
                outputs["language"] = cut.lang
            elif self._references is not None or self.global_lang_id:
                has_reference_lang = self.has_reference_lang(get_cut_recording_id(cut)) if hasattr(cut,
                                                                                                   "recording_id") else False
                outputs["language"] = has_reference_lang or self.global_lang_id
            else:
                raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


def load_cutsets(cutset_list):
    cutsets = []
    for cut_path in cutset_list:
        cutset = lhotse.load_manifest(cut_path)
        cutsets.append(cutset)

    return cutsets


def build_datasets(cutset_paths: List[Union[str, Path]], data_args: DataArguments,
                   text_norm, container, diar_cutset_paths=None, use_ids_as_transcripts=True,
                   dataset_class=LhotseLongFormDataset):
    logger.info('Using LhotseLongFormDataset')
    if cutset_paths is None or len(cutset_paths) == 0:
        raise ValueError("'cutset_paths' is None or empty. Please provide valid 'cutset_paths' for the dataset")

    cutsets = load_cutsets(cutset_paths)

    if data_args.merge_eval_cutsets:
        cutsets = [reduce(lambda a, b: a + b, cutsets)]
        cutset_paths = ["reduced_from" + "_".join([os.path.basename(path) for path in cutset_paths])]
    if data_args.use_diar:
        if diar_cutset_paths is None or len(diar_cutset_paths) == 0:
            raise ValueError(
                "'diar_cutset_paths' is None or empty. Please provide valid 'diar_cutset_paths' for the dataset")
        if not all(Path(p).exists() for p in diar_cutset_paths):
            wrong_paths = os.linesep.join(
                [f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in diar_cutset_paths])
            raise ValueError(f"Some diar cutset paths do not exist:{os.linesep}{wrong_paths}")
        refs = cutsets
        cutsets = [CutSet.from_file(path) for path in diar_cutset_paths]
        if data_args.merge_eval_cutsets:
            cutsets = [reduce(lambda a, b: a + b, cutsets)]
    else:
        refs = [None for _ in cutsets]

    return {os.path.basename(path).removesuffix(".jsonl.gz"): dataset_class(cutset=cutset, references=ref,
                                                                            use_timestamps=data_args.use_timestamps,
                                                                            text_norm=text_norm,
                                                                            feature_extractor=container.feature_extractor,
                                                                            global_lang_id=data_args.global_lang_id,
                                                                            provide_gt_lang=data_args.provide_gt_lang,
                                                                            load_channel_zero_only=data_args.load_channel_zero_only,
                                                                            break_to_characters="break_to_chars" in path,
                                                                            use_ids_as_transcripts=use_ids_as_transcripts
                                                                            ) for cutset, ref, path in
            zip(cutsets, refs, cutset_paths)}


class TS_QA_Dataset(TS_ASR_Dataset):
    def __init__(self, *args, audio_cache_size=8, **kwargs):
        self._audio_cache = OrderedDict()
        self._audio_cache_size = audio_cache_size
        super().__init__(*args, **kwargs)

    def _get_cached_features(self, cut):
        if self.musan_augment_prob > 0:
            return self.get_features(cut)
        if cut.id not in self._audio_cache:
            if len(self._audio_cache) >= self._audio_cache_size:
                self._audio_cache.popitem(last=False)
            self._audio_cache[cut.id] = self.get_features(cut)
        return self._audio_cache[cut.id]

    @staticmethod
    def get_number_of_questions_from_monocut(cut):
        number_of_questions = 0
        for spk in cut.custom["speakers"]:
            number_of_questions += len(cut.custom["speakers"][spk])
        return number_of_questions

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset in self.cutsets:
            with ThreadPoolExecutor() as executor:
                qa_per_cut = list(executor.map(self.get_number_of_questions_from_monocut, cutset.cuts))
            qa_per_cut = np.array(qa_per_cut)
            self.to_index_mapping.append(qa_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))

    def cut_to_sample(self, cut: Cut, speaker_id: str, qa: Dict[str, str], idx: int = -1, is_nested: bool = False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self._get_cached_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "is_long_form": True}

        outputs["prompt"] = qa["prompt"]
        outputs["gt_answer"] = qa["gt_answer"]

        return outputs

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        questions = []
        for speaker in cut.custom["speakers"]:
            questions.extend(cut.custom["speakers"][speaker])

        local_sid = (idx - self.to_index_mapping[cut_index]) % len(questions)
        question = questions[local_sid]
        spk = question["speaker"]
        return self.cut_to_sample(cut, spk, question, idx)
