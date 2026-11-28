import numpy as np
import torch
from PIL import ImageOps
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    close_video,
    extract_key_frames,
    extract_video_frames_uniformly,
    load_data_with_context,
    load_video,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

OP_NAME = "video_frames_text_similarity_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoFramesTextSimilarityFilter(Filter):
    """Filter to keep samples based on the similarity between video frame images and text
    within a specific range.

    This operator uses a Hugging Face CLIP model to compute the similarity between video
    frames and associated text. It keeps samples where the computed similarity scores fall
    within a specified range. The operator supports different frame sampling methods,
    including 'all_keyframes' and 'uniform', and allows for horizontal and vertical flipping
    of the frames. The similarity score is reduced using one of three modes: 'avg', 'max',
    or 'min'. The operator also supports two strategies for keeping samples: 'any' (keep if
    any video meets the condition) or 'all' (keep only if all videos meet the condition).
    The key metric is stored in the 'video_frames_text_similarity' field."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_clip="openai/clip-vit-base-patch32",
        trust_remote_code=False,
        min_score: float = 0.1,
        max_score: float = 1.0,
        frame_sampling_method: str = "all_keyframes",
        frame_num: PositiveInt = 3,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        any_or_all: str = "any",
        reduce_mode: str = "avg",
        *args,
        max_batch_frames: PositiveInt = 32,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the similarity between frame image and text. It's kind of
            language-related. For example, for Chinese datasets, ChineseCLIP
            might be a better choice.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param min_score: the min similarity to keep samples.
        :param max_score: the max similarity to keep samples.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos.
            Should be one of ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param horizontal_flip: flip frame image horizontally (left to right).
        :param vertical_flip: flip frame image vertically (top to bottom).
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param reduce_mode: reduce mode when one text corresponds to
            multiple video frame images in a chunk.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param max_batch_frames: maximum number of frames per CLIP forward when
            batch_mode is enabled.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["memory"] = "1500MB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method "
                f"[{frame_sampling_method}] is not supported. "
                f'Can only be one of ["all_keyframes", "uniform"].'
            )
        if reduce_mode not in ["avg", "max", "min"]:
            raise ValueError(
                f"Reduce mode [{reduce_mode}] is not supported. " f'Can only be one of ["avg", "max", "min"].'
            )
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_clip, trust_remote_code=trust_remote_code
        )
        self.reduce_mode = reduce_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        if max_batch_frames < 1:
            raise ValueError("max_batch_frames must be positive")
        self.max_batch_frames = max_batch_frames

        self.sampled_frames_key_suffix = f"-{frame_sampling_method}" + (
            "" if frame_sampling_method == "all_keyframes" else f"-{frame_num}"
        )

    def _video_text_chunks(self, sample, context):
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)
        offset = 0
        try:
            for chunk in sample[self.text_key].split(SpecialTokens.eoc):
                count = chunk.count(SpecialTokens.video)
                if not count:
                    continue
                images = []
                for video_key in loaded_video_keys[offset : offset + count]:
                    video = videos[video_key]
                    sampled_frames_key = video_key + self.sampled_frames_key_suffix
                    if context and sampled_frames_key in sample[Fields.context]:
                        frames = sample[Fields.context][sampled_frames_key]
                    else:
                        if self.frame_sampling_method == "all_keyframes":
                            frames = extract_key_frames(video)
                        else:
                            frames = extract_video_frames_uniformly(video, self.frame_num)
                        if context:
                            sample[Fields.context][sampled_frames_key] = frames

                    for frame in frames:
                        image = frame.to_image()
                        if self.horizontal_flip:
                            image = ImageOps.mirror(image)
                        if self.vertical_flip:
                            image = ImageOps.flip(image)
                        images.append(image)
                offset += count
                yield remove_special_tokens(chunk), images
        finally:
            if not context:
                for video in videos.values():
                    close_video(video)

    def _clip_logits(self, texts, images, rank):
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.text_config.max_position_embeddings,
            padding=True,
        ).to(model.device)
        return model(**inputs).logits_per_text

    def _reduce_scores(self, scores):
        if self.reduce_mode == "avg":
            return float(scores.mean())
        if self.reduce_mode == "max":
            return float(scores.max())
        return float(scores.min())

    def compute_stats_single(self, sample, rank=None, context=False):
        key = StatsKeys.video_frames_text_similarity
        stats = sample[Fields.stats]
        if stats.get(key) is not None:
            return sample
        if not sample.get(self.video_key):
            stats[key] = np.array([], dtype=np.float64)
            return sample

        stats[key] = [
            self._reduce_scores(self._clip_logits(text, images, rank) / 100.0) if images else 0.0
            for text, images in self._video_text_chunks(sample, context)
        ]
        return sample

    @torch.inference_mode()
    def _score_frame_batch(self, frame_batch, rank):
        images = [image for _, frames, _ in frame_batch for image in frames]
        logits = self._clip_logits([text for text, _, _ in frame_batch], images, rank)
        sizes = [len(frames) for _, frames, _ in frame_batch]
        # Keep only the frame range owned by each text from the pairwise logits.
        matched = [block[i] for i, block in enumerate(logits.split(sizes, dim=1))]
        scores = (torch.cat(matched) / 100.0).cpu()
        for (_, _, chunk_scores), part in zip(frame_batch, scores.split(sizes), strict=True):
            chunk_scores.append(part)

    def compute_stats_batched(self, samples, rank=None, context=False):
        if not self.batch_mode:
            return super().compute_stats_batched(samples, rank=rank, context=context)

        key = StatsKeys.video_frames_text_similarity
        pending = []
        pending_frames = 0
        rows_to_reduce = []
        for i, stats in enumerate(samples[Fields.stats]):
            if stats.get(key) is not None:
                continue
            sample = {column: values[i] for column, values in samples.items()}
            if not sample.get(self.video_key):
                stats[key] = np.array([], dtype=np.float64)
                continue
            row_chunks = []
            rows_to_reduce.append((stats, row_chunks))
            for text, images in self._video_text_chunks(sample, context):
                chunk_scores = []
                row_chunks.append(chunk_scores)
                for start in range(0, len(images), self.max_batch_frames):
                    frames = images[start : start + self.max_batch_frames]
                    if pending_frames + len(frames) > self.max_batch_frames:
                        self._score_frame_batch(pending, rank)
                        pending.clear()
                        pending_frames = 0
                    pending.append((text, frames, chunk_scores))
                    pending_frames += len(frames)
        if pending:
            self._score_frame_batch(pending, rank)
        for stats, row_chunks in rows_to_reduce:
            stats[key] = [self._reduce_scores(torch.cat(parts)) if parts else 0.0 for parts in row_chunks]
        return samples

    def process_single(self, sample, rank=None):
        similarity = sample[Fields.stats][StatsKeys.video_frames_text_similarity]
        if len(similarity) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(sim_value, self.min_score, self.max_score) for sim_value in similarity]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
