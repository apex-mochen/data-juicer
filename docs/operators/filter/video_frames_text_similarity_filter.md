# video_frames_text_similarity_filter

Filter to keep samples based on the similarity between video frame images and text within a specific range.

This operator uses a Hugging Face CLIP model to compute the similarity between video frames and associated text. It keeps samples where the computed similarity scores fall within a specified range. The operator supports different frame sampling methods, including 'all_keyframes' and 'uniform', and allows for horizontal and vertical flipping of the frames. The similarity score is reduced using one of three modes: 'avg', 'max', or 'min'. The operator also supports two strategies for keeping samples: 'any' (keep if any video meets the condition) or 'all' (keep only if all videos meet the condition). The key metric is stored in the 'video_frames_text_similarity' field.

基于视频帧图像与文本之间的相似度在特定范围内的过滤保留样本。

该算子使用 Hugging Face CLIP 模型来计算视频帧与关联文本之间的相似度。它保留计算出的相似度得分在指定范围内的样本。该算子支持不同的帧采样方法，包括 'all_keyframes' 和 'uniform'，并且允许对帧进行水平和垂直翻转。相似度得分通过三种模式之一进行归约：'avg'、'max' 或 'min'。该算子还支持两种保留样本的策略：'any'（如果有任何视频满足条件则保留）或 'all'（只有当所有视频都满足条件时才保留）。关键指标存储在 'video_frames_text_similarity' 字段中。

Type 算子类型: **filter**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` |  | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute the similarity between frame image and text. It's kind of language-related. For example, for Chinese datasets, ChineseCLIP might be a better choice. |
| `trust_remote_code` |  | `False` | whether to trust the remote code of HF models. |
| `min_score` | <class 'float'> | `0.1` | the min similarity to keep samples. |
| `max_score` | <class 'float'> | `1.0` | the max similarity to keep samples. |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame images from the videos. Should be one of ["all_keyframes", "uniform"]. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. Default: "all_keyframes". |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video. Only works when frame_sampling_method is "uniform". If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. |
| `horizontal_flip` | <class 'bool'> | `False` | flip frame image horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | flip frame image vertically (top to bottom). |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all videos. 'any': keep this sample if any videos meet the condition. 'all': keep this sample only if all videos meet the condition. |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode when one text corresponds to multiple video frame images in a chunk. 'avg': Take the average of multiple values 'max': Take the max of multiple values 'min': Take the min of multiple values |
| `max_batch_frames` | positive integer | `32` | maximum number of frames sent to CLIP in one forward pass with `batch_mode: true`. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

Enable batched inference with the sample-level `batch_mode` and `batch_size` options. A chunk with more than `max_batch_frames` frames is scored in several forwards and reduced once. Floating-point scores can differ slightly from single-sample inference, especially near a filter boundary.

使用 `batch_mode` 和 `batch_size` 启用批量打分。`max_batch_frames` 限制单次模型前向的总帧数。超出上限的文本片段会分次打分，最后统一聚合。

```yaml
process:
  - video_frames_text_similarity_filter:
      batch_mode: true
      batch_size: 16
      max_batch_frames: 32
      frame_sampling_method: uniform
      frame_num: 3
```

## 📊 Effect demonstration 效果演示
### test_default_filter
```python
VideoFramesTextSimilarityFilter('openai/clip-vit-base-patch32')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。&lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;__dj__video&gt; &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
The operator computes the similarity between video frames and text, keeping all samples since no specific score range is set. The target list contains all original videos as none are filtered out.
算子计算视频帧与文本之间的相似度，由于没有设置特定的分数范围，因此保留所有样本。目标列表包含所有原始视频，因为没有任何被过滤掉。

### test_any
```python
VideoFramesTextSimilarityFilter('openai/clip-vit-base-patch32', min_score=0.26, max_score=0.3, frame_sampling_method='uniform', frame_num=2, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。&lt;|__dj__eoc|&gt; &lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。&lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 2 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。&lt;|__dj__eoc|&gt; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;__dj__video&gt; &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 2 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。&lt;|__dj__eoc|&gt; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;__dj__video&gt; &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### ✨ explanation 解释
The operator filters to keep samples where any of the multiple videos meets the condition of having a similarity score within the specified range (0.26 to 0.3). The target list includes samples with at least one video meeting the criteria, while others are excluded because they do not meet the 'any' condition for the given range.
算子过滤并保留多个视频中任意一个满足指定范围（0.26到0.3）内相似度得分条件的样本。目标列表包括至少有一个视频满足条件的样本，而其他不满足给定范围内'任意'条件的样本则被排除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_frames_text_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_frames_text_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)
