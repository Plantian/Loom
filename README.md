<div align="center">

# 🚀 [CVPR 2026 Findings] Loom: Diffusion-Transformer for Interleaved Generation

<a href='https://openaccess.thecvf.com/content/CVPR2026F/papers/Ye_Loom_Diffusion-Transformer_for_Interleaved_Generation_CVPRF_2026_paper.pdf'><img src='https://img.shields.io/badge/Loom-Paper-blue?logo=arxiv'></a>
<a href='https://huggingface.co/datasets/plantian/Loom_01'><img src='https://img.shields.io/badge/Loom-Huggingface Dataset-yellow?logo=huggingface'></a>

![theme](/assets/theme.jpg)

**[Mingcheng Ye](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=xMO3ISAAAAAJ)**<sup>1</sup>, **[Jiaming Liu](https://scholar.google.com/citations?hl=zh-CN&user=SmL7oMQAAAAJ&view_op=list_works&sortby=pubdate)**<sup>2</sup>, **[Yiren Song](https://scholar.google.com/citations?hl=zh-CN&user=L2YS0jgAAAAJ&view_op=list_works&sortby=pubdate)**<sup>3</sup><sup>\#</sup>

<sup>\#</sup> *Corresponding author*
<br>
<sup>1</sup> *Beijing Institute of Technology*, <sup>2</sup> *Alibaba Group*, <sup>3</sup> *National University of Singapore*

</div>

---

## 📖 Introduction

![showcase_1](/assets/image.png)

## 🏗️ Model Architecture

![Model](/assets/image_2.png)

## 💡 Motivation

A vast and challenging class of real-world scenarios demands reasoning over interleaved, mixed-modality sequences. These **N-to-M** tasks, which require models to consume and produce multiple, related inputs and outputs, include:

- 🍳 **(1) Procedural Generation**: Producing step-by-step tutorials where visual frames and textual explanations are interleaved to guide a user, such as in cooking guides or artistic workflows.
- 🧩 **(2) Compositional Reasoning**: Synthesizing a single, coherent scene from multiple, disparate visual and textual inputs, or the inverse, decomposing a scene into its constituent parts, for applications like virtual try-on.
- 🎨 **(3) Multi-Reference Generation**: Transforming a content image based on the semantic or stylistic properties of several reference images, such as in complex style transfer. Current open-source frameworks lack a unified mechanism to handle this full spectrum of multi-modal, multi-turn reasoning.

> *Note: We would like to thank the [Bagel team](https://github.com/ByteDance-Seed/Bagel) for integrating strong text and image generation capabilities into a single model, which enables Loom to be implemented elegantly at current time.*

## ✨ Key Innovation

To realize this unified approach, Loom treats text and image embeddings as sequentially composable elements within a shared latent space. We introduce a dual set of conditioning mechanisms to manage the complexity of **N-to-M** tasks. For procedural tasks, a language-planning strategy decomposes global instructions into local steps, which are associated with temporal frame embeddings and sparse historical frame sampling to maintain long-horizon coherence. For compositional and stylistic tasks, control is achieved via learnable entity tokens for structured grounding.

Our contributions are as follows:

- 🥇 **(1)** We propose Loom, a unified diffusion-transformer framework for interleaved text–image generation, supporting style transfer, compositional synthesis, and procedural tutorials within a single model.
- ⚙️ **(2)** We introduce a unified control and conditioning mechanism for N-to-M tasks, including a language-planning strategy and sparse historical frame sampling for temporal coherence, and learnable entity tokens for structured compositional grounding.
- 📊 **(3)** We curate a 50K interleaved tutorial dataset and present comprehensive experiments demonstrating Loom’s superior compositionality, temporal coherence, and text–image alignment.

---

## 🚀 Quick Start

> **⏳ Status Update:** Full training code, model weights, and the dataset pipeline are **coming soon**! We will release them once the paper is accepted. Stay tuned!

### 1. Preparation
You can clone this repository in advance to get ready for the upcoming code release:
```bash
git clone https://github.com/Plantian/Loom.git
cd Loom
```

### 2. Explore the Dataset
While waiting for the code, you can already explore our curated **50K interleaved tutorial dataset**, which is now available on Hugging Face:
- 🤗 **[Access Loom_01 Dataset on HuggingFace](https://huggingface.co/datasets/plantian/Loom_01)**

---

## 📦 Dataset Construction & Details

![dataset](/assets/image_7.png)

This is what an interleaved tutorial sample from our dataset looks like:
![dataset_show](/assets/image_10.png)

## 🖼️ More Awesome Showcases of Loom

- **StyleRef Task** 🎭
  ![style_ref](/assets/image_3.png)
- **Multi-images Ref Task** 🖼️➕🖼️
  ![multi](/assets/image_4.png)
- **Text2Interleaved Task** 📝➡️🎞️
  ![t2interleaved](/assets/image_5.png)
- **Image2Interleaved Task** 🖼️➡️🎞️
  ![i2interleaved](/assets/image_6.png)
- **More Showcases** ✨
  ![showcase_2](/assets/image_8.png)
  ![showcase_3](/assets/image_9.png)

## 📄 License

Loom is licensed under the **Apache 2.0** License.

## ✒️ Citation

If you think this project is helpful, please reference this paper:

```bibtex
@InProceedings{Ye_2026_CVPR,
    author    = {Ye, Mingcheng and Liu, Jiaming and Song, Yiren},
    title     = {Loom: Diffusion-Transformer for Interleaved Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {4582-4592}
}
```
