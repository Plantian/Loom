# Loom: Diffusion-Transformer for Interleaved Generation

![theme](/assets/theme.jpg)

## Introduction

![showcase_1](/assets/image.png)

## Model Architecture

![Model](/assets/image_2.png)

## Motivation

A vast and challenging class of real-world scenarios demands reasoning over interleaved, mixed-modality sequences. These **N-to-M** tasks, which require models to consume and produce multiple, related inputs and outputs, include:

- (1) **Procedural Generation**: Producing step-by-step tutorials where visual frames and textual explanations are interleaved to guide a user, such as in cooking guides or artistic workflows.
- (2) **Compositional Reasoning**: Synthesizing a single, coherent scene from multiple, disparate visual and textual inputs, or the inverse, decomposing a scene into its constituent parts, for applications like virtual try-on.
- (3) **Multi-Reference Generation**: Transforming a content image based on the semantic or stylistic properties of several reference images, such as in complex style transfer. Current open-source frameworks lack a unified mechanism to handle this full spectrum of multi-modal, multi-turn reasoning.

## Key Innovation

To realize this unified approach, Loom treats text and image embeddings as sequentially composable elements within a shared latent space. We introduce a dual set of conditioning mechanisms to manage the complexity of **N-to-M** tasks. For procedural tasks, a language-planning strategy decomposes global instructions into local steps, which are associated with temporal frame embeddings and sparse historical frame sampling to maintain long-horizon coherence. For compositional and stylistic tasks, control is achieved via learnable entity tokens for structured grounding.

Our contributions are as follows:

- (1) We propose Loom, a unified diffusion-transformer framework for interleaved text–image generation, supporting style transfer, compositional synthesis, and procedural tutorials within a single model.

- (2) We introduce a unified control and conditioning mechanism for N-to-M tasks, including a language-planning strategy and sparse historical frame sampling for temporal coherence, and learnable entity tokens for structured compositional grounding.

- (3) We curate a 50K interleaved tutorial dataset and present comprehensive experiments demonstrating Loom’s superior compositionality, temporal coherence, and text–image alignment.

## Dataset

### Dataset Construction

![dataset](/assets/image_7.png)

### Dataset Release

Full dataset is coming soon(we will release on HuggingFace).
And this is the interleaved tutorial sample belike:
![dataset_show](/assets/image_10.png)

## More Awesome Showcases of Loom

- StyleRef Task![style_ref](/assets/image_3.png)
- Multi-images Ref Task![multi](/assets/image_4.png)
- Text2Interleaved Task![t2interleaved](/assets/image_5.png)
- Image2Interleaved Task![i2interleaved](/assets/image_6.png)
- More![showcase_2](/assets/image_8.png)
  ![showcase_3](/assets/image_9.png)

## License

Loom is licensed under the Apache 2.0.

## Citation

If you think this project is helpful, please reference this paper：

```bib
@misc{ye2025loomdiffusiontransformerinterleavedgeneration,
      title={Loom: Diffusion-Transformer for Interleaved Generation}, 
      author={Mingcheng Ye and Jiaming Liu and Yiren Song},
      year={2025},
      eprint={2512.18254},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.18254}, 
}
```
