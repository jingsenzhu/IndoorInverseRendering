**InteriorVerse dataset -- Update 2026/08**

We thank Lezhong Wang for sharing the complete copy of the dataset files, which can be accessed via this [Hugging Face](https://huggingface.co/datasets/Lez/InteriorVerse) repo.

**InteriorVerse dataset -- Update 2026/04**

We regret to inform users that the original dataset download link is no longer valid due to unexpected file loss on the cloud service. We sincerely apologize for the inconvenience this has caused.

We have made every effort to recover the dataset. While we were only able to restore part of the original data, the recovered subset has now been uploaded to a new server. The updated download links are available [here](interiorverse/Interiorverse_dataset.csv). **Note that the original way to access the data by sending an email is no longer valid.**

Please note the following details about the recovered dataset:

- The portion rendered with an 85° field of view (FOV) is largely intact.
- The portion rendered with a 120° FOV has suffered substantial data loss.

As a result of the missing data, the original train/validation/test splits are no longer valid.

<!-- **If you previously downloaded the complete dataset and still have a copy, we would greatly appreciate it if you could contact [Jingsen Zhu](mailto:zhujingsen.p32@gmail.com) to help with potential recovery.** -->

Thank you for your understanding and continued interest in our work.

**News**

- `04/12/2022` repository created
- `31/12/2022` code release for material-geometry network
- `17/01/2023` dataset release: InteriorVerse material-geometry part
- `24/02/2023` code release for lighting network
- `28/02/2023` pretrained model and testing data (object insertion) released

**TODO**

- [x] Code release for Material-Geometry network
- [x] Code release for Lighting network
- [x] Release of pretrained model
- [x] Dataset release: InteriorVerse material-geometry part
- [ ] Dataset release: InteriorVerse lighting part

# Learning-based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing

### [Project Page](https://jingsenzhu.github.io/invrend/) | [Paper](https://arxiv.org/abs/2211.03017) | [Dataset](interiorverse/Interiorverse_dataset.csv) | [Hugging Face](https://huggingface.co/datasets/Lez/InteriorVerse)

![teaser](assets/teaser.png)

This repository implements the paper "Learning-Based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing" in SIGGRAPH Asia'22. It includes training and testing code of material-geometry network (MGNet) and testing code of lighting network (LightNet).

**Also check our following work: [I<sup>2</sup>-SDF](https://github.com/jingsenzhu/i2-sdf) !**

## Pretrained Models

Pretrained models are available [here](https://1drv.ms/u/s!As0jHj7lvvu5gQTjaGZu80GDAF0j?e=3KEB65), including MGNet and LightNet.

## Citation

If you find our work is useful, please consider cite:

```
@inproceedings{zhu2022learning,
    author = {Zhu, Jingsen and Luan, Fujun and Huo, Yuchi and Lin, Zihao and Zhong, Zhihua and Xi, Dianbing and Wang, Rui and Bao, Hujun and Zheng, Jiaxiang and Tang, Rui},
    title = {Learning-Based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing},
    year = {2022},
    publisher = {ACM},
    url = {https://doi.org/10.1145/3550469.3555407},
    booktitle = {SIGGRAPH Asia 2022 Conference Papers},
    articleno = {6},
    numpages = {8}
}
```

