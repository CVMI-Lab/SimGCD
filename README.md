# SimGCD: A Simple Parametric Classification Baseline for Generalized Category Discovery

This repo contains code for our paper: [A Simple Parametric Classification Baseline for Generalized Category Discovery](https://arxiv.org/abs/2211.11727).

![teaser](assets/teaser.jpg)

Generalized category discovery (GCD) is a problem setting where the goal is to discover novel categories within an unlabelled dataset using the knowledge learned from a set of labelled samples.
Recent works in GCD argue that a non-parametric classifier formed using semi-supervised $k$-means can outperform strong baselines which use parametric classifiers as it can alleviate the over-fitting to seen categories in the labelled set.

In this paper, we revisit the reason that makes previous parametric classifiers fail to recognise new classes for GCD.
By investigating the design choices of parametric classifiers from the perspective of model architecture, representation learning, and classifier learning, we conclude that the less discriminative representations and unreliable pseudo-labelling strategy are key factors that make parametric classifiers lag behind non-parametric ones. 
Motivated by our investigation, we present a simple yet effective parametric classification baseline that outperforms the previous best methods by a large margin on multiple popular GCD benchmarks.
We hope the investigations and the simple baseline can serve as a cornerstone to facilitate future studies.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

Then check the results starting with `Metrics with best model on test set:` in the logs.
This means the model is picked according to its performance on the test set, and then evaluated on the unlabelled instances of the train set.

## Results
Our results in three independent runs:

|    Dataset    	|    All   	|    Old   	|    New   	|
|:-------------:	|:--------:	|:--------:	|:--------:	|
|    CIFAR10    	| 93.2±0.4 	| 82.0±1.2 	| 98.9±0.0 	|
|    CIFAR100   	| 78.1±0.8 	| 77.6±1.5 	| 78.0±2.5 	|
|  ImageNet-100 	| 82.4±0.9 	| 90.7±0.6 	| 78.3±1.2 	|
|      CUB      	| 60.3±0.1 	| 65.6±0.9 	| 57.7±0.4 	|
| Stanford Cars 	| 46.8±1.8 	| 64.9±1.3 	| 38.0±2.1 	|
| FGVC-Aircraft 	| 48.8±2.2 	| 51.0±2.2 	| 47.8±2.7 	|
|  Herbarium 19 	| 43.3±0.3 	| 57.9±0.5 	| 35.3±0.2 	|

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@article{wen2022simgcd,
  title={A Simple Parametric Classification Baseline for Generalized Category Discovery},
  author={Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2211.11727},
  year={2022}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
