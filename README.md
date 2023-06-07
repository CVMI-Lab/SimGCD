# Parametric Classification for Generalized Category Discovery: A Baseline Study

This repo contains code for our paper: [Parametric Classification for Generalized Category Discovery: A Baseline Study](https://arxiv.org/abs/2211.11727).

![teaser](assets/teaser.jpg)

Generalized Category Discovery (GCD) aims to discover novel categories in unlabelled datasets using knowledge learned from labelled samples.
Previous studies argued that parametric classifiers are prone to overfitting to seen categories, and endorsed using a non-parametric classifier formed with semi-supervised $k$-means.

However, in this study, we investigate the failure of parametric classifiers, verify the effectiveness of previous design choices when high-quality supervision is available, and identify unreliable pseudo-labels as a key problem. We demonstrate that two prediction biases exist: the classifier tends to predict seen classes more often, and produces an imbalanced distribution across seen and novel categories. 
Based on these findings, we propose a simple yet effective parametric classification method that benefits from entropy regularisation, achieves state-of-the-art performance on multiple GCD benchmarks and shows strong robustness to unknown class numbers.
We hope the investigation and proposed simple framework can serve as a strong baseline to facilitate future studies in this field.

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

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

~~Then check the results starting with `Metrics with best model on test set:` in the logs.
This means the model is picked according to its performance on the test set, and then evaluated on the unlabelled instances of the train set.~~

We found picking the model according to 'Old' class performance could lead to possible over-fitting, and since 'New' class labels on the held-out validation set should be assumed unavailable, we suggest not to perform model selection, and simply use the last-epoch model.

## Results
Our results in three independent runs:

|    Dataset    	|    All   	|    Old   	|    New   	|
|:-------------:	|:--------:	|:--------:	|:--------:	|
|    CIFAR10    	| 97.1±0.0 	| 95.1±0.1 	| 98.1±0.1 	|
|    CIFAR100   	| 80.1±0.9 	| 81.2±0.4 	| 77.8±2.0 	|
|  ImageNet-100 	| 83.0±1.2 	| 93.1±0.2 	| 77.9±1.9 	|
|  ImageNet-1K  	| 57.1±0.1 	| 77.3±0.1 	| 46.9±0.2 	|
|      CUB      	| 60.3±0.1 	| 65.6±0.9 	| 57.7±0.4 	|
| Stanford Cars 	| 53.8±2.2 	| 71.9±1.7 	| 45.0±2.4 	|
| FGVC-Aircraft 	| 54.2±1.9 	| 59.1±1.2 	| 51.8±2.3 	|
|  Herbarium 19 	| 44.0±0.4 	| 58.0±0.4 	| 36.4±0.8 	|

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@article{wen2022simgcd,
  title={Parametric Classification for Generalized Category Discovery: A Baseline Study},
  author={Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2211.11727},
  year={2022}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
