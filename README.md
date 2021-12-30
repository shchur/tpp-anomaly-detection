# Detecting Anomalous Event Sequences with Temporal Point Processes
Pytorch implementation of the paper ["Detecting Anomalous Event Sequences with Temporal Point Processes"](https://papers.neurips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html), by Oleksandr Shchur, Ali Caner Turkmen, Tim Januschowski, Jan Gasthaus, and Stephan Günnemann, NeurIPS 2021.


## Installation
1. Install the dependencies
    ```
    conda env create -f environment.yml
    ```
2. Activate the conda environment
    ```
    conda activate anomaly_tpp
    ```
3. Install the package (this command must be run in the `tpp-anomaly-detection` folder)
    ```
    pip install -e .
    ```
4. Unzip the data
    ```
    unzip data.zip
    ```

## Reproducing the results from the paper
- `notebooks/spp_experiment.ipynb`: Standard Poisson process vs. other toy TPPs (Section 6.1 in the paper).
- `notebooks/multivariate_experiment.ipynb`: Multivariate TPPs inspired by real-world scenarios (Section 6.2).
- `notebooks/real_world_experiment.ipynb`: Real-world datasets (Section 6.3).


## Citation
Please cite our paper if you use the code or the datasets in your own work

```
@article{
    shchur2021detecting,
    title={Detecting Anomalous Event Sequences with Temporal Point Processes},
    author={Oleksandr Shchur and Ali Caner Turkmen and Tim Januschowski and Jan Gasthaus and and Stephan G\"{u}nemann},
    journal={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2021},
}
```
