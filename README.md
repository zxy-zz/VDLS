# VDLS: A Vulnerability Detection Approach Based on Execution Paths with Local Structures and Semantics

## Dataset
To investigate the effectiveness of VDLS, we adopt two vulnerability datasets from these paper:
* FFMPeg+Qemu [1]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF]
* Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF]

## Requirements
* Python3.9]
* torch (>=1.13.1)
* numpy(>=1.26.4)
* pandas (>=2.2.2)
* scikit-learn (>=1.5.1)
* tqdm



## Joern tool
We used the joern tool to generate the CFG corresponding to the code.So if using the newer versions of Joern to generate code structure graph, the model may have a different performance compared with the results we reported in the paper. The corresponding version can be selected between Joern-Cli1.1.260 and Joern-Cli1.1.700.
* This document describes the Joern tool ：https://github.com/joernio/joern/releases?page=59 
* Download path of the Joern tool ： https://github.com/joernio/joern/releases?page=59


## Experimental process
#### Path generation
* Step 1: We first run ```joern_graph_gen.py``` to generate a CFG diagram of the corresponding code.
     first generate .bin files
     ```
     python joern_graph_gen.py  -i ./data/sard/Vul -o ./data/sard/bins/Vul -t parse
     python joern_graph_gen.py  -i ./data/sard/No-Vul -o ./data/sard/bins/No-Vul -t parse
     ```
     
     then generate pdgs (.dot files)
     ```
     python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/cfgs/Vul -t export -r cfg
     python joern_graph_gen.py  -i ./data/sard/bins/No-Vul -o ./data/sard/cfgs/No-Vul -t export -r cfg
     ```
* Step 2: Next, run ```path_gen.py``` to extract the relevant execution path of CFG for the corresponding code.
* Step 3: Use ```data_split.py``` for data set partitioning operations.

#### Running the model
```shell


```


After parsing the functions with joern, the code for graph construction and simplification is under the ```data_processing\``` folder. ```data_processing\word2vec.py``` is used to train word2vec model. We also provide our trained word2vec model [here](https://zenodo.org/record/7333062#.Y3c5SHZByUk).

## Running the model
The model implementation code is under the ``` AMPLE_code\``` folder. The model can be runned from ```AMPLE_code\main.py```.

## Attention weight
We provide all the attention weights learned by our proposed model AMPLE for the test samples. Each dataset corresponds to a json file under ```attention weight\``` folder.

## Experiment results
### PR-AUC & MCC && G-measure && T-test
<center>Table 1. Experiment results for Reveal and AMPLE. "*" denotes sttistical significance in comparision to Reveal in terms of accuracy and F1 score (i.e., two-sided t-test with p-value < 0.05).</center>

## References
[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[4] M. Fu and C. Tantithamthavorn. 2022. Linevul: A transformer-based line-level vulnerability prediction. In The 2022 International Conference on Mining Software Repositories (MSR). IEEE.


