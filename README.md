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
python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=../models/codebert \
--model_name_or_path=../models/codebert \
--do_train --train_data_file=../dataset/good-3/train.jsonl \
--eval_data_file=../dataset/good-3/valid.jsonl \
--test_data_file=../dataset/good-3/test.jsonl \
--epoch 8 \
--block_size 400 \
--train_batch_size 12 \
--eval_batch_size 12 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123456 \
--cnn_size 128 \
--filter_size 4 \
--d_size 128

```


