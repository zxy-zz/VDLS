# VDLS: A Vulnerability Detection Approach Based on Execution Paths with Local Structures and Semantics

## 项目简介

## Dtasets
To investigate the effectiveness of VDLS, we adopt two vulnerability datasets from these paper:
-[FFMPeg+Qemu [1]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF]
-[Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF]

## Requirements
```bash
- [Python3.9]
- [torch (>=1.13.1)]
- [numpy(>=1.26.4)]
- [pandas (>=2.2.2)]
- [scikit-learn (>=1.5.1)]
- [tqdm]

## Joern tool
We used the joern tool to generate the CFG corresponding to the code, and the corresponding version can be selected between Joern-Cli1.1.260 and Joern-Cli1.1.700.
- [This document describes the Joern tool ：https://github.com/joernio/joern/releases?page=59 ] 
- [Download path of the Joern tool ： https://github.com/joernio/joern/releases?page=59]  

## 安装
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/yourproject.git
