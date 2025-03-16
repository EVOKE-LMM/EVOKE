
<h1 align="center"> <a href="https://arxiv.org/abs/2406.11194">When Large Multimodal Models Confront Evolving Knowledge: Challenges and Pathways</a></h1>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2502.19870-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2502.19870) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EVOKE-blue)](https://huggingface.co/datasets/kailinjiang/MMKE-Bench-dataset)  [![paperwithcode](https://img.shields.io/badge/PWC-EVOKE-blue?logo=paperswithcode)](https://paperswithcode.com/paper/mmke-bench-a-multimodal-editing-benchmark-for)  [![code](https://img.shields.io/badge/Code-EVOKE-blue?logo=github)](https://github.com/EVOKE-LMM/EVOKE) [![website](https://img.shields.io/badge/Website-EVOKE-orange?logo=homepage)](https://mmke-bench-iclr.github.io/) [![airchina](https://img.shields.io/badge/数源AI-EVOKE-red?logo=airchina)](https://mp.weixin.qq.com/s/iN826lITi5Xyz-3GnrdVIQ)



</h5>



## Table of Contents

- [🤗EVOKE](#evoke)
- [🛠️Requirements and Installation](#️requirements-and-installation)
- [🌟Retrieval](#retrieval)
- [💥Training](#training)
- [🤖Evaluation](#evaluation)



## 🤗EVOKE

<div align="center">   <img src="fig/motivation.jpg" width="700px"> </div>

To evaluate evolving knowledge injection in LMMs, we propose a pipeline to automatically collect evolving knowledge, constructing the <u><b>EVO</b></u>lving <u><b>K</b></u>nowledg<u><b>E</b></u> <b>(EVOKE)</b> benchmark.

<div align="center">   <img src="fig/data_construction.png" width="700px"> </div>

You can download data 🤗 [Huggingface Dataset](https://huggingface.co/datasets/kailinjiang/EVOKE). And the expected structure of files is:



```text
EVOKE
|-- json/jsonl
|   |-- evoke_injection_data.json
|   |-- evoke_evaluation_data.jsonl
|-- imgs
|   |-- injection
|   |   |-- evoke_entity_injection_imgs.zip
|   |   |-- evoke_news_injection_imgs.zip
|   |-- evaluation
|   |   |-- evoke_news_evaluation_imgs.zip
|   |   |-- evoke_entity_evaluation_imgs.zip
```


<div align="center">   <img src="fig/data_display.png" width="700px"> </div>

## 🛠️Requirements and Installation

```text
Please refer to the code repository

https://github.com/haotian-liu/LLaVA

https://github.com/QwenLM/Qwen-VL

https://github.com/TIGER-AI-Lab/UniIR
```

## 🌟Retrieval

```shell
For image_only:

python retrieval/retrieval_image_only.py

For text_only:
python retrieval/retrieval_text_only.py

For UniIR:
step1
python retrieval/UniIR/src/common/mbeir_retriever.py

get retrieval/UniIR/retrieval_results/CLIP_SF/Large/Instruct/UniRAG/run_files/mbeir_new_self_union_pool_test_k10_run.txt

step2
python retrieval/retrieval_UniIR.py
```







## 💥Training


```shell
Please refer to the code repository

https://github.com/haotian-liu/LLaVA

https://github.com/QwenLM/Qwen-VL
```



## 🤖Evaluation

```shell
step1
python evaluation/eval_acc_f1.py

step2
python evaluation/all_type_score.py
```
