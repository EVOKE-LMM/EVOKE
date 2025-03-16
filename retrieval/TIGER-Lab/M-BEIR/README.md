---
license: "mit"
pretty_name: "M-BEIR"
task_categories:
- text-retrieval
- text-to-image
- image-to-text
- visual-question-answering
language:
- "en"
configs:
- config_name: query
  data_files:
  - split: train
    path: "query/train/*.jsonl"
  - split: union_train
    path: "query/union_train/*.jsonl"
  - split: val
    path: "query/val/*.jsonl"
  - split: test
    path: "query/test/*.jsonl"
- config_name: cand_pool
  data_files:
  - split: mbeir_local
    path: "cand_pool/local/*.jsonl"
  - split: mbeir_global
    path: "cand_pool/global/*.jsonl"
---

### **UniIR: Training and Benchmarking Universal Multimodal Information Retrievers** (ECCV 2024)
[**🌐 Homepage**](https://tiger-ai-lab.github.io/UniIR/) | [**🤗 Model(UniIR Checkpoints)**](https://huggingface.co/TIGER-Lab/UniIR) | [**🤗 Paper**](https://huggingface.co/papers/2311.17136) | [**📖 arXiv**](https://arxiv.org/pdf/2311.17136.pdf)  | [**GitHub**](https://github.com/TIGER-AI-Lab/UniIR)

<a href="#install-git-lfs" style="color: red;">How to download the M-BEIR Dataset</a>

## 🔔News

- **🔥[2023-12-21]: Our M-BEIR Benchmark is now available for use.**

## **Dataset Summary**

**M-BEIR**, the **M**ultimodal **BE**nchmark for **I**nstructed **R**etrieval, is a comprehensive large-scale retrieval benchmark designed to train and evaluate unified multimodal retrieval models (**UniIR models**).
The M-BEIR benchmark comprises eight multimodal retrieval tasks and ten datasets from a variety of domains and sources. 
Each task is accompanied by human-authored instructions, encompassing 1.5 million queries and a pool of 5.6 million retrieval candidates in total.


## **Dataset Structure Overview**
The M-BEIR dataset is structured into five primary components: Query Data, Candidate Pool, Instructions, Qrels, and Images.

### Query Data

Below is the directory structure for the query data:
```
query/
│
├── train/
│   ├── mbeir_cirr_train.jsonl
│   ├── mbeir_edis_train.jsonl
│   ...
├── union_train/
│   └── mbeir_union_up_train.jsonl
├── val/
│   ├── mbeir_visualnews_task0_val.jsonl
│   ├── mbeir_visualnews_task3_val.jsonl
│   ...
└── test/
    ├── mbeir_visualnews_task0_test.jsonl
    ├── mbeir_visualnews_task3_test.jsonl
    ...
```
`train`: Contains all the training data from 8 different datasets formatted in the M-BEIR style.

`mbeir_union_up_train.jsonl`: This file is the default training data for in-batch contrastive training specifically designed for UniIR models. 
It aggregates all the data from the train directory and datasets with relatively smaller sizes have been upsampled to balance the training process.

`val`: Contains separate files for validation queries, organized by task.

`test`: Contains separate files for test queries, organized by task.

Every M-BEIR query instance has at least one positive candidate data and possibly no negative candidate data
Each line in a Query Data file represents a unique query. The structure of each query JSON object is as follows::
```json
{
  "qid": "A unique identifier formatted as {dataset_id}:{query_id}",
  "query_txt": "The text component of the query",
  "query_img_path": "The file path to the associated query image",
  "query_modality": "The modality type of the query (text, image or text,image)",
  "query_src_content": "Additional content from the original dataset, presented as a string by json.dumps()",
  "pos_cand_list": [
    {
      "did": "A unique identifier formatted as {dataset_id}:{doc_id}"
    }
    // ... more positive candidates
  ],
  "neg_cand_list": [
    {
      "did": "A unique identifier formatted as {dataset_id}:{doc_id}"
    }
    // ... more negative candidates
  ]
}
```

### Candidate Pool
The Candidate Pool contains potential matching documents for the queries. 

#### M-BEIR_5.6M
Within the global directory, the default retrieval setting requires models to retrieve positive candidates from a heterogeneous pool encompassing various modalities and domains. 
The M-BEIR's global candidate pool, comprising 5.6 million candidates, includes the retrieval corpus from all tasks and datasets.

#### M-BEIR_local
Within the local directory, we provide dataset-task-specific pool as M-BEIR_local. Dataset-task-specific pool contains homogeneous candidates that originate from by the original dataset.

Below is the directory structure for the candidate pool:
```
cand_pool/
│
├── global/
│   ├── mbeir_union_val_cand_pool.jsonl
│   └──mbeir_union_test_cand_pool.jsonl
│
└── local/
    ├── mbeir_visualnews_task0_cand_pool.jsonl
    ├── mbeir_visualnews_task3_cand_pool.jsonl
    ...
```
The structure of each candidate JSON object in cand_pool file is as follows::
```json
{
  "did": "A unique identifier for the document, formatted as {dataset_id}:{doc_id}",
  "txt": "The text content of the candidate document",
  "img_path": "The file path to the candidate document's image",
  "modality": "The modality type of the candidate (e.g., text, image or text,image)",
  "src_content": "Additional content from the original dataset, presented as a string by json.dumps()"
}
```

### Instructions
`query_instructions.tsv` contains human-authorized instructions within the UniIR framework. Each task is accompanied by four human-authored instructions. For detailed usage, please refer to [**GitHub Repo**](https://github.com/TIGER-AI-Lab/UniIR).

### Qrels
Within the `qrels` directory, you will find qrels for both the validation and test sets. These files serve the purpose of evaluating UniIR models.  For detailed information, please refer to [**GitHub Repo**](https://github.com/TIGER-AI-Lab/UniIR).

## **How to Use**
### Downloading the M-BEIR Dataset
<a name="install-git-lfs"></a>
#### Step 1: Install Git Large File Storage (LFS)
Before you begin, ensure that **Git LFS** is installed on your system. Git LFS is essential for handling large data files. If you do not have Git LFS installed, follow these steps:

Download and install Git LFS from the official website.
After installation, run the following command in your terminal to initialize Git LFS:
```
git lfs install
```

#### Step 2: Clone the M-BEIR Dataset Repository
Once Git LFS is set up, you can clone the M-BEIR repo from the current Page. Open your terminal and execute the following command:
```
git clone https://huggingface.co/datasets/TIGER-Lab/M-BEIR
```
Please note that the M-BEIR dataset is quite large, and downloading it can take several hours, depending on your internet connection. 
During this time, your terminal may not show much activity. The terminal might appear stuck, but if there's no error message, the download process is still ongoing.

### Decompressing M-BEIR Images
After downloading, you will need to decompress the image files. Follow these steps in your terminal:
```bash
# Navigate to the M-BEIR directory
cd path/to/M-BEIR

# Combine the split tar.gz files into one
sh -c 'cat mbeir_images.tar.gz.part-00 mbeir_images.tar.gz.part-01 mbeir_images.tar.gz.part-02 mbeir_images.tar.gz.part-03 > mbeir_images.tar.gz'

# Extract the images from the tar.gz file
tar -xzf mbeir_images.tar.gz
```
Now, you are ready to use the M-BEIR benchmark.

### Dataloader and Evaluation Pipeline
We offer a dedicated dataloader and evaluation pipeline for the M-BEIR benchmark. Please refer to [**GitHub Repo**](https://github.com/TIGER-AI-Lab/UniIR) for detailed information.

## **Citation**
Please cite our paper if you use our data, model or code. 

```
@article{wei2023uniir,
  title={UniIR: Training and Benchmarking Universal Multimodal Information Retrievers},
  author={Wei, Cong and Chen, Yang and Chen, Haonan and Hu, Hexiang and Zhang, Ge and Fu, Jie and Ritter, Alan and Chen, Wenhu},
  journal={arXiv preprint arXiv:2311.17136},
  year={2023}
}
```
