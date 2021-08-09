### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- using poetry to manage env stuffs
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
poetry install
 ```
It is highly recommended using poetry to manage python env, for usage, please refer to https://python-poetry.org/.

 Next, navigate to any file and run it.   
 ```bash
# spawn your env's shell 
poetry shell

mv configs/train.yaml.template configs/train.yaml
vi configs/train.yaml # do some modification

# run training
python main.py train configs/train.yaml.template

# run testing with two gpus
python main.py test configs/train.yaml {your_ckpt_path} {your_hparams_path} --gpus 2 --accelerator ddp
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
