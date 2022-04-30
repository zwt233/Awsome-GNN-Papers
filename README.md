# Reading List on GNN Models

Papers about GNNs. Topics include scalable GNNs, knowledge distillation for graphs, AutoML on  GNNs, etc. If you feel there are papers with related topics missing or any mistake in this list, do not hesitate to let us know (via issues or pull requests). 

The **AutoML on Graph** part refers to [[awesome-auto-graph-learning]](https://github.com/THUMNLab/awesome-auto-graph-learning) and completes its code links.

The **Graph Data Augmentation** part refers to [[graph-data-augmentation-papers]](https://github.com/zhao-tong/graph-data-augmentation-papers) and replaces some of its links.

---

## Contents

- [Data]

- [Model]

- [System]

- [Application]


- [Scalable-GNN](#Scalable-GNN)
   - [Linear Model](#Linear-Model)
   - [Sampling](#Sampling)
   - [Distributed Training](#Distributed-Training)
- [Knowledge Distillation on Graphs](#KD)
- [AutoML on Graphs](#AutoML)
   - [Survey](#AutoML-survey)
   - [Tool](#AutoML-Tool)
   - [Graph Neural Architecture Search](#GNAS)
   - [Graph Hyper-Parameter Optimization](#GHPO)
- [Graph Data Augmentation](#GDA)
   - [Node-level tasks](#nlt)
   - [Graph-level tasks](#glt)
   - [Edge-level tasks](#elt)
   - [Graph data augmentation with self-supervised learning objectives](ssl)
 - [Graph Structure Learning](#gsl)
   - [Survey](#gsl-survey)
   - [Representive GSL methods](#gsl-models)

<!-- - [Papers](#papers)
  - [Recuurent Graph Neural Networks](#rgnn)
  - [Convolutional Graph Neural Networks](#cgnn)
  - [Graph Autoencoders](#gae)
  	  - [Network Embedding](#ne)
  	  - [Graph Generation](#gg)
  - [Spatial-Temporal Graph Neural Networks](#stgnn)
  - [Application](#application)
     - [Computer Vision](#cv)
     - [Natural Language Processing](#nlp)
     - [Internet](#web)
     - [Recommender Systems](#rec)
     - [Healthcare](#health)
     - [Chemistry](#chemistry)
     - [Physics](#physics)
     - [Others](#others)
- [Library](#library) -->

<a name="Scalable-GNN" />

# Scalable-GNN

<a name="Linear-Model" />

## Linear Model

1. **Simplifying Graph Convolutional Networks** [ICML 2019] [[paper]](https://arxiv.org/abs/1902.07153) [[code]](https://github.com/Tiiiger/SGC)
2. **Scalable Graph Neural Networks via Bidirectional Propagation** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2010.15421) [[code]](https://github.com/chennnM/GBP)
3. **SIGN: Scalable Inception Graph Neural Networks** [ICML 2020] [[paper]](https://arxiv.org/abs/2004.11198) [[code]](https://github.com/twitter-research/sign)
4. **Simple Spectral Graph Convolution** [ICLR 2021] [[paper]](https://openreview.net/forum?id=CYO5T-YjWZV) [[code]](https://github.com/allenhaozhu/SSGC)
5. **Node Dependent Local Smoothing for Scalable Graph Learning** [NeurIPS 2021] [[paper]](https://arxiv.org/abs/2110.14377) [[code]](https://github.com/zwt233/NDLS)
6. **Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.09376) [[code]](https://github.com/skepsun/SAGN_with_SLE)
7. **Graph Attention Multi-Layer Perceptron** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2108.10097) [[code]](https://github.com/zwt233/GAMLP)
7. **NAFS: A Simple yet Tough-to-Beat Baseline for Graph Representation Learning** [OpenReview 2022] [[paper]](https://openreview.net/forum?id=dHJtoaE3yRP) [[code]](https://openreview.net/attachment?id=dHJtoaE3yRP&name=supplementary_material)

<a name="Sampling" />

## Sampling

### Node-wise sampling

1. **Inductive Representation Learning on Large Graphs** [NIPS 2017] [[paper]](https://arxiv.org/abs/1706.02216) [[code]](https://github.com/twjiang/graphSAGE-pytorch)
2. **Scaling Graph Neural Networks with Approximate PageRank** [KDD 2020] [[paper]](https://arxiv.org/abs/2007.01570) [[code]](https://github.com/TUM-DAML/pprgo_pytorch)
3. **Stochastic Training of Graph Convolutional Networks with Variance Reduction** [ICML 2018] [[paper]](https://arxiv.org/abs/1710.10568) [[code]](https://github.com/thu-ml/stochastic_gcn)
4. **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings** [ICML 2021] [[paper]](https://arxiv.org/abs/2106.05609) [[code]](https://github.com/rusty1s/pyg_autoscale)
5. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems** [KDD 2018] [[paper]](https://arxiv.org/abs/1806.01973)

### Layer-wise sampling

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling** [ICLR 2018]  [[paper]](https://arxiv.org/abs/1801.10247)[[code]](https://github.com/matenure/FastGCN)
2. **Accelerating Large Scale Real-Time GNN Inference using Channel Pruning** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2105.04528) [[code]](https://github.com/tedzhouhk/GCNP)
3. **Adaptive Sampling Towards Fast Graph Representation Learning** [NeurIPS 2018] [[paper]](https://arxiv.org/abs/1809.05343) [[code_pytorch]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/_deprecated/adaptive_sampling) [[code_tentsor_flow]](https://github.com/huangwb/AS-GCN)

### Graph-wise sampling

1. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks** [KDD 2019] [[paper]](https://arxiv.org/abs/1905.07953) [[code]](https://github.com/google-research/google-research/tree/master/cluster_gcn)
2. **GraphSAINT: Graph Sampling Based Inductive Learning Method** [ICLR 2020] [[paper]](https://arxiv.org/abs/1907.04931) [[code]](https://github.com/GraphSAINT/GraphSAINT)
3. **Large-Scale Learnable Graph Convolutional Networks** [KDD 2018] [[paper]](https://dl.acm.org/doi/abs/10.1145/3219819.3219947) [[code]](https://github.com/divelab/lgcn)

<a name="Distributed-Training" />

## Distributed Training

1. **DistGNN: Scalable Distributed Training for Large-Scale Graph Neural Networks** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.06700) 
2. **Towards Efficient Large-Scale Graph Neural Network Computing** [Arxiv 2018] [[paper]](https://arxiv.org/abs/1810.08403)
3. **Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2111.06483) [[code]](https://github.com/IntelLabs/SAR)

<a name="KD" />

# Knowledge Distillation on Graphs

1. **Distilling Knowledge from Graph Convolutional Networks** [CVPR 2020] [[paper]](https://arxiv.org/abs/2003.10477) [[code]](https://github.com/ihollywhy/DistillGCN.PyTorch)
2. **Graph-Free Knowledge Distillation for Graph Neural Networks** [IJCAI 2021] [[paper]](https://arxiv.org/pdf/2105.07519.pdf) [[code]](https://github.com/Xiang-Deng-DL/GFKD)
3. **ROD: Reception-aware Online Distillation for Sparse Graphs** [KDD 2021] [[paper]](https://arxiv.org/abs/2107.11789) [[code]](https://github.com/zwt233/ROD)
4. **Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework** [WWW 2021] [[paper]](https://arxiv.org/abs/2103.02885) [[code]](https://github.com/BUPT-GAMMA/CPF)
5. **Graph Representation Learning via Multi-task Knowledge Distillation** [Arxiv 2019] [[paper]](https://arxiv.org/abs/1911.05700) 
6. **Knowledge Distillation: A Survey** [International Journal of Computer Vision] [[paper]](https://arxiv.org/pdf/2006.05525.pdf)

---

<a name="AutoML" />

# AutoML on Graphs

<a name="AutoML-survey" />

## Survey

1. **Automated Machine Learning on Graphs: A Survey** [IJCAI 2021] [[paper]](https://arxiv.org/abs/2103.00742)

<a name="AutoML-Tool" />

## Tool

1. **AutoGL: A Library for Automated Graph Learning** [ICLR 2021 GTRL workshop] [[paper]](https://openreview.net/pdf?id=0yHwpLeInDn) [[code]](https://github.com/THUMNLab/AutoGL) [[homepage]](https://mn.cs.tsinghua.edu.cn/AutoGL)

<a name="GNAS" />

## Graph Neural Architecture Search

### 2022

1. **PaSca a Graph Neural Architecture Search System under the Scalable Paradigm** [WWW 2022] [[paper]](https://arxiv.org/abs/2203.00638) [[code]](https://github.com/PKU-DAIR/SGL)
2. **Designing the Topology of Graph Neural Networks A Novel Feature Fusion Perspective** [WWW 2022] [[paper]](https://arxiv.org/abs/2112.14531) 
3. **AutoHEnsGNN Winning Solution to AutoGraph Challenge for KDD Cup 2020** [ICDE 2022] [[paper]](https://arxiv.org/abs/2111.12952) [[code]](https://github.com/aister2020/KDDCUP_2020_AutoGraph_1st_Place)
4. **Auto-GNAS: A Parallel Graph Neural Architecture Search Framework** [TPDS 2022] [[paper]](https://ieeexplore.ieee.org/document/9714826) [[code]](https://github.com/AutoMachine0/Auto-GNAS)
5. **Profiling the Design Space for Graph Neural Networks based Collaborative Filtering** [WSDM 2022] [[paper]](http://www.shichuan.org/doc/125.pdf) [[code]](https://github.com/BUPT-GAMMA/Design-Space-for-GNN-based-CF)

### 2021

1. **Graph Differentiable Architecture Search with Structure Learning** [NeurIPS 2021] [[paper]](https://openreview.net/forum?id=kSv_AMdehh3) [[code]](https://github.com/THUMNLab/AutoGL)
2. **AutoGEL: An Automated Graph Neural Network with Explicit Link Information** [NeurIPS 2021] [[paper]](https://openreview.net/forum?id=PftCCiHVQP) [[code]](https://github.com/zwangeo/AutoGEL)
3. **Heterogeneous Graph Neural Architecture Search** [ICDM 2021] [[paper]](https://ieeexplore.ieee.org/document/9679011) 
4. **Automated Graph Representation Learning for Node Classification** [IJCNN 2021] [[paper]](https://ieeexplore.ieee.org/document/9533811) 
5. **ALGNN Auto-Designed Lightweight Graph Neural Network** [PRICAI 2021] [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-89188-6_37) 
6. **Pooling Architecture Search for Graph Classification** [CIKM 2021] [[paper]](https://arxiv.org/abs/2108.10587) [[code]](https://github.com/AutoML-Research/PAS)
7. **DiffMG Differentiable Meta Graph Search for Heterogeneous Graph Neural Networks** [KDD 2021] [[paper]](https://arxiv.org/abs/2010.03250) [[code]](https://github.com/AutoML-4Paradigm/DiffMG)
8. **Learn Layer-wise Connections in Graph Neural Networks** [KDD 2021 DLG Workshop] [[paper]](https://arxiv.org/abs/2112.13585) 
9. **AutoAttend Automated Attention Representation Search** [ICML 2021] [[paper]](http://proceedings.mlr.press/v139/guan21a/guan21a.pdf) [[code]](https://github.com/THUMNLab/AutoAttend)
10. **GraphPAS Parallel Architecture Search for Graph Neural Networks** [SIGIR 2021] [[paper]](https://dl.acm.org/doi/abs/10.1145/3404835.3463007)
11. **Rethinking Graph Neural Network Search from Message-passing** [CVPR 2021] [[paper]](https://arxiv.org/abs/2103.14282) [[code]](https://github.com/phython96/GNAS-MP)
12. **Fitness Landscape Analysis of Graph Neural Network Architecture Search Spaces** [GECCO 2021] [[paper]](https://dl.acm.org/doi/10.1145/3449639.3459318) [[code]](https://github.com/mhnnunes/fla_nas_gnn)
13. **Learned low precision graph neural networks** [EuroSys 2021 EuroMLSys workshop] [[paper]](https://arxiv.org/abs/2009.09232) 
14. **Autostg: Neural architecture search for predictions of spatio-temporal graphs** [WWW 2021] [[paper]](https://zhangjunbo.org/pdf/2021_WWW_AutoSTG.pdf) [[code]](https://github.com/panzheyi/AutoSTG)
15. **Search to aggregate neighborhood for graph neural network** [ICDE 2021] [[paper]](https://arxiv.org/abs/2104.06608) [[code]](https://github.com/AutoML-4Paradigm/SANE)
16. **One-shot graph neural architecture search with dynamic search space** [AAAI 2021] [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-3441.LiY.pdf) 
17. **Search For Deep Graph Neural Networks** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2109.10047)
18. **G-CoS GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency** [ICCAD 2021] [[paper]](https://arxiv.org/abs/2109.08983) 
19. **Edge-featured Graph Neural Architecture Search** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2109.01356)
20. **FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.04141) [[code]](https://github.com/1173710224/FL-AGCNS)

### 2020

1. **Graph Neural Architecture Search** [IJCAI 2020] [[paper]](https://www.ijcai.org/proceedings/2020/195) [[code]](https://github.com/GraphNAS/GraphNAS)
2. **Design space for graph neural networks** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2011.08843) [[code]](https://github.com/snap-stanford/GraphGym)
3. **Autograph: Automated graph neural network** [ICONIP 2020] [[paper]](https://arxiv.org/abs/2011.11288) 
4. **Graph neural network architecture search for molecular property prediction** [BigData 2020] [[paper]](https://arxiv.org/abs/2008.12187) [[code]](https://github.com/0oshowero0/GEMS)
5. **Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network** [CIKM 2020] [[paper]](https://arxiv.org/abs/2102.10550) [[code]](https://github.com/0oshowero0/GEMS)
6. **Simplifying architecture search for graph neural network** [CIKM 2020 CSSA workshop] [[paper]](https://arxiv.org/abs/2008.11652) [[code]](https://github.com/AutoML-4Paradigm/SNAG)
7. **Neural architecture search in graph neural networks** [BRACIS 2020] [[paper]](https://arxiv.org/abs/2008.00077) [[code]](https://github.com/mhnnunes/nas_gnn)
8. **SGAS: Sequential Greedy Architecture Search** [CVPR 2020] [[paper]](https://arxiv.org/abs/1912.00195) [[code]](https://github.com/lightaime/sgas)
9. **Learning graph convolutional network for skeleton-based human action recognition by neural searching** [AAAI 2020] [[paper]](https://arxiv.org/abs/1911.04131) [[code]](https://github.com/xiaoiker/GCN-NAS)
10. **Efficient graph neural architecture search** [OpenReview 2020] [[paper]](https://openreview.net/forum?id=IjIzIOkK2D6) 
11. **Evolutionary architecture search for graph neural networks** [Arxiv 2020] [[paper]](https://arxiv.org/abs/2009.10199) [[code]](https://github.com/IRES-FAU/Evolutionary-Architecture-Search-for-Graph-Neural-Networks)
12. **Probabilistic dual network architecture search on graphs** [Arxiv 2020] [[paper]](https://arxiv.org/abs/2003.09676) 

### 2019

1. **Auto-GNN: Neural Architecture Search of Graph Neural Networks** [Arxiv 2019] [[paper]](https://arxiv.org/abs/1909.03184) 

<a name="GHPO" />

## Graph Hyper-Parameter Optimization

### 2021

1. **Explainable Automated Graph Representation Learning with Hyperparameter Importance** [ICML 2021] [[paper]](http://proceedings.mlr.press/v139/wang21f/wang21f.pdf)
2. **Automated Graph Learning via Population Based Self-Tuning GCN** [SIGIR 2021] [[paper]](https://arxiv.org/abs/2107.04713)
3. **Automatic Graph Learning with Evolutionary Algorithms: An Experimental Study** [PRICAI 2021] [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-89188-6_38)
4. **Which Hyperparameters to Optimise? An Investigation of Evolutionary Hyperparameter Optimisation in Graph Neural Network For Molecular Property Prediction** [GECCO 2021] [[paper]](https://arxiv.org/abs/2104.06046)
5. **ASFGNN Automated separated-federated graph neural network** [P2PNA 2021] [[paper]](https://arxiv.org/abs/2011.03248)
6. **A novel genetic algorithm with hierarchical evaluation strategy for hyperparameter optimisation of graph neural networks** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2101.09300)
7. **Jitune: Just-in-time hyperparameter tuning for network embedding algorithms** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2101.06427)

### 2020

1. **Autonomous graph mining algorithm search with best speed/accuracy trade-off** [ICDM 2020] [[paper]](https://arxiv.org/abs/2011.14925) [[code]](https://github.com/minjiyoon/ICDM20-AutoGM)

### 2019

1. **AutoNE: Hyperparameter optimization for massive network embedding** [KDD 2019] [[paper]](http://pengcui.thumedialab.com/papers/AutoNE.pdf) [[code]](https://github.com/minjiyoon/ICDM20-AutoGM)

---


<a name="GDA" />

# Graph Data Augmentation

<a name="nlt" />

## Node-level tasks

### 2022

1. **Robust Optimization as Data Augmentation for Large-scale Graphs** [CVPR2022] [[paper]](https://arxiv.org/abs/2010.09891) [[code]](https://github.com/devnkong/FLAG)

### 2021

1. **Local Augmentation for Graph Neural Networks** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2109.03856)
2. **Training Robust Graph Neural Networks with Topology Adaptive Edge Dropping** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2106.02892)
3. **FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.14210) [[code]](https://github.com/ispamm/FairDrop)
4. **Topological Regularization for Graph Neural Networks Augmentation** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.02478)
5. **Semi-Supervised and Self-Supervised Classification with Multi-View Graph Neural Networks** [CIKM 2021] [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482477)
6. **Metropolis-Hastings Data Augmentation for Graph Neural Networks** [NeurIPS 2021] [[paper]](https://arxiv.org/abs/2203.14082) 
7. **Action Sequence Augmentation for Early Graph-based Anomaly Detection** [CIKM 2021] [[paper]](https://arxiv.org/abs/2010.10016) [[code]](https://github.com/DM2-ND/Eland)
8. **Data Augmentation for Graph Neural Networks** [AAAI 2021] [[paper]](https://arxiv.org/abs/2006.06830) [[code]](https://github.com/zhao-tong/GAug)
9. **Automated Graph Representation Learning for Node Classification** [IJCNN 2021] [[paper]](https://ieeexplore.ieee.org/document/9533811/) 
10. **Mixup for Node and Graph Classification** [The WebConf 2021] [[paper]](https://bhooi.github.io/papers/mixup_web21.pdf) [[code]](https://github.com/vanoracai/MixupForGraph)
11. **Heterogeneous Graph Neural Network via Attribute Completion** [The Web Conf 2021] [[paper]](https://yangliang.github.io/pdf/www21.pdf) 
12. **FLAG: Adversarial Data Augmentation for Graph Neural Networks** [OpenReview 2021] [[paper]](https://openreview.net/forum?id=mj7WsaHYxj)

### 2020

1. **GraphMix: Improved Training of GNNs for Semi-Supervised Learning** [Arxiv 2020] [[paper]](https://arxiv.org/abs/1909.11715) [[code]](https://github.com/anon777000/GraphMix)
2. **Robust Graph Representation Learning via Neural Sparsification** [ICML 2020] [[paper]](https://proceedings.mlr.press/v119/zheng20d.html)
3. **DropEdge: Towards Deep Graph Convolutional Networks on Node Classification** [ICLR 2020] [[paper]](https://arxiv.org/abs/1907.10903) [[code]](https://github.com/DropEdge/DropEdge)
4. **Graph Structure Learning for Robust Graph Neural Networks** [KDD 2020] [[paper]](https://arxiv.org/abs/2005.10203) [[code]](https://github.com/ChandlerBang/Pro-GNN)
5. **Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View** [AAAI 2020] [[paper]](https://arxiv.org/abs/1909.03211) 

### 2019

1. **Diffusion Improves Graph Learning** [NeurIPS 2019] [[paper]](https://arxiv.org/abs/1911.05485) [[code]](https://github.com/klicperajo/gdc)


<a name="glt" />

## Graph-level tasks

### 2022

1. **Graph Augmentation Learning** [IW3C2] [[paper]](https://arxiv.org/abs/2203.09020)
2. **Automated Data Augmentations for Graph Classification** [Arxiv 2022] [[paper]](https://arxiv.org/abs/2202.13248)
3. **GAMS: Graph Augmentation with Module Swapping** [Arxiv 2022] [[paper]](https://www.scitepress.org/Papers/2022/108224/108224.pdf)
4. **Graph Transplant: Node Saliency-Guided Graph Mixup with Local Structure Preservation** [AAAI 2022] [[paper]](https://arxiv.org/abs/2111.05639)

### 2021

1. **ifMixup: Towards Intrusion-Free Graph Mixup for Graph Classification** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2110.09344)
2. **Mixup for Node and Graph Classification** [TheWebConf 2021] [[paper]](https://dl.acm.org/doi/abs/10.1145/3442381.3449796) [[code]](https://github.com/vanoracai/MixupForGraph)
3. **MoCL: Contrastive Learning on Molecular Graphs with Multi-level Domain Knowledge** [KDD 2021] [[paper]](https://arxiv.org/abs/2106.04509) [[code]](https://github.com/illidanlab/MoCL-DK)
4. **FLAG: Adversarial Data Augmentation for Graph Neural Networks** [OpenReview 2021] [[paper]](https://openreview.net/forum?id=mj7WsaHYxj)

### 2020

1. **GraphCrop: Subgraph Cropping for Graph Classification** [Arxiv 2020] [[paper]](https://arxiv.org/abs/2009.10564)
2. **M-Evolve: Structural-Mapping-Based Data Augmentation for Graph Classification** [CIKM 2020, IEEE TNSE 2021] [[paper CIKM]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412086) [[paper_TNSE]](https://arxiv.org/pdf/2007.05700.pdf)

<a name="elt" />

## Edge-level tasks

1. **Counterfactual Graph Learning for Link Prediction** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2106.02172) [[code]](https://github.com/DM2-ND/CFLP)
2. **Adaptive Data Augmentation on Temporal Graphs** [NeurIPS 2021] [[paper]](https://proceedings.neurips.cc/paper/2021/hash/0b0b0994d12ad343511adfbfc364256e-Abstract.html) 
3. **FLAG: Adversarial Data Augmentation for Graph Neural Networks** [OpenReview 2021] [[paper]](https://openreview.net/forum?id=mj7WsaHYxj)

<a name="ssl" />

## Graph data augmentation with self-supervised learning objectives

### Contrastive learning

#### 2022

1. **Learning Graph Augmentations to Learn Graph Representations** [OpenReview 2022] [[paper]](https://openreview.net/forum?id=hNgDQPe8Uj) [[code]](https://github.com/kavehhassani/lg2ar)
2. **Fair Node Representation Learning via Adaptive Data Augmentation** [Arxiv 2022] [[paper]](https://arxiv.org/abs/2201.08549)
3. **Large-Scale Representation Learning on Graphs via Bootstrapping** [ICLR 2022] [[paper]](https://arxiv.org/abs/2102.06514) [[code]](https://github.com/Namkyeong/BGRL_Pytorch)
4. **Augmentations in Graph Contrastive Learning: Current Methodological Flaws & Towards Better Practices** [TheWebConf 2022] [[paper]](https://arxiv.org/abs/2111.03220)

#### 2021

1. **Contrastive Self-supervised Sequential Recommendation with Robust Augmentation** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2108.06479) [[paper]](https://github.com/YChen1993/CoSeRec)
2. **Collaborative Graph Contrastive Learning: Data Augmentation Composition May Not be Necessary for Graph Representation Learning** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2111.03262)
3. **Molecular Graph Contrastive Learning with Parameterized Explainable Augmentations** [BIBM 2021] [[paper]](https://www.biorxiv.org/content/10.1101/2021.12.03.471150v1)
4. **Jointly Learnable Data Augmentations for Self-Supervised GNNs** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2108.10420) [[code]](https://github.com/zekarias-tilahun/graph-surgeon)
5. **InfoGCL: Information-Aware Graph Contrastive Learning** [NeurIPS 2021] [[paper]](https://openreview.net/forum?id=519VBzfEaKW) 
6. **Adversarial Graph Augmentation to Improve Graph Contrastive Learning** [NeurIPS 2021] [[paper]](https://openreview.net/forum?id=ioyq7NsR1KJ) [[code]](https://github.com/susheels/adgcl)
7. **Graph Contrastive Learning with Adaptive Augmentation** [The WebConf 2021] [[paper]](https://arxiv.org/abs/2010.14945) [[code]](https://github.com/CRIPAC-DIG/GCA)
8. **Semi-Supervised and Self-Supervised Classification with Multi-View Graph Neural Networks** [CIKM 2021] [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482477) 
9. **Graph Contrastive Learning Automated** [ICML 2021] [[paper]](https://arxiv.org/abs/2106.07594) [[code]](https://github.com/Shen-Lab/GraphCL_Automated)
10. **Graph Data Augmentation based on Adaptive Graph Convolution for Skeleton-based Action Recognition** [ICCSNT 2021] [[paper]](https://ieeexplore.ieee.org/abstract/document/9615451)

#### 2020

1. **Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning** [ICDM 2020] [[paper]](https://arxiv.org/abs/2009.10273) [[code]](https://github.com/yzjiao/Subg-Con)
2. **Contrastive Multi-View Representation Learning on Graphs** [ICML 2020] [[paper]](https://arxiv.org/abs/2006.05582) [[code]](https://github.com/kavehhassani/mvgrl)
3. **Graph Contrastive Learning with Augmentations** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2010.13902) [[code]](https://github.com/Shen-Lab/GraphCL)
4. **Deep Graph Contrastive Representation Learning** [ICML 2020] [[paper]](https://arxiv.org/abs/2006.04131) [[code]](https://github.com/CRIPAC-DIG/GRACE)

#### 2019

1. **Deep Graph Infomax** [ICLR 2019] [[paper]](https://arxiv.org/abs/1809.10341) [[openreview]](https://openreview.net/forum?id=rklz9iAcKQ) [[code]](https://github.com/PetarV-/DGI)

### Consistency learning

1. **NodeAug: Semi-Supervised Node Classification with Data Augmentation** [KDD 2020] [[paper]](https://bhooi.github.io/papers/nodeaug_kdd20.pdf) 
2. **Graph Random Neural Network for Semi-Supervised Learning on Graphs** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2005.11079) [[code]](https://github.com/THUDM/GRAND)

---

<a name="gsl" />

# Graph Structure Learning

<a name="gsl-survey" />


## Survey

1. **A Survey on Graph Structure Learning: Progress and Opportunities** [Arxiv 2022] [[paper]](https://arxiv.org/abs/2103.03036)

<a name="gsl-models" />


## Representive GSL methods

### Metric-based

1. **Adaptive Graph Convolutional Neural Networks** [AAAI 2018] [[paper]](https://arxiv.org/abs/1801.03226) [[code]](https://github.com/codemarsyu/Adaptive-Graph-Convolutional-Network)
2. **Graph-Revised Convolutional Network** [ECML-PKDD 2020] [[paper]](https://arxiv.org/abs/1911.07123) [[code]](https://github.com/Maysir/GRCN)
3. **Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2006.13009) [[code]](https://github.com/hugochan/IDGL)
4. **Heterogeneous Graph Structure Learning for Graph Neural Networks** [AAAI 2021] [[paper]](http://www.shichuan.org/doc/100.pdf) [[code]](https://github.com/Andy-Border/HGSL)
5. **Diffusion Improves Graph Learning** [NeurIPS 2019] [[paper]](https://arxiv.org/abs/1911.05485) [[code]](https://github.com/klicperajo/gdc)
6. **AM-GCN: Adaptive Multi-channel Graph Convolutional Networks** [KDD 2020] [[paper]](https://arxiv.org/abs/2007.02265) [[code]](https://github.com/zhumeiqiBUPT/AM-GCN)

### Neural

1. **Semi-supervised Learning with Graph Learning Convolutional Networks** [CVPR 2019] [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) [[code_tensorflow]](https://github.com/jiangboahu/GLCN-tf) 

2. **Learning to Drop: Robust Graph Neural Network via Topological Denoising** [WSDM 2021] [[paper]](https://arxiv.org/abs/2011.07057) [[code]](https://github.com/flyingdoog/PTDNet)
3. **Graph Structure Learning with Variational Information Bottleneck** [AAAI 2022] [[paper]](https://arxiv.org/abs/2112.08903) 
4. **Rethinking Graph Transformers with Spectral Attention** [NeurIPS 2021] [[paper]](https://arxiv.org/abs/2106.03893) [[openreview]](https://openreview.net/forum?id=huAdB-Tj4yG) [[code]](https://github.com/DevinKreuzer/SAN)

### Direct

1. **Exploring Structure-Adaptive Graph Learning for Robust SemiSupervised Classification** [ICME 2020] [[paper]](https://arxiv.org/abs/1904.10146) 
2. **Graph Structure Learning for Robust Graph Neural Networks** [KDD 2020] [[paper]](https://arxiv.org/abs/2005.10203) [[code]](https://github.com/ChandlerBang/Pro-GNN)
3. **Speedup Robust Graph Structure Learning with Low-Rank Information** [CIKM 2021] [[paper]](http://xiangliyao.cn/papers/cikm21-hui.pdf) 
