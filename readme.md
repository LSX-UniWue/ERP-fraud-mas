# Agent-Based ERP Data Generation

This repository contains code for the paper 'Occupational Fraud Detection through Agent-based Data Generation' published at the 8th Workshop on MIning DAta for financial applicationS (MIDAS) as part of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD) 2023.

## Abstract
Occupational fraud is an increasing concern for enterprises that is estimated to cause losses of around 5% of company revenue each year. With the increasing data tracked by companies through enterprise resource planning systems, recent research has taken interest in the automated detection of occupational fraud. Automated detection is however hindered by the unavailability of labeled fraud cases which require known occupational frauds within company data and costly expert annotation. Even despite the existence of anomaly detection methods that can be trained on unsupervised data, selecting the ideal preprocessing techniques, the most suitable model, and the optimal hyperparameters necessitates the availability of labeled data for evaluation purposes. To alleviate this issue, we propose to use simulation through multi-agent systems for generating business processes according to best practices from economics and creating labeled synthetic data that closely matches a given unlabeled real-world dataset. We extend an existing simulation by incorporating functionality for including, tracking and automatic labeling of occupational fraud cases. Using this simulation, we propose a framework that decides on important design choices for fraud detection models in enterprise resource planning data and does not require labeled real-world data. We demonstrate in multiple experiments that the framework can aid automated occupational fraud detection through data generation.

## Contained Materials
The multi-agent-based simulation for generating normal and fraudulent business processes is contained in the folder `multi-agent-simulation`.

Trends for the simulation can be generated through the provided python code in the folder `trend-generation`.

All machine learning experiments from the paper are contained in the folder `ml-framework`.

The data contained in `ml-framework/data/erp_fraud` is the ERP fraud detection data of Tritscher et al. [1]. The full data is available at https://professor-x.de/erp-fraud-data.

Results of the hyperparameter studies from the paper experiments can be found unter `ml-framework/outputs/logs/erp_mas`.
