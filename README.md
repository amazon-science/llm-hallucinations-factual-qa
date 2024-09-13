# Identifying Hallucinations in LLMs

## Setup

Set up the conda env by running `setup.sh`
It brings in basic plotting packages as well as captum, which is needed for collecting the token attributions.

## Data sources

An overview of the datasets/models used can be found in the paper under the section 4 **Experimental setup** section of the paper.
In particular, while our **result_collector.py** uses **TriviaQA** directly, for TREX we do/save a sampling in the form of founders/capitals/place_of_birth.csv.
Run `trex_parser.py` to create these data files.

## Artifact data collection

Classifiers and plots will be created on model/derived artifacts like activations, attention, softmax output, attributions.
Artifact data collection is done in **result_collector.py**, is **VERY** time consuming and best done on a powerful machine.
It will write picke files and it gathers more data than used in the paper (in the paper we look at last layer activations, etc).
Once acquired however, the same data can be used for a broader analysis if so desired.

We use models/tokenizers from Huggingface. Softmax/logits are collected directly from the model, attributions are collected using the 
integrated gradients (IG) method available in Captum and activations and attentions (model internal states) are collected using the **register_forward_hook** functionality.

## Plots

Data analysis (the plots in the paper) is done in **plots_tsne.ipynb** and **plots_entropy_and_pca.ipynb**. It corresponds to the 5.1 **Qualitative analysis** section of the paper, however most plots are collected in the appendix.

Once data is collected, we are iterested in comparative plots of softmax/IG attributions/activations across the models and datasets.
This is the reason why we collect the large dicts at the beginning of both notebooks. This is also a time consuming process, but note
that the notebook(s) can also be used on one model/dataset for fast experimentation.
Example: the data source directoiry (in our case **results**) would contain only capitals/falcon-40b_capitals_7_18.pickle while **founders**, **trivia**, **place_of_birth** stay empty.

## Classifiers

We train classifiers on IG, softmax, attention scores, FCC activativations across the models/datasets. The results are in tables 2 and 3 in the **Results** section of the paper. **classifier_model.ipynb** creates basic models and trains them on the data collected by **result_collector.py**.

## SelfCheckGPT

We try to use selfcheckgpt and compare to our results; a notebook is included. SelfcheckGPT does not perform well
with our models; we hypothesize that this is because the models we use are small and the output for nonzero temperature is often subpar. 
We use the **bert-score** and **n-gram** methods from the selfcheckgpt paper in **self_check_gpt.ipynb** and we report the results in the appendix **B** (additional results) of the paper.
