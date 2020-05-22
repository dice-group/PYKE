
### A Physical Embedding Model for Knowledge Graphs ###

This repository contains the implementation of our approach for learning continous vector representation of knowledge graphs.


- Our approach (PYKE) is a physical embedding model for learning embeddings of RDF graphs. 
By virtue of being akin to a physical simulation, PYKE retains a linear space complexity while generating high quality embeddings. 
We evaluated our approach with two benchmark datasets and showed that it outperforms state-of-the-art approaches on all tasks, while being close to linear in its time complexity on large KGs.

## Installation

```
git clone https://github.com/pyke-KGE/pyke-KGE.git
conda env create -f environment.yml
python execute.py
```

## Reproducing reported results
- [PYKE Drugbank](https://github.com/dice-group/PYKE/blob/master/PYKE_Drugbank.ipynb) and [PYKE DBpedia](https://github.com/dice-group/PYKE/blob/master/PYKE_DBpedia.ipynb) notebooks elucidates the workflow of PYKE as well as reproduces the results of type prediction and cluster purity evaluations.



