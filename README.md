<h1 style="text-align: center;">Rudy's Rangers Repository, TikTok Techjam 2025</h1>
<h2 style="text-align: center;">Chosen Problem Statment: Filtering the Noise: ML for Trustworthy Location Reviews</h2>

### Authors

1. Soo Weng Kit  
2. Tian Fengyao (Kyrie)  
3. Lee Chun Wayne  
4. Shane Vivek Bharathan  
5. Chen Runjia (Rudy)  

---

### Project Overview

This project tackles the challenge of distinguishing between trustworthy and untrustworthy reviews. Our approach combines machine learning and deep learning models in an ensemble framework. By stacking these models in ascending order of computational cost, the system can quickly filter obvious spam with lightweight models, while reserving more expensive deep learning methods for the harder cases. This layered strategy amortizes the overall cost of prediction, ensuring both efficiency and accuracy in filtering reviews.


---

### Setup Instructions

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

### How to Reproduce Results
*(To be added)*

---

### Citations

#### FastText
```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

#### Google Maps Dataset 
```
https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews
```

#### Internet Violence Study (InViS) Dataset
```
@data{DVN/ANGOX0_2025,
author = {Golbeck, Jen},
publisher = {Harvard Dataverse},
title = {{Internet Violence Study (InViS) Dataset}},
year = {2025},
version = {V1},
doi = {10.7910/DVN/ANGOX0},
url = {https://doi.org/10.7910/DVN/ANGOX0}
}
```

#### ToxiGen Dataset
```
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

### Twitter Toxic Comments Dataset
```
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset?utm_source=chatgpt.com
```
