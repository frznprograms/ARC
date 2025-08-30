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

For this project we used the `uv` package manager, which is efficient and easy to use. Please install `uv` to get started. For more instructions on installing `uv` for your system, please refer to this link: https://docs.astral.sh/uv/getting-started/installation/

To support large file storage, please also install Git Large File Storage (LFS) using the following command: 

```bash
git lfs install
```

To initialise the virtual environment, follow these steps: 

```bash
uv venv
source .venv/bin/activate
uv sync
```
For those who prefer to use a `requirements.txt`, we have you covered as well. Simply run these commands instead: 

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

### How to Reproduce Results

We have created several scripts (both .py and .sh) to allow users to run inference with our pipeline. In this README, we will cover the usage of the overarching inference pipeline, which is the culmination of all our work. For more details on how to run each of the components, please refer to the "sub-READMEs" in `src/encoder`, `src/fasttext` and `src/safety`. For now, here are instructions on how to run inference, for which the Python script is located in `src/pipelines/inference_pipeline.py`.

**Please make sure you run this from root. Unless otherwise stated, our scripts were all meant to be run from root so that users don't have to keep navigating between folders.**

```bash
uv run -m src.pipelines.inference_pipeline
```

Alternatively, for those who prefer to configure entirely in command line, you can run: 

```bash
./run.sh
``` 
from root. Below is a decscription of the possible args you can configure:

| Argument        | Type  | Required | Default                         | Description                                                  |
|-----------------|-------|----------|---------------------------------|--------------------------------------------------------------|
| `--safety-model` | str   | No       | `models/safety-model-test.pkl`  | Path to the safety model `.pkl` file                         |
| `--encoder-model` | str   | No       | `lora_sft_encoder.pth`          | Path to encoder model weights (`.pth`)                       |
| `--review-file`  | str   | No       | `data/for_model/review_1.json`  | Path to JSON review file (with name, category, description, review, rating) |
| `--threshold`    | float | No       | `0.7`                           | Threshold for fasttext heads                                 |

**Note**: the pipeline was designed to take only one review (i.e. one dictionary) at a time. This was a specific design choice, as logically reviews should be evaluated the moment they are posted, not after some time until enough samples are curated for batched inference. We have taken care to ensure our pipeline is **efficient** at inference for each sample.

---

### Sample Web Application 

We have included a minimal web application for users to play around with, just to get an idea of what our implementation is meant to do, and how it can serve as a dynamic review evaluator. Please note it is not exactly what the real implementation will look like, as in real life users will not need to manually enter the name of the place they are reviewing, description, etc. These are things that need to be extracted by developers. Feel free to launch the app and play around with the reviews, locations, and descriptions, to see how the pipeline checks reviews. Once again, from root: 

```bash
# launch app
./launch_app.sh
```

And simply close the tab in your browser or press `Ctrl + c` to terminate.

### Performance

Our pipeline components performed quite well as isolated components. You may refer to the model performance metrics included in the "sub-READMEs" in `src/encoder`, `src/fasttext` and `src/safety` for more details. 

For your own tests, we have included 3 sample reviews in `json` format, which you can find under `data/for_model`. Feel free to use those in our web app, and you can even play around with small word misspellings, or censors.

### Trigger Warning

To aid our model implementation, which utilises a lexicon-based component, we have included a set of toxic words under `src/utils/toxic_lexicon.py`, which can be offensive to some. 

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
