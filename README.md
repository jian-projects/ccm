# Conversation Clique-based Model for Emotion Recognition in Conversation

This repository contains the source code and additional resources for the paper titled "Conversation Clique-based Model for Emotion Recognition in Conversation" presented at the International Conference on Acoustics, Speech, & Signal Processing (ICASSP) 2024.

## Abstract

Effective extraction and integration of valuable contextual information is the core of models for the Emotion Recognition in Conversation (ERC) task. However, a significant amount of irrelevant information is inevitably introduced when integrating long-range contextual information, perplexing the model greatly and resulting in incorrect emotion identification. To this end, we proposed a Conversation Clique-based Model (CCM), designed to extract the most efficacious contextual information to bolster the semantic quality of utterances. Specifically, we devise an utterance spatial relationship module (SpaRel) to explicitly model structural-level correlations among utterances by using GAT, and an emotion temporal relationship module (TemRel) to implicitly capture the emotion sequence constraints by employing HMM. We conduct extensive experiments on the publicly available MELD dataset, and the experimental results indicate the effectiveness of our proposed model, achieving new state-of-the-art results.

## Installation

Instructions for setting up the environment and installing dependencies.

```bash
# Clone this repository
git clone https://github.com/jianzquan/CCM-ERC.git

# Navigate to the repository directory
cd CCM-ERC

# Install dependencies
# Add here instructions appropriate for your project
```

## Usage

How to run scripts or reproduce the results.

```bash
python run_erc.py
```

## Contents

A list of key files and folders in the repository.

- `src/`: directory containing all the source code.
- `data/`: directory for placing the dataset (if public or synthetic).
- `scripts/`: utility scripts to process data, train models, etc.
- `results/`: model outputs, evaluation results, figures, etc. (if applicable).
- `requirements.txt`: list of dependencies.

## Dataset

Details about the dataset used, if it is available or how to obtain it, and any preprocessing steps that are needed.

## Citing Our Work

If you find the code or the paper useful for your research, please consider citing our paper:

```bibtex
@inproceedings{AuthorYearICASSP,
  title     = {Conversation Clique-based Model for Emotion Recognition in Conversation},
  author    = {Jian, Zhongquan and Li, Jiajian and Yao, Junfeng and Wang, Meihong and Wu, Qingqiang},
  booktitle = {Proceedings of the International Conference on Acoustics, Speech, & Signal Processing (ICASSP)},
  year      = {2024},
  pages     = {}
}
```

## Contributing

Information for others who want to contribute to the project.

- Fork the project.
- Create your feature branch (`git checkout -b feature/AmazingFeature`).
- Commit your changes (`git commit -m 'Add some AmazingFeature'`).
- Push to the branch (`git push origin feature/AmazingFeature`).
- Open a pull request.

## License

Specify the license under which your code is released. Common licenses for academic and open-source projects include MIT, Apache 2.0, and GNU GPL.

`This project is licensed under the [LICENSE NAME] - see the LICENSE.md file for details`

## Contact

- Your Name - email@example.com
- Project Link: https://github.com/yourusername/icassp2024-papercode.git

## Acknowledgments

Credit any collaborators or third-party resources you used. For example:

- Hat tip to anyone whose code was used
- Inspiration
- etc.
