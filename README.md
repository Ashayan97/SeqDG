# Domain Generalization using Action Sequences for Egocentric Action Recognition

**Contributors**  
Amirshayan Nasirimajda  
Chiara Plizzaria  
Simone Alberto Peironea  
Marco Cicconea  
Giuseppe Avertaa  
Barbara Caputoa  

Politecnico di Torino

![Project Illustration](content/teaser_ral.pdf) <!-- Replace with actual image path after uploading -->

## Abstract

Recognizing human activities from visual inputs, particularly through a first-person viewpoint, is essential for enabling robots to replicate human behavior. Egocentric vision, characterized by cameras worn by observers, captures diverse changes in illumination, viewpoint, and environment. This variability leads to a notable drop in the performance of Egocentric Action Recognition models when tested in environments not seen during training. 

In this paper, we tackle these challenges by proposing a **domain generalization approach** for Egocentric Action Recognition. Our insight is that action sequences often reflect consistent user intent across visual domains. By leveraging action sequences, we aim to enhance the model’s generalization ability across unseen environments.

Our proposed method, named **SeqDG**, introduces:
- A **visual-text sequence reconstruction objective** (**SeqRec**) that uses contextual cues from both text and visual inputs to reconstruct the central action of the sequence.
- A **domain mixing strategy** (**SeqMix**) to enhance robustness by training on mixed action sequences from different domains.

We validate SeqDG on the **EGTEA** and **EPIC-KITCHENS-100** datasets.  
**Results**:
- On **EPIC-KITCHENS-100**, SeqDG leads to **+2.4%** relative average improvement in cross-domain action recognition in unseen environments.
- On **EGTEA**, SeqDG achieves **+0.6% Top-1 accuracy** over the SOTA in intra-domain action recognition.

---

## ⚠️ Warning

This repository is still under development.  
The code is **not yet cleaned**, and final components are subject to change.  
Stay tuned for updates and a stable release.

---

## Data

| Dataset                | Link                                                                 |
|------------------------|----------------------------------------------------------------------|
| EPIC-KITCHENS-100      | [EPIC-KITCHENS-100 Dataset](https://epic-kitchens.github.io/2020)    |
| EGTEA Gaze+ Dataset    | [EGTEA Dataset](https://cbs.ic.gatech.edu/egtea/)                   |

---

## Citation

<!-- If you use this code or ideas from the paper, please cite our work: -->

