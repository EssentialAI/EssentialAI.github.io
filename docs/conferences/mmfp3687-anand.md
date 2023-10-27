# Multi-label Emotion Analysis in Conversation via Multimodal Knowledge Distillation

<figure markdown="1" style="margin:0 auto; text-align: center;">
![Image title](../assets/images/conferences/acmmm.png){ width="700" }
<!-- <figcaption>Proposed Architecture of the SeMuL-PCD model.</figcaption> -->
</figure>

<p style="text-align: center; margin-top:-10px;">Proposed Architecture of the SeMuL-PCD model.</p>

<center>[PDF](https://cse.buffalo.edu/~jsyuan/papers/2023/mmfp3687-anand.pdf) | [Code](https://github.com/devulapa/multimodal-emotion-recognition) | [Poster]()</center>

Evaluating speaker emotion in conversations is crucial for various applications requiring human-computer interaction. However, co-occurrences of multiple emotional states (e.g. ‘anger’ and ‘frustration’ may occur together or one may influence the occurrence of the other) and their dynamic evolution may vary dramatically
due to the speaker’s internal (e.g., influence of their personalized socio-cultural-educational and demographic backgrounds) and external contexts. Thus far, the previous focus has been on evaluating only the dominant emotion observed in a speaker at a given time, which is susceptible to producing misleading classification decisions
for difficult multi-labels during testing. In this work, we present Self-supervised Multi-Label Peer Collaborative Distillation (SeMuLPCD) Learning via an efficient Multimodal Transformer Network, in which complementary feedback from multiple mode-specific peer networks (e.g.transcript, audio, visual) are distilled into a single mode-ensembled fusion network for estimating multiple emotions simultaneously. The proposed Multimodal Distillation Loss calibrates the fusion network by minimizing the Kullback–Leibler divergence with the peer networks. Additionally, each peer network is conditioned using a self-supervised contrastive objective to improve the generalization across diverse socio-demographic speaker backgrounds. By enabling peer collaborative learning that allows each network to independently learn their mode-specific discriminative patterns, SeMUL-PCD is effective across different conversation environments. In particular, the model not only outperforms the current state-of-the-art models on several large-scale public datasets (e.g., MOSEI, EmoReact and ElderReact), but with around 17% improved weighted F1-score in the cross-dataset experimental settings. The model also demonstrates an impressive
generalization ability across age and demography-diverse populations.

