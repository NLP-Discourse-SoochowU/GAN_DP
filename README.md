## Introduction

This project presents the recently proposed GAN-based DRS parser in 
**Adversarial Learning for Discourse Rhetorical Structure Parsing (ACL-IJCNLP2021)**. 
For any questions please directly send e-mails to zzlynx@outlook.com (Longyin Zhang).

#### Installation
- Python 3.6.10 
- transformers 3.0.2
- pytorch 1.5.0
- numpy 1.19.1
- cudatoolkit 9.2 cudnn 7.6.5
- other necessary python packages (etc.)

#### Project Structure
```
---DP_GAN
 |-ch_dp_gan  Chinese discourse parser based on Qiu-W2V
 |-ch_dp_gan_xlnet  Chinese discourse parser based on XLNet
 |-en_dp_gan  English discourse parser based on GloVe and ELMo
 |-en_dp_gan_xlnet  English discourse parser based on XLNet
```
We do not provide data in this project for the data is too large to be uploaded to Github. Please prepare 
the data by yourself if you want to train a parser. By the way, we will provide a pre-trained end2end DRS 
parser (automatic EDU segmentation and discourse parsing) in the project **sota_end2end_parser** for other 
researchers to directly apply it to other NLP tasks.

We have explained in paper that there are some differences between the XLNet-based systems and the other 
systems. I re-checked the system and found that there are still two points that I forgot to mention: 
**(i)** We did not use the sentence and paragraph boundaries in the Chinese system, because the performance 
is bad; 

**(ii)** In the English XLNet-based parser, we did not input the EDU vectors into RNNs before split point 
encoding.

#### Performance Evaluation

As stated in the paper, we employ the *original Parseval (Morey et al. 2017)* to evaluate our English DRS 
parser and report the **micro-averaged F1-score** as performance. We did not report the results based on Marcu's
RST Parseval because the metric will overestimate the performance level of DRS parsing. 

As we know, when using *RST Parseval*, we actually have 19 relation categories considered for evaluation, i.e., 
the 18 rhetorical relations and the **SPAN** tag. Among these tags, the SPAN relation accounts for more than a 
half of the relation labels in RST-DT, which may enlarge the uncertainty of performance comparison. 
Specifically, the systems that predict relation tags for the parent node will show weaker performance than the 
systems that predict the relation category of each child node. **Why?** Usually, the second kind of systems also
employ SPAN tags for model training and this brings in additional gradients for the model to greedily maximize 
the rewards by assigning SPAN label to appropriate tree nodes. However, for the first kind of systems, the SPAN 
labels are assigned only according to their predicted Nuclearity category (our system belongs to this kind). 

Here we report the results of (Yu et al. 2018) and ours on **SPAN** for reference:
```
--- system ---------- P ---- R ---- F
 Yu et al. (2018)    60.9   63.7   62.3 (The parsing results are from Yu Nan)
 Ours                46.1   43.1   44.5
```

Obviously, it's hard to judge whether the performance improvements come from the 18 rhetorical relations or the 
fake relation "SPAN" when using RST Parseval. For more clear performance comparison, we explicitly recommend 
other DRS researchers to use the original Parseval to evaluate their parsers.
 
For Chinese DRS parsing, we use a strict method for performance evaluation, and one can refer to 
https://github.com/NLP-Discourse-SoochowU/t2d_discourseparser for details.

#### Model Training
In this project, we tuned the hyper-parameters for best performance and the details are well shown in the ACL 
paper. Although we had tried our best to check the correction of the paper content, we still find one inaccuracy:
**We trained the XLNet systems for 50 rounds instead of the 30 written in the Appendix.** 

To train the parsing models, run the following commands:
```
   (Chinese) python -m treebuilder.dp_nr.train
   (English) python main.py
```

Due to the utilization of GAN nets, we found that the parsing performance has some fluctuation, it is related to
the hardware and the software running environment you use. It should be noted that some researchers may 
use the preprocessed RST-DT corpus for experiments, there could be some tiny differences when compared with the 
original data. We recommend using the original formal RST-DT corpus for experimentation. 

We will provide a pre-trained end-to-end discourse parser at 
https://github.com/NLP-Discourse-SoochowU/sota_end2end_parser, 
and one can directly use it to generate rhetorical structures for your own article data.

#### Tips
RST-style discourse parsing has long been known to be complicated, which is why the system we provide actually 
contains so many experimental settings. We had conducted a set of experimental comparisons in this system, and 
we found that some experimental details could be helpful for your own system, e.g., the **EDU attention**, the 
**Context attention**, the **residual connection**, the **parameter initialization method**, etc. These code 
details can hardly bring great research value in this period, but they will make your system more stable.

<b>-- License</b>
```
   Copyright (c) 2019, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that
   the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
```
