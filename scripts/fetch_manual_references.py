from io import StringIO

import pandas as pd

# From https://github.com/CYHSM/awesome-neuro-ai-papers

references_0 = """Schneider, S., Lee, J. H., & Mathis, M. W. [**Learnable latent embeddings for joint behavioral and neural analysis**](https://arxiv.org/abs/2204.00673) arXiv (2022)

Raju, R. V., Guntupalli, J. S., Zhou, G., Lázaro-Gredilla, M., & George, D. [**Space is a latent sequence: Structured sequence learning as a unified theory of representation in the hippocampus**](https://arxiv.org/pdf/2212.01508.pdf) arXiv (2022)

Millet, J., Caucheteux, C., Orhan, P., Boubenec, Y., Gramfort, A., Dunbar, E., ... & King, J. R. [**Toward a realistic model of speech processing in the brain with self-supervised learning**](https://arxiv.org/abs/2206.01685) arXiv (2022)

Ivanova, A. A., Schrimpf, M., Anzellotti, S., Zaslavsky, N., Fedorenko, E., & Isik, L. [**Beyond linear regression: mapping models in cognitive neuroscience should align with research goals**](https://arxiv.org/abs/2208.10668) arXiv (2022)

Sucevic, J., & Schapiro, A. C. [**A neural network model of hippocampal contributions to category learning**](https://www.biorxiv.org/content/10.1101/2022.01.12.476051v1.full.pdf) bioRxiv (2022)

Schmidgall, Samuel, and Joe Hays. [**Learning to learn online with neuromodulated synaptic plasticity in spiking neural networks.**](https://www.biorxiv.org/content/10.1101/2022.06.24.497562v1) bioRxiv (2022)

Adolfi, F., Bowers, J. S., & Poeppel, D. [**Successes and critical failures of neural networks in capturing human-like speech recognition**](https://arxiv.org/abs/2204.03740) arXiv (2022)

Bakhtiari, S., Mineault, P., Lillicrap, T., Pack, C., & Richards, B. [**The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning**](https://www.biorxiv.org/content/10.1101/2021.06.18.448989v3.full) NeurIPS (2021)

Conwell, C., Mayo, D., Barbu, A., Buice, M., Alvarez, G., & Katz, B. [**Neural regression, representational similarity, model zoology & neural taskonomy at scale in rodent visual cortex**](https://proceedings.neurips.cc/paper/2021/file/2c29d89cc56cdb191c60db2f0bae796b-Paper.pdf) NeurIPS (2021)

Krotov, Dmitry. [**Hierarchical associative memory**](https://arxiv.org/abs/2107.06446) arXiv (2021)

Krotov, Dmitry, and John Hopfield. [**Large associative memory problem in neurobiology and machine learning**](https://arxiv.org/abs/2008.06996) ICLR (2021)

Whittington, J. C., Warren, J., & Behrens, T. E. [**Relating transformers to models and neural representations of the hippocampal formation**](https://arxiv.org/abs/2112.04035) arXiv (2021)

Nonaka, S., Majima, K., Aoki, S. C., & Kamitani, Y. [**Brain hierarchy score: Which deep neural networks are hierarchically brain-like?**](https://www.sciencedirect.com/science/article/pii/S2589004221009810) IScience (2021)

Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., ... & Fedorenko, E. [**The neural architecture of language: Integrative modeling converges on predictive processing**](https://evlab.mit.edu/assets/papers/Schrimpf_et_al_2021_PNAS.pdf) PNAS (2021)

Liang, Yuchen, Chaitanya K. Ryali, Benjamin Hoover, Leopold Grinberg, Saket Navlakha, Mohammed J. Zaki, and Dmitry Krotov. [**Can a Fruit Fly Learn Word Embeddings?**](https://arxiv.org/abs/2101.06887) ICLR (2021)

Liu, Helena Y., Stephen Smith, Stefan Mihalas, Eric Shea-Brown, and Uygar Sümbül [**Cell-type–specific neuromodulation guides synaptic credit assignment in a spiking neural network.**](https://www.pnas.org/doi/10.1073/pnas.2111821118) Proceedings of the National Academy of Sciences of the United States of America (2021)

George, D., Rikhye, R. V., Gothoskar, N., Guntupalli, J. S., Dedieu, A., & Lázaro-Gredilla, M. [**Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps**](https://www.nature.com/articles/s41467-021-22559-5) Nature communications (2021)

Whittington, J. C., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. [**The Tolman-Eichenbaum machine: Unifying space and relational memory through generalization in the hippocampal formation**](https://www.sciencedirect.com/science/article/pii/S009286742031388X) Cell (2020)

Banino, A., Badia, A. P., Köster, R., Chadwick, M. J., Zambaldi, V., Hassabis, D. & Blundell, C. [**Memo: A deep network for flexible combination of episodic memories**](https://arxiv.org/abs/2001.10913) arXiv (2020)

Chengxu Zhuang, Siming Yan, Aran Nayebi, Martin Schrimpf, Michael C. Frank, James J. DiCarlo, Daniel L. K. Yamins [**Unsupervised Neural Network Models of the Ventral Visual Stream**](https://www.biorxiv.org/content/10.1101/2020.06.16.155556v1.full.pdf) bioRxiv (2020)

Tyler Bonnen, Daniel L.K. Yaminsa, Anthony D. Wagner [**When the ventral visual stream is not enough: A deep learning account of medial temporal lobe involvement in perception**](https://www.biorxiv.org/content/10.1101/2020.10.07.327171v1.full.pdf) bioRxiv (2020)

Kim, K., Sano, M., De Freitas, J., Haber, N., & Yamins, D. [**Active World Model Learning with Progress Curiosity**](https://arxiv.org/abs/2007.07853) arXiv (2020)

Guangyu Robert Yang, Xiao-Jing Wang [**Artificial Neural Networks for Neuroscientists: A Primer**](https://www.cell.com/neuron/fulltext/S0896-6273(20)30705-4) Neuron (2020)

Glaser G.I., Benjamin, S.A., Chowdhury, H.R., Perich G.M., Miller, L.E., Kording, K.P. [**Machine Learning for Neural Decoding**](https://www.eneuro.org/content/7/4/ENEURO.0506-19.2020) eNeuro (2020)

Jones, I. S., & Kording, K. P. [**Can Single Neurons Solve MNIST? The Computational Power of Biological Dendritic Trees**](https://arxiv.org/abs/2009.01269) arXiv (2020)

Rolnick, D., & Kording, K. [**Reverse-engineering deep ReLU networks**](http://proceedings.mlr.press/v119/rolnick20a/rolnick20a.pdf) ICML (2020)

Geirhos, R., Narayanappa, K., Mitzkus, B., Bethge, M., Wichmann, F. A., & Brendel, W. [**On the surprising similarities between supervised and self-supervised models**](https://arxiv.org/abs/2010.08377) arXiv (2020)

Storrs, K. R., Kietzmann, T. C., Walther, A., Mehrer, J., & Kriegeskorte, N. [**Diverse deep neural networks all predict human IT well, after training and fitting**](https://www.biorxiv.org/content/10.1101/2020.05.07.082743v1.abstract) bioRxiv (2020)

Yonatan Sanz Perl, Hernán Boccacio, Ignacio Pérez-Ipiña, Federico Zamberlán, Helmut Laufs, Morten Kringelbach, Gustavo Deco, Enzo Tagliazucchi [**Generative embeddings of brain collective dynamics using variational autoencoders**](https://arxiv.org/pdf/2007.01378.pdf) arXiv (2020)

George, D., Lazaro-Gredilla, M., Lehrach, W., Dedieu, A., & Zhou, G. [**A detailed mathematical theory of thalamic and cortical microcircuits based on inference in a generative vision model**](https://www.biorxiv.org/content/10.1101/2020.09.09.290601v1.abstract) bioRxiv (2020)

van Bergen, R. S., & Kriegeskorte, N. [**Going in circles is the way forward: the role of recurrence in visual inference**](https://arxiv.org/pdf/2003.12128.pdf) arXiv (2020)

Joseph G. Makin, David A. Moses, Edward F. Chang [**Machine translation of cortical activity to text with an encoder–decoder framework**](https://www.nature.com/articles/s41593-020-0608-8) Nature Neuroscience (2020)

Richards, B. A., & Lillicrap, T. P. [**Dendritic solutions to the credit assignment problem**](https://www.sciencedirect.com/science/article/pii/S0959438818300485) Current opinion in neurobiology (2019)

Sinz, F. H., Pitkow, X., Reimer, J., Bethge, M., & Tolias, A. S. [**Engineering a less artificial intelligence**](https://www.sciencedirect.com/science/article/pii/S0896627319307408) Neuron (2019)

Kubilius, J., Schrimpf, M., Kar, K., Rajalingham, R., Hong, H., Majaj, N. & DiCarlo, J. J. [**Brain-like object recognition with high-performing shallow recurrent ANNs**](https://papers.nips.cc/paper/2019/hash/7813d1590d28a7dd372ad54b5d29d033-Abstract.html) Advances in Neural Information Processing Systems (2019)

Barrett, D. G., Morcos, A. S., & Macke, J. H. [**Analyzing biological and artificial neural networks: challenges with opportunities for synergy?**](https://www.sciencedirect.com/science/article/pii/S0959438818301569) Current opinion in neurobiology (2019)

Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M., & Harris, K. D. [**High-dimensional geometry of population responses in visual cortex**](https://www.nature.com/articles/s41586-019-1346-5) Nature (2019)

Beniaguev David, Segev Idan, London Michael [**Single Cortical Neurons as Deep Artificial Neural Networks**](https://www.biorxiv.org/content/10.1101/613141v1.full.pdf) bioRxiv (2019)

Krotov, D. & Hopfield, J.J. [**Unsupervised learning by competing hidden units**](https://www.pnas.org/content/pnas/116/16/7723.full.pdf) PNAS (2019)

Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass [**A solution to the learning dilemma for recurrent 2 networks of spiking neurons**](https://www.biorxiv.org/content/10.1101/738385v3) bioRxiv (2019)

Albert Gidon, Timothy Adam Zolnik, Pawel Fidzinski, Felix Bolduan, Athanasia Papoutsi, Panayiota Poirazi, Martin Holtkamp, Imre Vida, Matthew Evan Larkum [**Dendritic action potentials and computation in human layer 2/3 cortical neurons**](https://science.sciencemag.org/content/367/6473/83.long) Science (2019)

Adam Gaier, David Ha [**Weight Agnostic Neural Networks**](https://arxiv.org/abs/1906.04358) arXiv (2019)

Ben Sorscher, Gabriel C. Mel, Surya Ganguli, Samuel A. Ocko [**A unified theory for the origin of grid cells through the lens of pattern formation**](https://papers.nips.cc/paper/9191-a-unified-theory-for-the-origin-of-grid-cells-through-the-lens-of-pattern-formation.pdf) NeurIPS (2019)

Sara Hooker, Aaron Courville, Yann Dauphin, Andrea Frome [**Selective Brain Damage: Measuring the Disparate Impact of Model Pruning**](https://arxiv.org/abs/1911.05248) arXiv (2019)

Walker, E. Y., Sinz, F. H., Cobos, E., Muhammad, T., Froudarakis, E., Fahey, P. G. & Tolias, A. S. [**Inception loops discover what excites neurons most using deep predictive models**](https://www.nature.com/articles/s41593-019-0517-x) Nature neuroscience (2019)

Alessio Ansuini, Alessandro Laio, Jakob H. Macke, Davide Zoccolan [**Intrinsic dimension of data representations in deep neural networks**](https://arxiv.org/abs/1905.12784) arXiv (2019)

Josh Merel, Diego Aldarondo, Jesse Marshall, Yuval Tassa, Greg Wayne, Bence Ölveczky [**Deep neuroethology of a virtual rodent**](https://arxiv.org/abs/1911.09451) arXiv (2019)

Zhe Li, Wieland Brendel, Edgar Y. Walker, Erick Cobos, Taliah Muhammad, Jacob Reimer, Matthias Bethge, Fabian H. Sinz, Xaq Pitkow, Andreas S. Tolias [**Learning From Brains How to Regularize Machines**](https://arxiv.org/abs/1911.05072) arXiv (2019)

Hidenori Tanaka, Aran Nayebi, Niru Maheswaranathan, Lane McIntosh, Stephen Baccus, Surya Ganguli [**From deep learning to mechanistic understanding in neuroscience: the structure of retinal prediction**](https://papers.nips.cc/paper/9060-from-deep-learning-to-mechanistic-understanding-in-neuroscience-the-structure-of-retinal-prediction) NeurIPS (2019)

Stefano Recanatesi, Matthew Farrell ,Guillaume Lajoie, Sophie Deneve, Mattia Rigotti, and Eric Shea-Brown [**Predictive learning extracts latent space representations from sensory observations**](https://www.biorxiv.org/content/biorxiv/early/2019/07/13/471987.full.pdf) BiorXiv (2019)

Nasr, Khaled, Pooja Viswanathan, and Andreas Nieder. [**Number detectors spontaneously emerge in a deep neural network designed for visual object recognition.**](https://advances.sciencemag.org/content/5/5/eaav7903) Science Advances (2019)

Bashivan, Pouya, Kohitij Kar, and James J. DiCarlo. [**Neural population control via deep image synthesis.**](https://science.sciencemag.org/content/364/6439/eaav9436) Science (2019)

Ponce, Carlos R., Will Xiao, Peter F. Schade, Till S. Hartmann, Gabriel Kreiman, and Margaret S. Livingstone. [**Evolving Images for Visual Neurons Using a Deep Generative Network Reveals Coding Principles and Neuronal Preferences**](https://www.sciencedirect.com/science/article/pii/S0092867419303915) Cell (2019)

Kar, Kohitij, Jonas Kubilius, Kailyn M. Schmidt, Elias B. Issa, and James J. DiCarlo. [**Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior.**](https://www.nature.com/articles/s41593-019-0392-5) Nature Neuroscience (2019)

Russin, Jake, Jason Jo, and Randall C. O'Reilly. [**Compositional generalization in a deep seq2seq model by separating syntax and semantics.**](https://arxiv.org/abs/1904.09708) arXiv (2019)

Rajalingham, Rishi, Elias B. Issa, Pouya Bashivan, Kohitij Kar, Kailyn Schmidt, and James J. DiCarlo. [**Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks.**](http://www.jneurosci.org/content/38/33/7255) Journal of Neuroscience (2018)

Eslami, SM Ali, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos, Marta Garnelo, Avraham Ruderman et al. [**Neural scene representation and rendering.**](https://science.sciencemag.org/content/360/6394/1204) Science (2018)

Banino, Andrea, Caswell Barry, Benigno Uria, Charles Blundell, Timothy Lillicrap, Piotr Mirowski, Alexander Pritzel et al. [**Vector-based navigation using grid-like representations in artificial agents.**](https://www.nature.com/articles/s41586-018-0102-6) Nature (2018)

Schrimpf, Martin, Kubilius, Jonas, Hong, Ha, Majaj, Najib J., Rajalingham, Rishi, Issa, Elias B., Kar, Kohitij, Bashivan, Pouya, Prescott-Roy, Jonathan, Geiger, Franziska, Schmidt, Kailyn, Yamins, Daniel L. K., and DiCarlo, James J. [**Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?**](https://www.biorxiv.org/content/10.1101/407007) bioRxiv (2018)

Kell, A. J., Yamins, D. L., Shook, E. N., Norman-Haignere, S. V., & McDermott, J. H. [**A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy**](https://www.sciencedirect.com/science/article/pii/S0896627318302502) Neuron (2018)

Guerguiev, Jordan, Timothy P. Lillicrap, and Blake A. Richards. [**Towards deep learning with segregated dendrites.**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5716677/) ELife (2017).

Kanitscheider, I., & Fiete, I. [**Training recurrent networks to generate hypotheses about how the brain solves hard navigation problems**](https://arxiv.org/abs/1609.09059) arXiv (2017)

George, D., Lehrach, W., Kansky, K., Lázaro-Gredilla, M., Laan, C., Marthi, B., ... & Phoenix, D. S. [**A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs**](https://www.science.org/doi/full/10.1126/science.aag2612) Science (2017)

DeWolf, T., Stewart, T. C., Slotine, J. J., & Eliasmith, C. [**A spiking neural model of adaptive arm control**](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2016.2134) Proceedings of the Royal Society B: Biological Sciences, (2016)

Bengio, Yoshua, Dong-Hyun Lee, Jorg Bornschein, Thomas Mesnard, and Zhouhan Lin. [**Towards biologically plausible deep learning.**](https://arxiv.org/abs/1502.04156) arXiv (2015).

Güçlü, Umut, and Marcel AJ van Gerven. [**Deep neural networks reveal a gradient in the complexity of neural representations across the ventral stream.**](http://www.jneurosci.org/content/35/27/10005) Journal of Neuroscience (2015)

Cadieu, Charles F., Ha Hong, Daniel LK Yamins, Nicolas Pinto, Diego Ardila, Ethan A. Solomon, Najib J. Majaj, and James J. DiCarlo. [**Deep neural networks rival the representation of primate IT cortex for core visual object recognition.**](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003963) PLoS computational biology (2014)
"""

references_1 = """Zador, A., Richards, B., Ölveczky, B., Escola, S., Bengio, Y., Boahen, K., ... & Tsao, D. [**Toward next-generation artificial intelligence: catalyzing the NeuroAI revolution**](https://arxiv.org/abs/2210.08340) arXiv (2022)

Doerig, A., Sommers, R., Seeliger, K., Richards, B., Ismael, J., Lindsay, G., ... & Kietzmann, T. C. [**The neuroconnectionist research programme**](https://arxiv.org/abs/2209.03718) arXiv (2022)

Lindsay, G. W. [**Convolutional neural networks as a model of the visual system: Past, present, and future**](https://arxiv.org/pdf/2001.07092.pdf) arXiv (2021)

Hasselmo, M. E., Alexander, A. S., Hoyland, A., Robinson, J. C., Bezaire, M. J., Chapman, G. W., ... & Dannenberg, H. [**The Unexplored Territory of Neural Models: Potential Guides for Exploring the Function of Metabotropic Neuromodulation**](https://www.sciencedirect.com/science/article/abs/pii/S0306452220302141) Neuroscience (2021)

Bermudez-Contreras, E., Clark, B.J., Wilber, A. [**The Neuroscience of Spatial Navigation and the Relationship to Artificial Intelligence**](https://www.frontiersin.org/articles/10.3389/fncom.2020.00063/full) Front. Comput. Neurosci. (2020)

Botvinick, M., Wang, J.X., Dabney, W., Miller, K.J., Kurth-Nelson, Z. [**Deep Reinforcement Learning and Its Neuroscientific Implications**](https://www.cell.com/neuron/fulltext/S0896-6273(20)30468-2) Neuron (2020)

Lillicrap, T.P., Santoro, A., Marris, L., Akerman, C.J. & Hinton, G. [**Backpropagation and the brain**](https://www.nature.com/articles/s41583-020-0277-3) Nature Reviews Neuroscience, (2020)

Saxe, A., Nelli, S. & Summerfield, C. [**If deep learning is the answer, then what is the question?**](https://arxiv.org/abs/2004.07580) arXiv, (2020)

Hasson, U., Nastase, S. A., & Goldstein, A. [**Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks.**](https://www.sciencedirect.com/science/article/abs/pii/S089662731931044X) Neuron (2020)

Schrimpf, M., Kubilius, J., Lee, M. J., Ratan Murty, N. A., Ajemian, R., & DiCarlo, J. J. [**Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence.**](https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X) Neuron (2020)

Merel, J., Botvinick, M., & Wayne, G. [**Hierarchical motor control in mammals and machines**](https://www.nature.com/articles/s41467-019-13239-6) Nature communications (2019)

Storrs, K. R., & Kriegeskorte, N. [**Deep learning for cognitive neuroscience.**](https://arxiv.org/pdf/1903.01458.pdf) arXiv (2019)

Zador, M.Z. [**A critique of pure learning and what artificial neural networks can learn from animal brains**](https://www.nature.com/articles/s41467-019-11786-6), Nature Communications, (2019)

Richards, Blake A., Timothy P. Lillicrap, Philippe Beaudoin, Yoshua Bengio, Rafal Bogacz, Amelia Christensen, Claudia Clopath et al. [**A deep learning framework for neuroscience.**](https://www.nature.com/articles/s41593-019-0520-2) Nature neuroscience (2019)

Kietzmann, T. C., McClure, P., & Kriegeskorte, N. (2018). [**Deep neural networks in computational neuroscience**](https://www.biorxiv.org/content/biorxiv/early/2018/06/05/133504.full.pdf) BioRxiv, (2018)

Hassabis, Demis, Dharshan Kumaran, Christopher Summerfield, and Matthew Botvinick. [**Neuroscience-inspired artificial intelligence.**](https://www.cell.com/neuron/fulltext/S0896-6273(17)30509-3) Neuron (2017)

Lake, Brenden M., Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. [**Building machines that learn and think like people.**](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A9535B1D745A0377E16C590E14B94993/S0140525X16001837a.pdf/building_machines_that_learn_and_think_like_people.pdf) Behavioral and brain sciences (2017). 

Marblestone, Adam H., Greg Wayne, and Konrad P. Kording. [**Toward an integration of deep learning and neuroscience.**](https://www.frontiersin.org/articles/10.3389/fncom.2016.00094/full) Frontiers in computational neuroscience (2016)
"""

# From https://github.com/FrancescoInnocenti/Neuro_AI_Papers

references_2 = """Title,Link,Category
"Is the brain a good model for machine intelligence?",https://www.nature.com/articles/482462a,"Neuro-AI papers 🧠💻"
"What Intelligent Machines Need to Learn From the Neocortex",https://ieeexplore.ieee.org/abstract/document/7934229,"Neuro-AI papers 🧠💻"
"To Advance Artificial Intelligence, Reverse-Engineer the Brain",https://www.wired.com/story/to-advance-artificial-intelligence-reverse-engineer-the-brain/,"Neuro-AI papers 🧠💻"
"The intertwined quest for understanding biological intelligence and creating artificial intelligence",https://neuroscience.stanford.edu/news/intertwined-quest-understanding-biological-intelligence-and-creating-artificial-intelligence,"Neuro-AI papers 🧠💻"
"How AI and neuroscience drive each other forwards",https://www.nature.com/articles/d41586-019-02212-4,"Neuro-AI papers 🧠💻"
"Using neuroscience to develop artificial intelligence",https://science.sciencemag.org/content/363/6428/692,"Neuro-AI papers 🧠💻"
"Neuroscience-Inspired Artificial Intelligence",http://www.sciencedirect.com/science/article/pii/S0896627317305093,"Surveys"
"Building machines that learn and think like people",https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/A9535B1D745A0377E16C590E14B94993,"Surveys"
"Cognitive computational neuroscience",https://www.nature.com/articles/s41593-018-0210-5,"Surveys"
"Natural and Artificial Intelligence: A brief introduction to the interplay between AI and neuroscience research",https://www.sciencedirect.com/science/article/pii/S0893608021003683,"Surveys"
"The roles of supervised machine learning in systems neuroscience",https://www.sciencedirect.com/science/article/pii/S0301008218300856,"Surveys"
"What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated",https://www.sciencedirect.com/science/article/pii/S1364661316300432,"Surveys"
"Computational Foundations of Natural Intelligence",https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5770642/,"Surveys"
"Insights from the brain: The road towards Machine Intelligence",https://www.insightsfromthebrain.com,"Surveys"
"The Mutual Inspirations of Machine Learning and Neuroscience",https://www.sciencedirect.com/science/article/pii/S089662731500255X,"Surveys"
"A deep learning framework for neuroscience",https://www.nature.com/articles/s41593-019-0520-2,"Reviews & perspectives"
"How learning unfolds in the brain: toward an optimization view",https://www.sciencedirect.com/science/article/abs/pii/S0896627321006772,"Reviews & perspectives"
"If deep learning is the answer, what is the question?",https://www.nature.com/articles/s41583-020-00395-8,"Reviews & perspectives"
"Biological constraints on neural network models of cognitive function",https://www.nature.com/articles/s41583-021-00473-5,"Reviews & perspectives"
"Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks",http://www.sciencedirect.com/science/article/pii/S089662731931044X,"Reviews & perspectives"
"Engineering a Less Artificial Intelligence",http://www.sciencedirect.com/science/article/pii/S0896627319307408,"Reviews & perspectives"
"Deep Neural Networks Help to Explain Living Brains",https://www.quantamagazine.org/deep-neural-networks-help-to-explain-living-brains-20201028/,"Reviews & perspectives"
"Artificial Neural Networks for Neuroscientists: A Primer",https://www.sciencedirect.com/science/article/pii/S0896627320307054,"Reviews & perspectives"
"Lessons From Deep Neural Networks for Studying the Coding Principles of Biological Neural Networks",https://www.frontiersin.org/articles/10.3389/fnsys.2020.615129/full,"Reviews & perspectives"
"A neural network walks into a lab: towards using deep nets as models for human behavior",http://arxiv.org/abs/2005.02181,"Reviews & perspectives"
"Deep Learning for Cognitive Neuroscience",http://arxiv.org/abs/1903.01458,"Reviews & perspectives"
"Deep neural network models of sensory systems: windows onto the role of task constraints",https://www.sciencedirect.com/science/article/pii/S0959438818302034,"Reviews & perspectives"
"What does it mean to understand a neural network?",http://arxiv.org/abs/1907.06374,"Reviews & perspectives"
"Deep Neural Networks in Computational Neuroscience",https://www.biorxiv.org/content/10.1101/133504v2,"Reviews & perspectives"
"Principles for models of neural information processing",https://www.sciencedirect.com/science/article/pii/S1053811917306638,"Reviews & perspectives"
"Toward an Integration of Deep Learning and Neuroscience",https://www.frontiersin.org/articles/10.3389/fncom.2016.00094/full,"Reviews & perspectives"
"Using goal-driven deep learning models to understand sensory cortex",https://www.nature.com/articles/nn.4244,"Reviews & perspectives"
"Deep Neural Networks: A New Framework for Modeling Biological Vision and Brain Information Processing",https://www.annualreviews.org/doi/10.1146/annurev-vision-082114-035447,"Reviews & perspectives"
"From the neuron doctrine to neural networks",https://www.nature.com/articles/nrn3962,"Reviews & perspectives"
"The recent excitement about neural networks",https://europepmc.org/article/med/2911347,"Reviews & perspectives"
"Implications of neural networks for how we think about brain function",https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/implications-of-neural-networks-for-how-we-think-about-brain-function/BF0C676BD8290F6F02235C82865A0623,"Reviews & perspectives"
"On logical inference over brains, behaviour, and artificial neural networks",https://psyarxiv.com/tbmcg/,"Philosophical takes"
"Principles for models of neural information processing",https://www.sciencedirect.com/science/article/pii/S1053811917306638,"Philosophical takes"
"Deep Neural Networks as Scientific Models",http://www.sciencedirect.com/science/article/pii/S1364661319300348,"Philosophical takes"
"Explanatory models in neuroscience: Part 2, constraint-based intelligibility",http://arxiv.org/abs/2104.01489,"Philosophical takes"
"Explanatory models in neuroscience: Part 1, taking mechanistic abstraction seriously",http://arxiv.org/abs/2104.01490,"Philosophical takes"
"Convolutional Neural Networks as a Model of the Visual System: Past, Present, and Future",https://direct.mit.edu/jocn/article/doi/10.1162/jocn_a_01544/97402/Convolutional-Neural-Networks-as-a-Model-of-the,"Vision"
"Going in circles is the way forward: the role of recurrence in visual inference",https://www.sciencedirect.com/science/article/pii/S0959438820301768,"Vision"
"Capturing the objects of vision with neural networks",https://www.nature.com/articles/s41562-021-01194-6,"Vision"
"Deep Learning: The Good, the Bad, and the Ugly",https://www.annualreviews.org/doi/10.1146/annurev-vision-091718-014951,"Vision"
"Unsupervised neural network models of the ventral visual stream",https://www.pnas.org/content/118/3/e2014196118,"Vision"
"Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex",https://arxiv.org/abs/1604.03640,"Vision"
"Learning to see stuff",https://www.sciencedirect.com/science/article/pii/S2352154619300397,"Vision"
"Learning About the World by Learning About Images",https://journals.sagepub.com/doi/10.1177/0963721421990334,"Vision"
"A Unified Theory of Early Visual Representations from Retina to Cortex through Anatomically Constrained Deep CNNs",http://arxiv.org/abs/1901.00945,"Vision"
"Visual Cortex and Deep Networks: Learning Invariant Representations",https://mitpress.mit.edu/books/visual-cortex-and-deep-networks,"Vision"
"Deep Neural Networks Rival the Representation of Primate IT Cortex for Core Visual Object Recognition",https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003963,"Vision"
"Deep Neural Networks Reveal a Gradient in the Complexity of Neural Representations across the Ventral Stream",https://www.jneurosci.org/content/35/27/10005.short,"Vision"
"Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation",https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003915,"Vision"
"Performance-optimized hierarchical models predict neural responses in higher visual cortex",https://www.pnas.org/content/111/23/8619,"Vision"
"A Task-Optimized Neural Network Replicates Human Auditory Behavior, Predicts Brain Responses, and Reveals a Cortical Processing Hierarchy",https://www.sciencedirect.com/science/article/pii/S0896627318302502,"Audition"
"Toward goal-driven neural network models for the rodent whisker-trigeminal system",https://arxiv.org/abs/1706.07555,"Somatosensation"
"A neural network that finds a naturalistic solution for the production of muscle activity",https://www.nature.com/articles/nn.4042,"Motor control"
"Analyzing biological and artificial neural networks: challenges with opportunities for synergy?",https://www.sciencedirect.com/science/article/pii/S0959438818301569,"Validation methods"
"How can deep learning advance computational modeling of sensory information processing?",http://arxiv.org/abs/1810.08651,"Validation methods"
"Neural population control via deep image synthesis",https://science.sciencemag.org/content/364/6439/eaav9436,"Closed-loop experiments"
"Evolving Images for Visual Neurons Using a Deep Generative Network Reveals Coding Principles and Neuronal Preferences",https://www.sciencedirect.com/science/article/pii/S0092867419303915,"Closed-loop experiments"
"Inception loops discover what excites neurons most using deep predictive models",https://www.nature.com/articles/s41593-019-0517-x,"Closed-loop experiments"
"Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?",https://www.biorxiv.org/content/10.1101/407007v2,"Model benchmarks"
"Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence",https://www.sciencedirect.com/science/article/pii/S089662732030605X,"Model benchmarks"
"The Algonauts Project 2021 Challenge: How the Human Brain Makes Sense of a World in Motion",http://arxiv.org/abs/2104.13714,"Model benchmarks"
"Algonauts 2021 Challenge",http://algonauts.csail.mit.edu,"Model benchmarks"
"Cichy et al., 2019",http://arxiv.org/abs/1905.05675,"Model benchmarks"
"The Algonauts Project",https://www.nature.com/articles/s42256-019-0127-z,"Model benchmarks"
"Brain hierarchy score: Which deep neural networks are hierarchically brain-like?",https://www.sciencedirect.com/science/article/pii/S2589004221009810,"Model benchmarks"
"brain hierarchy (BH) score",https://github.com/KamitaniLab/BHscore,"Model benchmarks"
"Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits",https://www.nature.com/articles/s41593-021-00857-x,"Backprop in the brain?"
"Backpropagation and the brain",https://www.nature.com/articles/s41583-020-0277-3,"Backprop in the brain?"
"Artificial Neural Nets Finally Yield Clues to How Brains Learn",https://www.quantamagazine.org/artificial-neural-nets-finally-yield-clues-to-how-brains-learn-20210218/,"Backprop in the brain?"
"Dendritic solutions to the credit assignment problem",https://www.sciencedirect.com/science/article/pii/S0959438818300485,"Backprop in the brain?"
"Control of synaptic plasticity in deep cortical networks",https://www.nature.com/articles/nrn.2018.6,"Backprop in the brain?"
"Reply to ‘Can neocortical feedback alter the sign of plasticity?’",https://www.nature.com/articles/s41583-018-0048-6,"Backprop in the brain?"
"Can the Brain Do Backpropagation? Exact Implementation of Backpropagation in Predictive Coding Networks",https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7610561/,"Backprop in the brain?"
"Dendritic Computing: Branching Deeper into Machine Learning",https://www.sciencedirect.com/science/article/pii/S0306452221005017,"Artificial & biological neurons"
"Single cortical neurons as deep artificial neural networks",https://www.sciencedirect.com/science/article/pii/S0896627321005018,"Artificial & biological neurons"
"How Computationally Complex Is a Single Neuron?",https://www.quantamagazine.org/how-computationally-complex-is-a-single-neuron-20210902/,"Artificial & biological neurons"
"Drawing inspiration from biological dendrites to empower artificial neural networks",https://www.sciencedirect.com/science/article/pii/S0959438821000544,"Artificial & biological neurons"
"Dendritic action potentials and computation in human layer 2/3 cortical neurons",https://science.sciencemag.org/content/367/6473/83,"Artificial & biological neurons"
"Hidden Computational Power Found in the Arms of Neurons",https://www.quantamagazine.org/neural-dendrites-reveal-their-computational-power-20200114/,"Artificial & biological neurons"
"Pyramidal Neuron as Two-Layer Neural Network",https://www.sciencedirect.com/science/article/pii/S0896627303001491,"Artificial & biological neurons"
"Deep learning in spiking neural networks",https://www.sciencedirect.com/science/article/pii/S0893608018303332,"Spiking neural networks"
"A critique of pure learning and what artificial neural networks can learn from animal brains",https://www.nature.com/articles/s41467-019-11786-6,"Nature & nurture"
"Innateness, AlphaZero, and Artificial Intelligence",http://arxiv.org/abs/1801.05667,"Nature & nurture"
"Deep Reinforcement Learning and Its Neuroscientific Implications",http://www.sciencedirect.com/science/article/pii/S0896627320304682,"Reviews & perspectives"
"A distributional code for value in dopamine-based reinforcement learning",https://www.nature.com/articles/s41586-019-1924-6,"Reviews & perspectives"
"Reinforcement Learning, Fast and Slow",https://www.sciencedirect.com/science/article/pii/S1364661319300610,"Reviews & perspectives"
"Reinforcement learning in artificial and biological systems",https://www.nature.com/articles/s42256-019-0025-4,"Reviews & perspectives"
"The successor representation in human reinforcement learning",https://www.nature.com/articles/s41562-017-0180-8,"Reviews & perspectives"
"Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments",https://www.sciencedirect.com/science/article/pii/S0896627320308990,"Experiments"
"Deep Reinforcement Learning and Its Neuroscientific Implications",http://www.sciencedirect.com/science/article/pii/S0896627320304682,"Experiments"
"Validating the Representational Space of Deep Reinforcement Learning Models of Behavior with Neural Data",https://www.biorxiv.org/content/10.1101/2021.06.15.448556v1.abstract,"Experiments"
"A thousand brains: toward biologically constrained AI",https://doi.org/10.1007/s42452-021-04715-0,"The Thousand Brains Theory"
"Grid Cell Path Integration For Movement-Based Visual Object Recognition",http://arxiv.org/abs/2102.09076,"The Thousand Brains Theory"
"A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex",https://www.frontiersin.org/articles/10.3389/fncir.2018.00121/full,"The Thousand Brains Theory"
"Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells",https://www.frontiersin.org/articles/10.3389/fncir.2019.00022/full,"The Thousand Brains Theory"
"A Theory of How Columns in the Neocortex Enable Learning the Structure of the World",https://www.frontiersin.org/articles/10.3389/fncir.2017.00081/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Neural_Circuits&id=295079,"The Thousand Brains Theory"
"Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex",https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full,"The Thousand Brains Theory"
"""

references_3 = """
Title,Link
Neural Circuit Architectural Priors for Embodied Control,https://openreview.net/forum?id=KblXjniQCHY
Causal mapping of human brain function,https://www.nature.com/articles/s41583-022-00583-8
High-performing neural network models of visual cortex benefit from high latent dimensionality,https://www.biorxiv.org/content/10.1101/2022.07.13.499969v1
Investigating Power laws in Deep Representation Learning,https://arxiv.org/abs/2202.05808
Brain-optimized neural networks learn non-hierarchical models of representation in human visual cortex,https://www.biorxiv.org/content/10.1101/2022.01.21.477293v1
Reassessing hierarchical correspondences between brain and deep networks through direct interface,https://www.science.org/doi/10.1126/sciadv.abm2219
Similarity of Neural Network Representations Revisited,http://proceedings.mlr.press/v97/kornblith19a.html
Generalized Shape Metrics on Neural Representations,https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html
Deep Problems with Neural Network Models of Human Vision,https://psyarxiv.com/5zf4s/
Successes and critical failures of neural networks in capturing human-like speech recognition,https://arxiv.org/abs/2204.03740
No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit,https://openreview.net/forum?id=mxi1xKzNFrb
Self-supervised models of audio effectively explain human cortical responses to speech,https://arxiv.org/abs/2205.14252
The neural architecture of language: Integrative modeling converges on predictive processing,https://www.pnas.org/doi/abs/10.1073/pnas.2105646118
Brains and algorithms partially converge in natural language processing,https://www.nature.com/articles/s42003-022-03036-1
A hierarchy of linguistic predictions during natural language comprehension,https://www.pnas.org/doi/10.1073/pnas.2201968119
Artificial neural network language models align neurally and behaviorally with humans even after a developmentally realistic amount of training,https://www.biorxiv.org/content/10.1101/2022.10.04.510681v1.abstract
Neuroprosthesis for Decoding Speech in a Paralyzed Person with Anarthria,https://www.nejm.org/doi/full/10.1056/nejmoa2027540
Semantic reconstruction of continuous language from non-invasive brain recordings,https://www.biorxiv.org/content/10.1101/2022.09.29.509744v1
Decoding speech from non-invasive brain recordings,http://arxiv.org/abs/2208.12266
Synthesizing Speech from Intracranial Depth Electrodes using an Encoder-Decoder Framework,http://arxiv.org/abs/2111.01457
Hybrid Neural Autoencoders for Stimulus Encoding in Visual and Other Sensory Neuroprostheses,http://arxiv.org/abs/2205.13623
A Differentiable Optimisation Framework for The Design of Individualised DNN-based Hearing-Aid Strategies,https://ieeexplore.ieee.org/document/9747683
NeuroGen: Activation optimized image synthesis for discovery neuroscience,https://www.sciencedirect.com/science/article/pii/S1053811921010831
It takes neurons to understand neurons: Digital twins of visual cortex synthesize neural metamers,http://biorxiv.org/lookup/doi/10.1101/2022.12.09.519708
Data-driven emergence of convolutional structure in neural networks,https://www.pnas.org/doi/10.1073/pnas.2201854119
A connectivity-constrained computational account of topographic organization in primate high-level visual cortex,https://www.pnas.org/doi/10.1073/pnas.2112566119
Topographic DCNNs trained on a single self-supervised task capture the functional organization of cortex into visual processing streams,https://openreview.net/forum?id=E1iY-d13smd
Learning Robust Kernel Ensembles with Kernel Average Pooling,https://arxiv.org/abs/2210.00062
Pooling strategies in V1 can account for the functional and structural diversity across species,https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010270
MouseNet: A biologically constrained convolutional neural network model for the mouse visual cortex,https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010427
Building Transformers from Neurons and Astrocytes,https://www.biorxiv.org/content/10.1101/2022.10.12.511910v1
In vitro neurons learn and exhibit sentience when embodied in a simulated game-world,https://www.sciencedirect.com/science/article/pii/S0896627322008066
Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding,http://arxiv.org/abs/2211.06956
High-resolution image reconstruction with latent diffusion models from human brain activity,https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2.abstract
Predictive coding is a consequence of energy efficiency in recurrent neural networks,https://www.biorxiv.org/content/10.1101/2021.02.16.430904v2
General object-based features account for letter perception,https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010522
DNNs trained for object recognition on ImageNet match representations in IT,https://openreview.net/forum?id=i_xiyGq6FNT
"Deep supervised, but not unsupervised, models may explain IT cortical representation",https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003915
Unsupervised neural network models of the ventral visual stream,https://www.pnas.org/content/118/3/e2014196118
Beyond category-supervision: Computational support for domain-general pressures guiding human visual system representation,https://www.biorxiv.org/content/10.1101/2020.06.15.153247v3
Your head is there to move you around: Goal-driven models of the primate dorsal pathway,https://your-head-is-there-to-move-you-around.netlify.app/
The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning,https://ventral-dorsal-model.netlify.app/
Shallow Unsupervised Models Best Predict Neural Responses in Mouse Visual Cortex,https://www.biorxiv.org/content/10.1101/2021.06.16.448730v2
"Neural Regression, Representational Similarity, Model Zoology & Neural Taskonomy at Scale in Rodent Visual Cortex",https://proceedings.neurips.cc//paper/2021/hash/2c29d89cc56cdb191c60db2f0bae796b-Abstract.html
Partial success in closing the gap between human and machine vision,https://arxiv.org/abs/2106.07411
Multimodal neural networks better explain multivoxel patterns in the hippocampus,https://openreview.net/forum?id=6dymbuga7nL
Unsupervised deep learning identifies semantic disentanglement in single inferotemporal face patch neurons,https://www.nature.com/articles/s41467-021-26751-5
Unsupervised learning predicts human perception and misperception of gloss,https://www.nature.com/articles/s41562-021-01097-6
"""


def main():
    # Initialize lists to store the data
    authors = []
    titles = []
    links = []
    categories = []

    r0 = references_0.split("\n")
    r1 = references_1.split("\n")

    ref0 = zip(["primary"] * len(r0), r0)
    ref1 = zip(["review"] * len(r1), r1)

    references = list(ref0) + list(ref1)

    # Iterate over the lines and extract the data
    for category, line in references:
        if line.strip():
            parts = line.strip().split("[")
            author = parts[0].strip()
            print(parts)
            title = parts[1].split("**")[1].strip()
            link = parts[1].split("(")[-1].strip(")]")

            authors.append(author)
            titles.append(title)
            links.append(link)
            categories.append(category)

    # Create a DataFrame and save it as a CSV
    data = {
        "Author": authors,
        "Title": titles,
        "Link": links,
        "Category": categories,
        "Source": "awesome-neuroscience",
    }
    df = pd.DataFrame(data)

    # Read in the references_2 variable as a csv string
    df_2 = pd.read_csv(StringIO(references_2))
    df_2["Source"] = "francesco-innocenti"
    df_3 = pd.read_csv(StringIO(references_3))
    df_3["Source"] = "xcorr"

    df_4 = pd.read_csv("data/raw/papers-hebart.csv")
    df_4.rename({"Keywords": "Category"}, axis=1, inplace=True)
    df_total = pd.concat([df_4, df, df_2, df_3], axis=0)

    print(f"Total number of references: {len(df_total)}")

    df_total.drop_duplicates(subset=["Link"], inplace=True)
    df_total.drop_duplicates(subset=["Title"], inplace=True)

    print(f"After simple de-duplication: {len(df_total)}")

    df_total.to_csv("data/raw/manual-references.csv", index=False)


if __name__ == "__main__":
    main()
