## Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning
**Author**: Yangru Huang · Peixi Peng · Yifan Zhao · Haoran Xu · Mengyue Geng · Yonghong Tian

**Abstract**: Integrating RGB frames with alternative modality inputs is gaining increasing traction in many vision-based reinforcement learning (RL) applications. Existing multi-modal vision-based RL methods usually follow a Global Value Estimation (GVE) pipeline, which uses a fused modality feature to obtain a unified global environmental description. However, such a feature-level fusion paradigm with a single critic may fall short in policy learning as it tends to overlook the distinct values of each modality. To remedy this, this paper proposes a Local modality-customized Value Estimation (LVE) paradigm, which dynamically estimates the contribution and adjusts the importance weight of each modality from a value-level perspective. Furthermore, a task-contextual re-fusion process is developed to achieve a task-level re-balance of estimations from both feature and value levels. To this end, a Hierarchical Adaptive Value Estimation (HAVE) framework is formed, which adaptively coordinates the contributions of individual modalities as well as their collective efficacy. Agents trained by HAVE are able to exploit the unique characteristics of various modalities while capturing their intricate interactions, achieving substantially improved performance. We specifically highlight the potency of our approach within the challenging landscape of autonomous driving, utilizing the CARLA benchmark with neuromorphic event and depth data to demonstrate HAVE's capability and the effectiveness of its distinct components.

**Abstract(Chinese)**: 将RGB帧与替代性模态输入集成在许多基于视觉的强化学习（RL）应用中越来越受到关注。现有的多模态视觉RL方法通常遵循全局价值评估（GVE）流程，该流程使用融合的模态特征来获得统一的全局环境描述。然而，具有单一评论者的此类特征级融合范式在策略学习上可能存在不足，因为它倾向于忽视每个模态的独特价值。为了解决这个问题，本文提出了一种局部模态定制价值评估（LVE）范式，该范式从价值级的角度动态地估计贡献，并调整每种模态的重要性权重。此外，还开发了一种任务上下文重新融合过程，以实现来自特征和价值级别估计的任务级别重新平衡。为此，形成了一个分层自适应价值评估（HAVE）框架，该框架自适应地协调了每种模态的贡献以及它们的集体有效性。通过HAVE训练的代理能够利用各种模态的独特特征，同时捕捉它们的错综复杂的相互作用，从而实现了显著改进的性能。我们特别强调了我们的方法在自动驾驶这一具有挑战性的领域中的潜力，利用CARLA基准与神经形事件和深度数据来展示HAVE的能力以及其独特组件的有效性。

**URL**: https://nips.cc/virtual/2023/poster/70701

---

## Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems
**Author**: Sihan Zeng · Thinh Doan · Justin Romberg

**Abstract**: The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger “equiconnectedness” property. To our best knowledge, these are novel and previously unknown discoveries.We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting class of robust reinforcement learning problems under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.

**Abstract(Chinese)**: 本文旨在提高对增强学习中政策优化问题的优化景观的理解。具体地，我们展示了有关政策参数的超级水平集（superlevel set）始终是一个连接集，无论是在表格设置中还是在一类神经网络代表的策略下。此外，我们表明，与政策参数和奖励相关的优化目标满足更强的“等连接性”属性。据我们所知，这些都是新颖的以及以前未知的发现。我们应用这些超级水平集的连接性来推导鲁棒性增强学习的极小极大定理。我们表明，任何一边是凸的最小最大优化程序，而另一边满足等连接性的，都遵守最小最大等式（即具有纳什均衡）。我们发现，在对抗性奖励攻击下，这种确切结构被一类有趣的鲁棒性增强学习问题所展现，并且其最小最大等式的有效性也随之而来。这是文献中首次建立这样的结果。

**URL**: https://nips.cc/virtual/2023/poster/71398

---

## Online Nonstochastic Model-Free Reinforcement Learning
**Author**: Udaya Ghai · Arushi Gupta · Wenhan Xia · Karan Singh · Elad Hazan

**Abstract**: We investigate robust model-free reinforcement learning algorithms designed for environments that may be dynamic or even adversarial. Traditional state-based policies often struggle to accommodate the challenges imposed by the presence of unmodeled disturbances in such settings. Moreover, optimizing linear state-based policies pose an obstacle for efficient optimization, leading to nonconvex objectives, even in benign environments like linear dynamical systems.Drawing inspiration from recent advancements in model-based control, we intro- duce a novel class of policies centered on disturbance signals. We define several categories of these signals, which we term pseudo-disturbances, and develop corresponding policy classes based on them. We provide efficient and practical algorithms for optimizing these policies.Next, we examine the task of online adaptation of reinforcement learning agents in the face of adversarial disturbances. Our methods seamlessly integrate with any black-box model-free approach, yielding provable regret guarantees when dealing with linear dynamics. These regret guarantees unconditionally improve the best-known results for bandit linear control in having no dependence on the state-space dimension. We evaluate our method over various standard RL benchmarks and demonstrate improved robustness.

**Abstract(Chinese)**: 我们研究了针对可能是动态甚至敌对的环境而设计的健壮无模型强化学习算法。传统的基于状态的策略通常很难适应这些环境中未建模扰动带来的挑战。此外，优化线性状态策略在高效优化方面存在障碍，导致非凸目标，即使在线性动力系统等良性环境中也是如此。受最近模型控制的进展的启发，我们引入了一类以扰动信号为中心的新型策略。我们定义了几类这些信号，我们将其称为伪扰动，并基于它们开发对应的策略类。我们提供了优化这些策略的高效实用算法。接下来，我们研究了在面对敌对扰动时强化学习代理的在线适应任务。我们的方法可以无缝集成到任何黑盒无模型方法中，在处理线性动力学时，在可证明的遗憾保证方面产生了最佳结果的改进。这些遗憾保证无条件地提高了线性控制的最佳已知结果，而与状态空间维度无关。我们在各种标准强化学习基准上评估了我们的方法，并展示了改进的健壮性。

**URL**: https://nips.cc/virtual/2023/poster/72510

---

## Supervised Pretraining Can Learn In-Context Reinforcement Learning
**Author**: Jonathan Lee · Annie Xie · Aldo Pacchiano · Yash Chandak · Chelsea Finn · Ofir Nachum · Emma Brunskill

**Abstract**: Large transformer models trained on diverse datasets have shown a remarkable ability to learn in-context, achieving high few-shot performance on tasks they were not explicitly trained to solve. In this paper, we study the in-context learning capabilities of transformers in decision-making problems, i.e., reinforcement learning (RL) for bandits and Markov decision processes. To do so, we introduce and study the Decision-Pretrained Transformer (DPT), a supervised pretraining method where a transformer predicts an optimal action given a query state and an in-context dataset of interactions from a diverse set of tasks. While simple, this procedure produces a model with several surprising capabilities. We find that the trained transformer can solve a range of RL problems in-context, exhibiting both exploration online and conservatism offline, despite not being explicitly trained to do so. The model also generalizes beyond the pretraining distribution to new tasks and automatically adapts its decision-making strategies to unknown structure. Theoretically, we show DPT can be viewed as an efficient implementation of Bayesian posterior sampling, a provably sample-efficient RL algorithm. We further leverage this connection to provide guarantees on the regret of the in-context algorithm yielded by DPT, and prove that it can learn faster than algorithms used to generate the pretraining data. These results suggest a promising yet simple path towards instilling strong in-context decision-making abilities in transformers.

**Abstract(Chinese)**: 大型变压器模型在多样化数据集上训练，表现出了非凡的学习能力，能够在上下文中实现高效的少样本性能，在没有明确训练解决的任务上表现出色。在这篇论文中，我们研究了变压器在决策问题中的上下文学习能力，即强化学习（RL）用于赌博机和马尔科夫决策过程。为此，我们引入并研究了决策预训练变压器（DPT），这是一种监督预训练方法，其中变压器在给定查询状态和来自多样化任务集的上下文交互数据集的情况下预测最佳行动。尽管简单，这一过程产生了一个模型具有几个令人惊讶的能力。我们发现，经过训练的变压器可以在上下文中解决一系列RL问题，表现出在线探索和离线保守性，尽管并非明确训练为此。该模型还可以推广到超出预训练分布的新任务，并自动调整其决策策略以适应未知结构。从理论上讲，我们展示了DPT可以被视为贝叶斯后验采样的高效实现，即可证明的样本高效RL算法。我们进一步利用该连接以确保由DPT产生的上下文算法的遗憾，并证明它可以比用于生成预训练数据的算法更快地学习。这些结果表明了一种有前途但简单的途径，可以赋予变压器强大的上下文决策能力。

**URL**: https://nips.cc/virtual/2023/poster/71039

---

## Policy Optimization for Continuous Reinforcement Learning
**Author**: HANYANG ZHAO · Wenpin Tang · David Yao

**Abstract**: We study reinforcement learning (RL) in the setting of continuous time and space, for an infinite horizon with a discounted objective and the underlying dynamics driven by a stochastic differential equation. Built upon recent advances in the continuous approach to RL, we develop a notion of occupation time (specifically for a discounted objective),  and show how it can be effectively used to derive performance difference and local approximation formulas. We further extend these results to illustrate their applications in the PG (policy gradient) and TRPO/PPO (trust region policy optimization/ proximal policy optimization) methods,  which have been familiar and powerful tools in the discrete RL setting but under-developed in continuous RL. Through numerical experiments, we demonstrate the effectiveness and advantages of our approach.

**Abstract(Chinese)**: 我们研究连续时间和空间的强化学习（RL），对于具有折现目标的无限时间跨度，其基础动态由随机微分方程驱动。基于最近在连续RL方面的进展，我们发展了占用时间的概念（特别是对于折现目标），并展示了如何有效地用它来推导性能差异和局部逼近公式。我们进一步将这些结果扩展到政策梯度（PG）和TRPO/PPO（信任区域策略优化/近端策略优化）方法，这些方法在离散RL设置中已经是熟悉且强大的工具，但在连续RL中尚未得到充分开发。通过数值实验，我们展示了我们方法的有效性和优势。

**URL**: https://nips.cc/virtual/2023/poster/70304

---

## Offline Reinforcement Learning with Differential Privacy
**Author**: Dan Qiao · Yu-Xiang Wang

**Abstract**: The offline reinforcement learning (RL) problem is often motivated by the need to learn data-driven decision policies in financial, legal and healthcare applications.  However, the learned policy could retain sensitive information of individuals in the training data (e.g., treatment and outcome of patients), thus susceptible to various privacy risks. We design offline RL algorithms with differential privacy guarantees which provably prevent such risks. These algorithms also enjoy strong instance-dependent learning bounds under both tabular and linear Markov Decision Process (MDP) settings. Our theory and simulation suggest that the privacy guarantee comes at (almost) no drop in utility comparing to the non-private counterpart for a medium-size dataset.

**Abstract(Chinese)**: 离线强化学习（RL）问题通常是出于在金融、法律和医疗健康应用中学习数据驱动决策政策的需要。然而，所学习的策略可能会保留训练数据中个体的敏感信息（例如，患者的治疗和结果），因此容易受到各种隐私风险的影响。我们设计了具有差分隐私保证的离线RL算法，可以明显地预防这些风险。这些算法在表格式和线性马尔可夫决策过程（MDP）设定下也具有强的实例相关学习界限。我们的理论和模拟表明，隐私保证几乎不会降低中等规模数据集的效用，与非私有对应物相比。

**URL**: https://nips.cc/virtual/2023/poster/71294

---

## SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems
**Author**: Christopher Yeh · Victor Li · Rajeev Datta · Julio Arroyo · Nicolas Christianson · Chi Zhang · Yize Chen · Mohammad Mehdi Hosseini · Azarang Golmohammadi · Yuanyuan Shi · Yisong Yue · Adam Wierman

**Abstract**: The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts. In this paper, we present SustainGym, a suite of five environments designed to test the performance of RL algorithms on realistic sustainable energy system tasks, ranging from electric vehicle charging to carbon-aware data center job scheduling. The environments test RL algorithms under realistic distribution shifts as well as in multi-agent settings. We show that standard off-the-shelf RL algorithms leave significant room for improving performance and highlight the challenges ahead for introducing RL to real-world sustainability tasks.

**Abstract(Chinese)**: 缺乏可持续发展领域强化学习（RL）的标准化基准，使得很难在特定领域追踪进展并确定研究人员应该集中精力解决的瓶颈。在本文中，我们提出 SustainGym，这是一套设计用于测试RL算法在真实可持续能源系统任务上的表现的五个环境，涵盖从电动汽车充电到考虑碳排放的数据中心作业调度。这些环境在现实的分布转变以及多智能体设置下测试RL算法。我们展示了标准的现成RL算法在提高性能方面还有很大的空间，并强调引入RL到真实世界可持续发展任务面临的挑战。

**URL**: https://nips.cc/virtual/2023/poster/73430

---

## When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning
**Author**: Siliang Zeng · Chenliang Li · Alfredo Garcia · Mingyi Hong

**Abstract**: Offline inverse reinforcement learning (Offline IRL) aims to recover the structure of rewards and environment dynamics that underlie observed actions in a fixed, finite set of demonstrations from an expert agent. Accurate models of expertise in executing a task has applications in safety-sensitive applications such as clinical decision making and autonomous driving. However, the structure of an expert's preferences implicit in observed actions is closely linked to the expert's model of the environment dynamics (i.e. the ``world''). Thus, inaccurate models of the world obtained from finite data with limited coverage could compound inaccuracy in estimated rewards. To address this issue, we propose a bi-level optimization formulation of the estimation task wherein the upper level is likelihood maximization based upon a conservative model of the expert's policy (lower level). The policy model is conservative in that it maximizes reward subject to a penalty that is increasing in the uncertainty of the estimated model of the world. We propose a new algorithmic framework to solve the bi-level optimization problem formulation and provide statistical and computational guarantees of performance for the associated optimal reward estimator. Finally,  we demonstrate that the proposed algorithm outperforms the state-of-the-art offline IRL and imitation learning benchmarks by a large margin, over the continuous control tasks in MuJoCo and different datasets in the D4RL benchmark.

**Abstract(Chinese)**: 离线逆强化学习（离线IRL）旨在从专家代理的一组固定、有限的示范中恢复潜在的观察行为下的奖励和环境动态结构。在执行任务方面的专业模型具有诸如临床决策和自动驾驶等安全敏感应用。然而，观察行为中专家偏好的结构与专家对环境动态（即“世界”）的模型密切相关。因此，从有限数据中得到的不准确世界模型可能导致对奖励的估计不准确。为了解决这个问题，我们提出了一个估计任务的双层优化形式，在这种形式中，上层是基于专家策略（下层）的保守模型的可能性最大化。策略模型是保守的，因为它在最大化奖励的同时受到估计世界模型不确定性增加的惩罚。我们提出了一个新的算法框架来解决双层优化问题的形式，并为相关最优奖励估计器的性能提供了统计和计算保证。最后，我们证明了所提出的算法在MuJoCo的连续控制任务和D4RL基准测试中的不同数据集上远远胜过了最先进的离线IRL和模仿学习基准。

**URL**: https://nips.cc/virtual/2023/poster/70463

---

## Model-Free Active Exploration in Reinforcement Learning
**Author**: Alessio Russo · Alessio Russo · Alexandre Proutiere

**Abstract**: We study the problem of exploration in Reinforcement Learning and present a novel model-free solution. We adopt an information-theoretical viewpoint and start from the  instance-specific lower bound of the number of samples that have to be collected to identify a nearly-optimal policy. Deriving this lower bound along with the optimal exploration strategy entails solving an intricate optimization problem and requires a model of the system. In turn, most existing sample optimal exploration algorithms rely on estimating the model. We derive an approximation of the instance-specific lower bound that only involves quantities that can be inferred using model-free approaches. Leveraging this approximation, we devise an ensemble-based model-free exploration strategy  applicable to both tabular and continuous Markov decision processes. Numerical results demonstrate that our strategy is able to identify efficient policies faster than state-of-the-art exploration approaches.

**Abstract(Chinese)**: 我们研究了强化学习中的探索问题，并提出了一种新颖的无模型解决方案。我们采用信息理论的视角，从实例特定的样本数下限开始，以确定近乎最优策略所需收集的样本数量。推导这一下限以及最优探索策略需要解决一个复杂的优化问题，并需要系统模型。反过来，大多数现有的样本最优探索算法依赖于对模型的估计。我们推导出了实例特定下限的近似，这仅涉及可以使用无模型方法推断的数量。利用这个近似值，我们设计了一种基于集成的无模型探索策略，适用于表格和连续的马尔可夫决策过程。数值结果表明，我们的策略能够比最先进的探索方法更快地识别出高效的策略。

**URL**: https://nips.cc/virtual/2023/poster/71308

---

## Probabilistic Inference in Reinforcement Learning Done Right
**Author**: Jean Tarbouriech · Tor Lattimore · Brendan O'Donoghue

**Abstract**: A popular perspective in Reinforcement learning (RL) casts the problem as probabilistic inference on a graphical model of the Markov decision process (MDP). The core object of study is the probability of each state-action pair being visited under the optimal policy. Previous approaches to approximate this quantity can be arbitrarily poor, leading to algorithms that do not implement genuine statistical inference and consequently do not perform well in challenging problems. In this work, we undertake a rigorous Bayesian treatment of the posterior probability of state-action optimality and clarify how it flows through the MDP. We first reveal that this quantity can indeed be used to generate a policy that explores efficiently, as measured by regret. Unfortunately, computing it is intractable, so we derive a new variational Bayesian approximation yielding a tractable convex optimization problem and establish that the resulting policy also explores efficiently. We call our approach VAPOR and show that it has strong connections to Thompson sampling, K-learning, and maximum entropy exploration. We conclude with some experiments demonstrating the performance advantage of a deep RL version of VAPOR.

**Abstract(Chinese)**: 在强化学习（RL）中，一种常见的视角将问题视为马尔可夫决策过程（MDP）的图模型上的概率推理。研究的核心对象是在最优策略下访问每个状态-动作对的概率。先前的近似方法可能非常糟糕，导致算法无法实现真正的统计推断，因此在面临挑战性问题时表现不佳。在这项工作中，我们对状态-动作最优性的后验概率进行了严格的贝叶斯处理，并阐明了其在MDP中的传播方式。我们首先揭示了这一量确实可以用来生成一个有效探索的策略，通过后悔进行衡量。不幸的是，计算它是不可行的，因此我们推导了一种新的变分贝叶斯近似，产生了一个可行的凸优化问题，并建立了由此产生的策略也能够有效探索。我们称这种方法为VAPOR，并且展示它与汤普森抽样、K学习和最大熵探索有着密切的联系。最后我们通过一些实验展示了VAPOR的深度RL版本的性能优势。

**URL**: https://nips.cc/virtual/2023/poster/72569

---

## Provably Efficient Offline Reinforcement Learning in Regular Decision Processes
**Author**: Roberto Cipollone · Anders Jonsson · Alessandro Ronca · Mohammad Sadegh Talebi

**Abstract**: This paper deals with offline (or batch) Reinforcement Learning (RL) in episodic Regular Decision Processes (RDPs). RDPs are the subclass of Non-Markov Decision Processes where the dependency on the history of past events can be captured by a finite-state automaton. We consider a setting where the automaton that underlies the RDP is unknown, and a learner strives to learn a near-optimal policy using pre-collected data, in the form of non-Markov sequences of observations, without further exploration. We present RegORL, an algorithm that suitably combines automata learning techniques and state-of-the-art algorithms for offline RL in MDPs. RegORL has a modular design allowing one to use any off-the-shelf offline RL algorithm in MDPs. We report a non-asymptotic high-probability sample complexity bound for RegORL to yield an $\varepsilon$-optimal policy, which makes appear a notion of concentrability relevant for RDPs. Furthermore, we present a sample complexity lower bound for offline RL in RDPs. To our best knowledge, this is the first work presenting a provably efficient algorithm for offline learning in RDPs.

**Abstract(Chinese)**: 本文涉及离线（或批处理）强化学习（RL）在情境式常规决策过程（RDPs）中的应用。 RDPs是非马尔可夫决策过程的子类，其中对过去事件历史的依赖可以由有限状态自动机捕获。我们考虑一种情况，即情境式常规决策过程的基础自动机是未知的，学习者努力利用预先收集的非马尔可夫观测序列数据来学习一个接近最优策略，而无需进行进一步探索。 我们提出RegORL，一种适当结合自动机学习技术和MDP离线RL领域最先进算法的算法。 RegORL具有模块化设计，允许使用MDPs中的任何现成离线RL算法。我们报告了RegORL提供ε-最优策略的非渐近高概率样本复杂度界限，这使得RDPs相关的一种集中性概念显现出来。此外，我们提出了关于RDPs中离线RL的样本复杂度下界。据我们所知，这是首个在RDPs中呈现出可证有效的离线学习算法的工作。

**URL**: https://nips.cc/virtual/2023/poster/72637

---

## Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples
**Author**: Hao Sun · Alihan Hüyük · Daniel Jarrett · Mihaela van der Schaar

**Abstract**: Learning controllers with offline data in decision-making systems is an essential area of research due to its potential to reduce the risk of applications in real-world systems. However, in responsibility-sensitive settings such as healthcare, decision accountability is of paramount importance, yet has not been adequately addressed by the literature.This paper introduces the Accountable Offline Controller (AOC) that employs the offline dataset as the Decision Corpus and performs accountable control based on a tailored selection of examples, referred to as the Corpus Subset. AOC operates effectively in low-data scenarios, can be extended to the strictly offline imitation setting, and displays qualities of both conservation and adaptability.We assess AOC's performance in both simulated and real-world healthcare scenarios, emphasizing its capability to manage offline control tasks with high levels of performance while maintaining accountability.

**Abstract(Chinese)**: 在决策系统中利用离线数据学习控制器是一个重要的研究领域，因为它有潜力降低实际系统中应用的风险。然而，在诸如医疗保健等责任敏感的环境中，决策的问责是至关重要的，但文献尚未充分解决这一问题。本文介绍了负责任的离线控制器（AOC），它利用离线数据集作为决策语料库，并根据精心挑选的示例（称为语料库子集）执行问责控制。AOC在低数据场景下有效运作，可以扩展到严格的离线模仿设置，并具有保守性和适应性的特点。我们评估了AOC在模拟和真实医疗场景中的表现，强调其具有在高性能水平下管理离线控制任务并保持问责能力的特点。

**URL**: https://nips.cc/virtual/2023/poster/70631

---

## Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization
**Author**: Xiangsen Wang · Haoran Xu · Yinan Zheng · Xianyuan Zhan

**Abstract**: Offline reinforcement learning (RL) has received considerable attention in recent years due to its attractive capability of learning policies from offline datasets without environmental interactions. Despite some success in the single-agent setting, offline multi-agent RL (MARL) remains to be a challenge. The large joint state-action space and the coupled multi-agent behaviors pose extra complexities for offline policy optimization. Most existing offline MARL studies simply apply offline data-related regularizations on individual agents, without fully considering the multi-agent system at the global level. In this work, we present OMIGA, a new offline multi-agent RL algorithm with implicit global-to-local value regularization. OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. Based on comprehensive experiments on the offline multi-agent MuJoCo and StarCraft II micro-management tasks, we show that OMIGA achieves superior performance over the state-of-the-art offline MARL methods in almost all tasks.

**Abstract(Chinese)**: 离线强化学习（RL）近年来受到了广泛关注，因为它具有从离线数据集中学习策略而无需进行环境交互的吸引人的能力。尽管单一智能体设置取得了一些成功，离线多智能体RL（MARL）仍然是一个挑战。庞大的联合状态-动作空间和耦合的多智能体行为为离线策略优化增加了额外的复杂性。大多数现有的离线MARL研究简单地在个体智能体上应用与离线数据相关的正规化，而没有充分考虑整个多智能体系统。在这项工作中，我们提出了OMIGA，一种新的离线多智能体RL算法，具有隐式的全局到局部值正则化。OMIGA提供了一个原则性的框架，将全局级别的值正则化转换为等效的隐式局部值正则化，并同时实现样本内学习，从而优雅地连接多智能体值分解和离线正则化的策略学习。基于对离线多智能体MuJoCo和StarCraft II微观管理任务的全面实验，我们展示了OMIGA在几乎所有任务中都比最先进的离线MARL方法实现了更优越的性能。

**URL**: https://nips.cc/virtual/2023/poster/72476

---

## Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning
**Author**: Shenzhi Wang · Qisen Yang · Jiawei Gao · Matthieu Lin · HAO CHEN · Liwei Wu · Ning Jia · Shiji Song · Gao Huang

**Abstract**: Offline-to-online reinforcement learning (RL) is a training paradigm that combines pre-training on a pre-collected dataset with fine-tuning in an online environment. However, the incorporation of online fine-tuning can intensify the well-known distributional shift problem. Existing solutions tackle this problem by imposing a policy constraint on the policy improvement objective in both offline and online learning. They typically advocate a single balance between policy improvement and constraints across diverse data collections. This one-size-fits-all manner may not optimally leverage each collected sample due to the significant variation in data quality across different states. To this end, we introduce Family Offline-to-Online RL (FamO2O), a simple yet effective framework that empowers existing algorithms to determine state-adaptive improvement-constraint balances. FamO2O utilizes a universal model to train a family of policies with different improvement/constraint intensities, and a balance model to select a suitable policy for each state. Theoretically, we prove that state-adaptive balances are necessary for achieving a higher policy performance upper bound. Empirically, extensive experiments show that FamO2O offers a statistically significant improvement over various existing methods, achieving state-of-the-art performance on the D4RL benchmark. Codes are available at https://github.com/LeapLabTHU/FamO2O.

**Abstract(Chinese)**: 离线到在线强化学习（RL）是一种训练范式，它将在预先收集的数据集上进行预训练，然后在在线环境中进行微调。然而，在线微调的整合可能会加剧众所周知的分布偏移问题。现有解决方案通过对离线和在线学习中的策略改进目标施加策略约束来解决这一问题。它们通常倡导在不同的数据集上实施策略改进和约束之间的平衡。这种一刀切的方式可能无法充分利用每个收集的样本，因为不同状态下数据质量的显著变化。为此，我们介绍了Family Offline-to-Online RL（FamO2O），这是一个简单但有效的框架，赋予现有算法决定状态自适应改进约束平衡的能力。FamO2O利用通用模型训练了一系列具有不同改进/约束强度的策略，并利用平衡模型为每个状态选择适当的策略。从理论上讲，我们证明了状态自适应平衡对于实现更高的策略性能上界是必要的。从经验上看，大量实验表明FamO2O相对于各种现有方法都有着显著改善，实现了D4RL基准测试的最新性能。代码可在https://github.com/LeapLabTHU/FamO2O找到。

**URL**: https://nips.cc/virtual/2023/poster/70086

---

## ODE-based Recurrent Model-free Reinforcement Learning for POMDPs
**Author**: Xuanle Zhao · Duzhen Zhang · Han Liyuan · Tielin Zhang · Bo Xu

**Abstract**: Neural ordinary differential equations (ODEs) are widely recognized as the standard for modeling physical mechanisms, which help to perform approximate inference in unknown physical or biological environments. In partially observable (PO) environments, how to infer unseen information from raw observations puzzled the agents. By using a recurrent policy with a compact context, context-based reinforcement learning provides a flexible way to extract unobservable information from historical transitions. To help the agent extract more dynamics-related information, we present a novel ODE-based recurrent model combines with model-free reinforcement learning (RL) framework to solve partially observable Markov decision processes (POMDPs). We experimentally demonstrate the efficacy of our methods across various PO continuous control and meta-RL tasks. Furthermore, our experiments illustrate that our method is robust against irregular observations, owing to the ability of ODEs to model irregularly-sampled time series.

**Abstract(Chinese)**: 神经常微分方程(ODEs)被广泛认可为物理机制建模的标准，有助于在未知的物理或生物环境中执行近似推断。在部分可观测(PO)环境中，如何从原始观察中推断看不见的信息困扰着代理。通过使用带有紧凑上下文的循环策略，基于上下文的强化学习提供了一种灵活的方式，从历史转换中提取不可观测的信息。为了帮助代理提取更多与动态相关的信息，我们提出了一种新颖的基于ODE的循环模型，结合无模型强化学习(RL)框架来解决部分可观测马尔可夫决策过程(POMDPs)。我们通过实验证明了我们的方法在各种部分可观测连续控制和元强化学习任务中的有效性。此外，我们的实验表明，我们的方法对不规则观测具有鲁棒性，这归因于ODE能够对不规则采样的时间序列进行建模。

**URL**: https://nips.cc/virtual/2023/poster/71950

---

## Reinforcement Learning with Fast and Forgetful Memory
**Author**: Steven Morad · Ryan Kortvelesy · Stephan Liwicki · Amanda Prorok

**Abstract**: Nearly all real world tasks are inherently partially observable, necessitating the use of memory in Reinforcement Learning (RL). Most model-free approaches summarize the trajectory into a latent Markov state using memory models borrowed from Supervised Learning (SL), even though RL tends to exhibit different training and efficiency characteristics. Addressing this discrepancy, we introduce Fast and Forgetful Memory, an algorithm-agnostic memory model designed specifically for RL. Our approach constrains the model search space via strong structural priors inspired by computational psychology. It is a drop-in replacement for recurrent neural networks (RNNs) in recurrent RL algorithms, achieving greater reward than RNNs across various recurrent benchmarks and algorithms without changing any hyperparameters. Moreover, Fast and Forgetful Memory exhibits training speeds two orders of magnitude faster than RNNs, attributed to its logarithmic time and linear space complexity. Our implementation is available at https://github.com/proroklab/ffm.

**Abstract(Chinese)**: 几乎所有真实世界任务在本质上是部分可观察的，这要求在强化学习（RL）中使用记忆。大多数无模型方法将轨迹总结为潜在的马尔可夫状态，使用从监督学习（SL）借鉴的记忆模型，尽管RL倾向于表现出不同的训练和效率特性。为了解决这一矛盾，我们引入了快速遗忘记忆（Fast and Forgetful Memory），这是一种算法不可知的记忆模型，专门为RL设计。我们的方法通过受计算心理学启发的强结构先验来限制模型搜索空间。它可以作为循环RL算法中递归神经网络（RNNs）的即插即用替代品，在不改变任何超参数的情况下，比RNNs获得更大的奖励，且在各种递归基准和算法中的培训速度比RNNs快两个数量级，这归因于其对数时间和线性空间复杂度。我们的实现可在https://github.com/proroklab/ffm找到。

**URL**: https://nips.cc/virtual/2023/poster/72005

---

## COOM: A Game Benchmark for Continual Reinforcement Learning
**Author**: Tristan Tomilin · Meng Fang · Yudi Zhang · Mykola Pechenizkiy

**Abstract**: The advancement of continual reinforcement learning (RL) has been facing various obstacles, including standardized metrics and evaluation protocols, demanding computational requirements, and a lack of widely accepted standard benchmarks. In response to these challenges, we present COOM ($\textbf{C}$ontinual D$\textbf{OOM}$), a continual RL benchmark tailored for embodied pixel-based RL. COOM presents a meticulously crafted suite of task sequences set within visually distinct 3D environments, serving as a robust evaluation framework to assess crucial aspects of continual RL, such as catastrophic forgetting, knowledge transfer, and sample-efficient learning. Following an in-depth empirical evaluation of popular continual learning (CL) methods, we pinpoint their limitations, provide valuable insight into the benchmark and highlight unique algorithmic challenges. This makes our work the first to benchmark image-based CRL in 3D environments with embodied perception. The primary objective of the COOM benchmark is to offer the research community a valuable and cost-effective challenge. It seeks to deepen our comprehension of the capabilities and limitations of current and forthcoming CL methods in an RL setting. The code and environments are open-sourced and accessible on GitHub.

**Abstract(Chinese)**: 摘要：持续强化学习（RL）的发展面临着多种障碍，包括标准化的度量标准和评估协议、需求的计算要求以及缺乏被广泛接受的标准基准。为了应对这些挑战，我们提出了COOM（$	extbf{C}$ontinual D$	extbf{OOM}$），这是一个专为基于像素的持续RL而定制的基准。COOM提供了一套精心设计的任务序列，设置在视觉上有别的3D环境中，作为评估持续RL关键方面的强大评估框架，如灾难性遗忘、知识转移和样本高效学习。通过对流行的持续学习（CL）方法进行深入的实证评估，我们指出了它们的局限性，为基准提供了宝贵的见解，并突出了独特的算法挑战。这使得我们的工作成为第一个在具有体验感知的3D环境中对基于图像的CRL进行基准测试的工作。COOM基准的主要目标是为研究社区提供一个有价值且成本效益的挑战。它旨在加深我们对当前和未来CL方法在RL环境中的能力和局限性的理解。代码和环境是开源的，可以在GitHub上获取。

**URL**: https://nips.cc/virtual/2023/poster/73450

---

## Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning
**Author**: Siming Lan · Rui Zhang · Qi Yi · Jiaming Guo · Shaohui Peng · Yunkai Gao · Fan Wu · Ruizhi Chen · Zidong Du · Xing Hu · xishan zhang · Ling Li · Yunji Chen

**Abstract**: In the field of multi-task reinforcement learning, the modular principle, which involves specializing functionalities into different modules and combining them appropriately, has been widely adopted as a promising approach to prevent the negative transfer problem that performance degradation due to conflicts between tasks. However, most of the existing multi-task RL methods only combine shared modules at the task level, ignoring that there may be conflicts within the task. In addition, these methods do not take into account that without constraints, some modules may learn similar functions, resulting in restricting the model's expressiveness and generalization capability of modular methods.In this paper, we propose the Contrastive Modules with Temporal Attention(CMTA) method to address these limitations. CMTA constrains the modules to be different from each other by contrastive learning and combining shared modules at a finer granularity than the task level with temporal attention, alleviating the negative transfer within the task and improving the generalization ability and the performance for multi-task RL.We conducted the experiment on Meta-World, a multi-task RL benchmark containing various robotics manipulation tasks. Experimental results show that CMTA outperforms learning each task individually for the first time and achieves substantial performance improvements over the baselines.

**Abstract(Chinese)**: 在多任务强化学习领域，模块化原则被广泛采用作为一种有希望的方法来防止任务之间的负面迁移问题，该原则涉及将功能专门化为不同的模块，并适当地组合它们。然而，大多数现有的多任务强化学习方法仅在任务级别上组合共享模块，忽视了任务内部可能存在的冲突。此外，这些方法未考虑没有约束时，某些模块可能学习类似的功能，从而限制了模块化方法的表达能力和泛化能力。在本文中，我们提出了对比模块与时间注意力（CMTA）方法来解决这些限制。CMTA通过对比学习约束模块之间的差异，并通过时间注意力在比任务更细粒度的层面上组合共享模块，减轻了任务内部的负面迁移，并提高了多任务强化学习的泛化能力和性能。我们在Meta-World上进行了实验，这是一个包含各种机器人操作任务的多任务强化学习基准。实验结果表明，CMTA首次在性能上超越了单独学习每个任务，并且比基线方法取得了显著的性能改进。

**URL**: https://nips.cc/virtual/2023/poster/71415

---

## Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies
**Author**: Michael Beukman · Devon Jarvis · Richard Klein · Steven James · Benjamin Rosman

**Abstract**: While reinforcement learning has achieved remarkable successes in several domains, its real-world application is limited due to many methods failing to generalise to unfamiliar conditions. In this work, we consider the problem of generalising to new transition dynamics, corresponding to cases in which the environment's response to the agent's actions differs. For example, the gravitational force exerted on a robot depends on its mass and changes the robot's mobility. Consequently, in such cases, it is necessary to condition an agent's actions on extrinsic state information and pertinent contextual information reflecting how the environment responds. While the need for context-sensitive policies has been established, the manner in which context is incorporated architecturally has received less attention. Thus, in this work, we present an investigation into how context information should be incorporated into behaviour learning to improve generalisation.  To this end, we introduce a neural network architecture, the Decision Adapter, which generates the weights of an adapter module and conditions the behaviour of an agent on the context information. We show that the Decision Adapter is a useful generalisation of a previously proposed architecture and empirically demonstrate that it results in superior generalisation performance compared to previous approaches in several environments. Beyond this, the Decision Adapter is more robust to irrelevant distractor variables than several alternative methods.

**Abstract(Chinese)**: 虽然强化学习在几个领域取得了显著成功，但由于许多方法无法推广到陌生条件，其真实世界的应用受到限制。在这项工作中，我们考虑了概括到新的转换动态的问题，这些问题对应于环境对代理行为的响应不同的情况。例如，施加在机器人上的重力依赖于其质量，并改变了机器人的机动性。因此，在这种情况下，有必要将代理的行为条件化为外部状态信息和反映环境响应方式的相关背景信息。虽然已经确定了需要上下文敏感的策略，但架构上如何纳入上下文的方式却受到了较少的关注。因此，在这项工作中，我们提出了一个调查，探讨了上下文信息如何被纳入到行为学习中以改进泛化性能。为此，我们引入了一个神经网络架构，决策适配器，它生成适配器模块的权重，并根据上下文信息对代理的行为进行条件化。我们证明了决策适配器是先前提出的架构的有用泛化，并通过实证表明，在几种环境中，它比以往方法具有更优越的泛化性能。除此之外，决策适配器比几种备选方法更能抵抗无关的分心变量。

**URL**: https://nips.cc/virtual/2023/poster/71771

---

## Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach
**Author**: Yudi Zhang · Yali Du · Biwei Huang · Ziyan Wang · Jun Wang · Meng Fang · Mykola Pechenizkiy

**Abstract**: A major challenge in reinforcement learning is to determine which state-action pairs are responsible for future rewards that are delayed. Reward redistribution serves as a solution to re-assign credits for each time step from observed sequences.  While the majority of current approaches construct the reward redistribution in an uninterpretable manner, we propose to explicitly model the contributions of state and action from a causal perspective, resulting in an interpretable reward redistribution and preserving policy invariance. In this paper, we start by studying the role of causal generative models in reward redistribution by characterizing the generation of Markovian rewards and trajectory-wise long-term return and further propose a framework, called Generative Return Decomposition (GRD), for policy optimization in delayed reward scenarios. Specifically, GRD first identifies the unobservable Markovian rewards and causal relations in the generative process. Then,  GRD makes use of the identified causal generative model to form a compact representation to train policy over the most favorable subspace of the state space of the agent. Theoretically, we show that the unobservable Markovian reward function is identifiable, as well as the underlying causal structure and causal models. Experimental results show that our method outperforms state-of-the-art methods and the provided visualization further demonstrates the interpretability of our method.The project page is located at https://reedzyd.github.io/GenerativeReturnDecomposition/.

**Abstract(Chinese)**: 在强化学习中的一个主要挑战是确定哪些状态-动作对应于延迟的未来奖励。奖励再分配作为一种解决方案，用于从观察到的序列中重新分配每个时间步骤的学分。虽然目前的大部分方法构建了难以解释的奖励再分配，我们提出从因果的角度明确建模状态和动作的贡献，从而实现可解释的奖励再分配并保持政策不变性。在本文中，我们首先研究因果生成模型在奖励再分配中的作用，通过表征马尔可夫奖励的生成和轨迹长期回报，并进一步提出了一个名为生成回报分解（GRD）的框架，用于延迟奖励情境下的策略优化。具体而言，GRD首先确定了不可观测的马尔可夫奖励和生成过程中的因果关系。然后，GRD利用识别的因果生成模型形成一个紧凑的表示，以在代理的状态空间的最有利子空间上训练策略。从理论上讲，我们展示了不可观测的马尔可夫奖励函数是可识别的，以及潜在的因果结构和因果模型。实验结果表明，我们的方法优于当前最先进的方法，提供的可视化进一步证明了我们的方法的可解释性。项目页面位于https://reedzyd.github.io/GenerativeReturnDecomposition/。

**URL**: https://nips.cc/virtual/2023/poster/70073

---

## Kernelized Reinforcement Learning with Order Optimal Regret Bounds
**Author**: Sattar Vakili · Julia Olkhovskaya

**Abstract**: Modern reinforcement learning (RL) has shown empirical success in various real world settings with complex models and large state-action spaces. The existing analytical results, however, typically focus on settings with a small number of state-actions or simple models such as linearly modeled state-action value functions. To derive RL policies that efficiently handle large state-action spaces with more general value functions, some recent works have considered nonlinear function approximation using kernel ridge regression. We propose $\pi$-KRVI, an optimistic modification of least-squares value iteration, when the action-value function is represented by an RKHS. We prove the first order-optimal regret guarantees under a general setting. Our results show a significant polynomial in the number of episodes improvement over the state of the art. In particular, with highly non-smooth kernels (such as Neural Tangent kernel or some Matérn kernels) the existing results lead to trivial (superlinear in the number of episodes) regret bounds. We show a sublinear regret bound that is order optimal in the cases where a lower bound on regret is known (which includes the kernels mentioned above).

**Abstract(Chinese)**: 现代强化学习（RL）已经在各种复杂模型和大状态动作空间的实际场景中取得了经验成功。但是现有的分析结果通常集中在状态动作数量较小或者简单模型（如线性模拟的状态动作价值函数）的情况下。为了得到能高效处理大状态动作空间和更一般价值函数的RL策略，一些最近的研究考虑使用核岭回归进行非线性函数逼近。我们提出了$\pi$-KRVI，这是对最小二乘值迭代的乐观修改，当动作值函数由RKHS表示时。我们在一般环境下证明了首次一阶最优遗憾保证。我们的结果显示在周期数的数量上，相比于现有技术，取得了显著的多项式改进。特别是，对于高度非光滑的核（如神经切向核或一些Matérn核），现有结果导致微不足道的（超线性的周期数）遗憾界。我们展示了一个亚线性的遗憾界，在已知遗憾下限的情况下是优化顺序的，这包括上述提到的核。

**URL**: https://nips.cc/virtual/2023/poster/70396

---

## The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning
**Author**: Kaiwen Wang · Kevin Zhou · Runzhe Wu · Nathan Kallus · Wen Sun

**Abstract**: While distributional reinforcement learning (DistRL) has been empirically effective, the question of when and why it is better than vanilla, non-distributional RL has remained unanswered.This paper explains the benefits of DistRL through the lens of small-loss bounds, which are instance-dependent bounds that scale with optimal achievable cost.Particularly, our bounds converge much faster than those from non-distributional approaches if the optimal cost is small.As warmup, we propose a distributional contextual bandit (DistCB) algorithm, which we show enjoys small-loss regret bounds and empirically outperforms the state-of-the-art on three real-world tasks.In online RL, we propose a DistRL algorithm that constructs confidence sets using maximum likelihood estimation. We prove that our algorithm enjoys novel small-loss PAC bounds in low-rank MDPs.As part of our analysis, we introduce the $\ell_1$ distributional eluder dimension which may be of independent interest. Then, in offline RL, we show that pessimistic DistRL enjoys small-loss PAC bounds that are novel to the offline setting and are more robust to bad single-policy coverage.

**Abstract(Chinese)**: 摘要：虽然分布式强化学习（DistRL）在经验上是有效的，但它比普通的非分布式RL更好的时机和原因仍然没有答案。本文通过小损失边界的视角解释了DistRL的好处，这些边界是随最优成本可达到的范围而变化的。特别地，我们的边界在最优成本较小时比非分布式方法快得多。作为铺垫，我们提出了一种分布式上下文强化学习（DistCB）算法，我们展示了它享有小损失后悔边界，并在三个真实世界任务上在经验上优于现有技术。在在线RL中，我们提出了一种DistRL算法，它使用最大似然估计构建置信集。我们证明了我们的算法在低秩MDPs中享有新颖的小损失PAC边界。作为我们分析的一部分，我们介绍了可能引起独立兴趣的l1分布式躲避者维度。然后在离线RL中，我们展示了悲观的DistRL享有小损失PAC边界，这些边界对离线设置是新颖的，并且对单策略覆盖不佳更加稳健。

**URL**: https://nips.cc/virtual/2023/poster/71631

---

## Information Design in Multi-Agent Reinforcement Learning
**Author**: Yue Lin · Wenhao Li · Hongyuan Zha · Baoxiang Wang

**Abstract**: Reinforcement learning (RL) is inspired by the way human infants and animals learn from the environment. The setting is somewhat idealized because, in actual tasks, other agents in the environment have their own goals and behave adaptively to the ego agent. To thrive in those environments, the agent needs to influence other agents so their actions become more helpful and less harmful. Research in computational economics distills two ways to influence others directly: by providing tangible goods (mechanism design) and by providing information (information design). This work investigates information design problems for a group of RL agents. The main challenges are two-fold. One is the information provided will immediately affect the transition of the agent trajectories, which introduces additional non-stationarity. The other is the information can be ignored, so the sender must provide information that the receiver is willing to respect. We formulate the Markov signaling game, and develop the notions of signaling gradient and the extended obedience constraints that address these challenges. Our algorithm is efficient on various mixed-motive tasks and provides further insights into computational economics. Our code is publicly available at https://github.com/YueLin301/InformationDesignMARL.

**Abstract(Chinese)**: 强化学习（RL）受到人类婴儿和动物从环境中学习的启发。这种设置有些理想化，因为在实际任务中，环境中的其他代理有自己的目标，并适应性地行为。为了在这些环境中茁壮成长，代理需要影响其他代理，使它们的行为更有帮助性，较少有害性。计算经济学的研究概括了直接影响他人的两种方式：通过提供有形商品（机制设计）和提供信息（信息设计）。本工作研究了一组RL代理的信息设计问题。主要挑战有两个。一个是提供的信息会立即影响代理轨迹的转变，这引入了额外的非稳态性。另一个是信息可能会被忽略，因此发送者必须提供接收者愿意尊重的信息。我们制定了马尔可夫信令博弈，并发展了信令梯度和扩展服从约束的概念来解决这些挑战。我们的算法在各种混合动机任务上效率高，并提供了对计算经济学的进一步见解。我们的代码可公开获取，网址为https://github.com/YueLin301/InformationDesignMARL。

**URL**: https://nips.cc/virtual/2023/poster/71832

---

## DIFFER:Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning
**Author**: Xunhan Hu · Jian Zhao · Wengang Zhou · Ruili Feng · Houqiang Li

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) is a challenging task, as agents must learn complex and diverse individual strategies from a shared team reward. However, existing methods struggle to distinguish and exploit important individual experiences, as they lack an effective way to decompose the team reward into individual rewards. To address this challenge, we propose DIFFER, a powerful theoretical framework for decomposing individual rewards to enable fair experience replay in MARL.By enforcing the invariance of network gradients, we establish a partial differential equation whose solution yields the underlying individual reward function. The individual TD-error can then be computed from the solved closed-form individual rewards, indicating the importance of each piece of experience in the learning task and guiding the training process. Our method elegantly achieves an equivalence to the original learning framework when individual experiences are homogeneous, while also adapting to achieve more muscular efficiency and fairness when diversity is observed.Our extensive experiments on popular benchmarks validate the effectiveness of our theory and method, demonstrating significant improvements in learning efficiency and fairness. Code is available in supplement material.

**Abstract(Chinese)**: 合作式多智能体强化学习（MARL）是一项具有挑战性的任务，因为智能体必须从共享的团队奖励中学习复杂和多样化的个体策略。然而，现有的方法很难区分和利用重要的个体经验，因为它们缺乏将团队奖励分解为个体奖励的有效方法。为了解决这一挑战，我们提出了DIFFER，这是一个用于分解个体奖励以实现公平经验回放的强大理论框架。通过强制网络梯度的不变性，我们建立了一个偏微分方程，其解决方案产生了潜在的个体奖励函数。然后可以从解决的封闭形式个体奖励中计算个体TD误差，指示学习任务中每个经验片段的重要性并指导训练过程。当个体经验是同质的时，我们的方法优雅地实现了与原始学习框架的等价性，同时也可以适应观察到的更多样化的更高效和更公平的情况。我们在流行的基准测试上进行了大量实验证实了我们的理论和方法的有效性，展示了学习效率和公平性的显著改进。代码可以在附加材料中找到。

**URL**: https://nips.cc/virtual/2023/poster/72548

---

## Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates
**Author**: Sarath Sreedharan · Michael Katz

**Abstract**: There has been an increasing interest in using symbolic models along with reinforcement learning (RL) problems, where these coarser abstract models are used as a way to provide RL agents with higher level guidance. However, most of these works are inherently limited by their assumption of having an access to a symbolic approximation of the underlying problem. To address this issue, we introduce a new method for learning optimistic symbolic approximations of the underlying world model. We will see how these representations, coupled with fast diverse planners developed by the automated planning community, provide us with a new paradigm for optimistic exploration in sparse reward settings. We investigate the possibility of speeding up the learning process by generalizing learned model dynamics across similar actions with minimal human input. Finally, we evaluate the method, by testing it on multiple benchmark domains and compare it with other RL strategies.

**Abstract(Chinese)**: 在使用符号模型和强化学习（RL）问题方面，人们对使用这些更为粗糙的抽象模型来给RL代理提供更高级别指导的兴趣日益增长。然而，大多数这些工作都因其假设可以访问潜在问题的符号近似而天然受到限制。为了解决这个问题，我们引入了一种新的方法来学习潜在世界模型的乐观符号近似。我们将看到这些表示与自动化规划社区开发的快速多样化规划者相结合，为我们提供了一种在稀疏奖励设置中进行乐观探索的新范式。我们研究通过在最小人类输入下跨相似动作泛化学习模型动态来加速学习过程的可能性。最后，我们通过在多个基准领域上进行测试并将其与其他RL策略进行比较来评估该方法。

**URL**: https://nips.cc/virtual/2023/poster/71595

---

## Percentile Criterion Optimization in Offline Reinforcement Learning
**Author**: Cyrus Cousins · Elita Lobo · Marek Petrik · Yair Zick

**Abstract**: In reinforcement learning, robust policies for high-stakes decision-making problems with limited data are usually computed by optimizing the percentile criterion. The percentile criterion is optimized by constructing an uncertainty set that contains the true model with high probability and optimizing the policy for the worst model in the set. Since the percentile criterion is non-convex, constructing these sets itself is challenging. Existing works use Bayesian credible regions as uncertainty sets, but they are often unnecessarily large and result in learning overly conservative policies. To overcome these shortcomings, we propose a novel Value-at-Risk based dynamic programming algorithm to optimize the percentile criterion without explicitly constructing any uncertainty sets. Our theoretical and empirical results show that our algorithm implicitly constructs much smaller uncertainty sets and learns less-conservative robust policies.

**Abstract(Chinese)**: 在强化学习中，通常通过优化百分位准则来计算有限数据下高风险决策问题的稳健策略。 百分位准则通过构建包含真实模型的不确定性集合来进行优化，该集合具有较高的概率，并针对集合中最差的模型来优化策略。 由于百分位准则是非凸的，构建这些集合本身就是具有挑战性的。 现有作品使用贝叶斯可信区间作为不确定性集合，但它们通常过大，导致学习出过于保守的策略。 为了克服这些缺点，我们提出了一种新的基于风险价值的动态规划算法，以优化百分位准则，而无需显式构建任何不确定性集合。 我们的理论和实证结果表明，我们的算法隐式构建了更小的不确定性集，并学习到了更少保守的稳健策略。

**URL**: https://nips.cc/virtual/2023/poster/71800

---

## Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning
**Author**: James Queeney · Mouhacine Benosman

**Abstract**: Many real-world domains require safe decision making in uncertain environments. In this work, we introduce a deep reinforcement learning framework for approaching this important problem. We consider a distribution over transition models, and apply a risk-averse perspective towards model uncertainty through the use of coherent distortion risk measures. We provide robustness guarantees for this framework by showing it is equivalent to a specific class of distributionally robust safe reinforcement learning problems. Unlike existing approaches to robustness in deep reinforcement learning, however, our formulation does not involve minimax optimization. This leads to an efficient, model-free implementation of our approach that only requires standard data collection from a single training environment. In experiments on continuous control tasks with safety constraints, we demonstrate that our framework produces robust performance and safety at deployment time across a range of perturbed test environments.

**Abstract(Chinese)**: 许多现实世界的领域需要在不确定的环境中进行安全决策。在这项工作中，我们引入了一种深度强化学习框架，以解决这个重要问题。我们考虑了过渡模型的分布，并通过使用连贯的畸变风险度量，对模型的不确定性采取风险厌恶的观点。我们通过展示该框架等价于一类特定的分布鲁棒安全强化学习问题，为该框架提供了健壮性保证。然而，与深度强化学习中现有的健壮性方法不同，我们的公式不涉及极小化最大化优化。这导致我们的方法的高效、无模型实现，只需要从单一训练环境中进行标准数据收集。在具有安全约束的连续控制任务的实验中，我们证明了我们的框架在各种扰动测试环境中都能产生强大的性能和安全性。

**URL**: https://nips.cc/virtual/2023/poster/71384

---

## Train Hard, Fight Easy: Robust Meta Reinforcement Learning
**Author**: Ido Greenberg · Shie Mannor · Gal Chechik · Eli Meirom

**Abstract**: A major challenge of reinforcement learning (RL) in real-world applications is the variation between environments, tasks or clients. Meta-RL (MRL) addresses this issue by learning a meta-policy that adapts to new tasks. Standard MRL methods optimize the average return over tasks, but often suffer from poor results in tasks of high risk or difficulty. This limits system reliability since test tasks are not known in advance. In this work, we define a robust MRL objective with a controlled robustness level. Optimization of analogous robust objectives in RL is known to lead to both biased gradients and data inefficiency. We prove that the gradient bias disappears in our proposed MRL framework. The data inefficiency is addressed via the novel Robust Meta RL algorithm (RoML). RoML is a meta-algorithm that generates a robust version of any given MRL algorithm, by identifying and over-sampling harder tasks throughout training. We demonstrate that RoML achieves robust returns on multiple navigation and continuous control benchmarks.

**Abstract(Chinese)**: 在强化学习（RL）在现实世界应用中的一个主要挑战是环境、任务或客户之间的差异。元强化学习（MRL）通过学习适应新任务的元策略来解决这个问题。标准的MRL方法优化任务的平均回报，但在高风险或难度较大的任务中往往效果不佳。这限制了系统的可靠性，因为测试任务事先未知。在这项工作中，我们定义了一个稳健的MRL目标，具有受控的稳健性水平。已知在RL中优化类似的稳健目标会导致偏置梯度和数据效率低下。我们证明在我们提出的MRL框架中梯度偏差消失了。通过新颖的稳健元强化学习算法（RoML），解决了数据效率低下的问题。RoML是一种元算法，通过在训练过程中识别和过采样更难的任务，生成任何给定MRL算法的稳健版本。我们证明RoML在多个导航和连续控制基准上实现了稳健的回报。

**URL**: https://nips.cc/virtual/2023/poster/72040

---

## StateMask: Explaining Deep Reinforcement Learning through State Mask
**Author**: Zelei Cheng · Xian Wu · Jiahao Yu · Wenhai Sun · Wenbo Guo · Wenbo Guo · Xinyu Xing

**Abstract**: Despite the promising performance of deep reinforcement learning (DRL) agents in many challenging scenarios, the black-box nature of these agents greatly limits their applications in critical domains. Prior research has proposed several explanation techniques to understand the deep learning-based policies in RL. Most existing methods explain why an agent takes individual actions rather than pinpointing the critical steps to its final reward. To fill this gap, we propose StateMask, a novel method to identify the states most critical to the agent's final reward. The high-level idea of StateMask is to learn a mask net that blinds a target agent and forces it to take random actions at some steps without compromising the agent's performance. Through careful design, we can theoretically ensure that the masked agent performs similarly to the original agent. We evaluate StateMask in various popular RL environments and show its superiority over existing explainers in explanation fidelity. We also show that StateMask  has better utilities, such as launching adversarial attacks and patching policy errors.

**Abstract(Chinese)**: 尽管深度强化学习（DRL）代理在许多具有挑战性的场景中表现出了很好的性能，但是这些代理的黑盒特性极大地限制了它们在关键领域的应用。先前的研究提出了几种解释技术来理解深度学习 based 策略在 RL 中的作用。大多数现有的方法解释了代理为什么采取个别动作，而不是准确指出到其最终奖励的关键步骤。为填补此差距，我们提出 StateMask，这是一种新颖的方法，用于识别对代理最终奖励至关重要的状态。StateMask 的高层理念是学习一个遮罩网络，可以使目标代理失明，并迫使其在某些步骤上采取随机动作，而不损害代理的性能。通过精心设计，我们可以理论上确保被遮蔽的代理的性能与原始代理类似。我们在各种流行的 RL 环境中评估了 StateMask，并展示了它在解释的忠实度方面的优越性。我们还展示了 StateMask 具有更好的效用，例如启动对抗性攻击和修补策略错误。

**URL**: https://nips.cc/virtual/2023/poster/70386

---

## When is Agnostic Reinforcement Learning Statistically Tractable?
**Author**: Zeyu Jia · Gene Li · Alexander Rakhlin · Ayush Sekhari · Nati Srebro

**Abstract**: We study the problem of agnostic PAC reinforcement learning (RL): given a policy class $\Pi$, how many rounds of interaction with an unknown MDP (with a potentially large state and action space) are required to learn an $\epsilon$-suboptimal policy with respect to \(\Pi\)? Towards that end, we introduce a new complexity measure, called the \emph{spanning capacity}, that depends solely on the set \(\Pi\) and is independent of the MDP dynamics. With a generative model, we show that the spanning capacity characterizes PAC learnability for every policy class $\Pi$. However, for online RL, the situation is more subtle. We show there exists a policy class $\Pi$ with a bounded spanning capacity that requires a superpolynomial number of samples to learn. This reveals a surprising separation for agnostic learnability between generative access and online access models (as well as between deterministic/stochastic MDPs under online access). On the positive side, we identify an additional \emph{sunflower} structure which in conjunction with bounded spanning capacity enables statistically efficient online RL via a new algorithm called POPLER, which takes inspiration from classical importance sampling methods as well as recent developments for reachable-state identification and policy evaluation in reward-free exploration.

**Abstract(Chinese)**: 我们研究了对独立的PAC增强学习（RL）问题：给定一个策略集 $\Pi$，与未知的MDP（可能具有大状态和动作空间）进行多少轮交互才能学习到一个关于 $\Pi$ 的 $\epsilon$-次优策略？为此，我们引入一个称为\emph{跨越容量}的新复杂度度量，它仅依赖于集合 $\Pi$，并且独立于MDP的动态。通过一个生成模型，我们展示了跨越容量刻画了每个策略类 $\Pi$ 的PAC可学习性。然而，对于在线RL，情况就更加微妙。我们展示了存在一个具有有界跨越容量的策略类 $\Pi$，需要超多项式数量的样本来学习。这揭示了在生成访问和在线访问模型之间（以及在线访问下确定性/随机MDP之间）对于独立可学习性的一个惊人分离。积极的一面是，我们确定了一个额外的\emph{向日葵}结构，它与有界跨越容量一起通过一种名为POPLER的新算法实现了统计高效的在线RL，这个算法借鉴了经典重要性抽样方法以及最近为无奖励探索的可达状态识别和策略评估的发展。

**URL**: https://nips.cc/virtual/2023/poster/72745

---

## The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model
**Author**: Laixi Shi · Gen Li · Yuting Wei · Yuxin Chen · Matthieu Geist · Yuejie Chi

**Abstract**: This paper investigates model robustness in reinforcement learning (RL) via the framework of distributionally robust Markov decision processes (RMDPs). Despite recent efforts, the sample complexity of RMDPs is much less understood regardless of the uncertainty set in use; in particular, there exist large gaps between existing upper and lower bounds, and it is unclear if distributional robustness bears any statistical implications when benchmarked against standard RL. In this paper, assuming access to a generative model, we derive the sample complexity of RMDPs---when the uncertainty set is measured via either total variation or $\chi^2$ divergence over the full range of uncertainty levels---using a model-based algorithm called distributionally robust value iteration, and develop  minimax lower bounds to benchmark its tightness. Our results not only strengthen the prior art in both directions of upper and lower bounds, but also deliver surprising messages that learning RMDPs is not necessarily easier or more difficult than standard MDPs. In the case of total variation, we establish the minimax-optimal sample complexity of RMDPs which is always smaller than that of standard MDPs. In the case of $\chi^2$ divergence, we establish the sample complexity of RMDPs that is tight up to polynomial factors of the effective horizon, and grows linearly with respect to the uncertainty level when it approaches infinity.

**Abstract(Chinese)**: 本文通过分布式鲁棒马尔可夫决策过程（RMDP）的框架，探讨了强化学习（RL）中模型的鲁棒性。尽管最近进行了努力，但是无论使用的不确定性集合是什么，RMDP的样本复杂性都了解得不多；尤其需要指出的是，现有上下界之间存在较大的差距，并且不清楚分布式鲁棒性在统计学意义上是否对标准RL产生任何影响。在本文中，假设可以访问一个生成模型，我们使用模型为基础的算法——分布式鲁棒值迭代，根据通过总变差或$\chi^2$散度测量的不确定性水平范围内的样本复杂性进行推导，并开发了用于检验其紧密性的极小化下界。我们的结果不仅加强了先前在上下界方向的艺术水平，而且传递了一个令人意外的信息，即学习RMDP不一定比标准MDP更容易或更困难。在总变差的情况下，我们建立了RMDP的极小化样本复杂性，始终小于标准MDP的样本复杂性。在$\chi^2$散度的情况下，我们建立了RMDP的样本复杂度，它与有效时间跨度的多项式因子相关，并且随着不确定性水平无限增长而线性增长。

**URL**: https://nips.cc/virtual/2023/poster/71095

---

## De novo Drug Design using Reinforcement Learning with Multiple GPT Agents
**Author**: Xiuyuan Hu · Guoqing Liu · Yang Zhao · Hao Zhang

**Abstract**: De novo drug design is a pivotal issue in pharmacology and a new area of focus in AI for science research. A central challenge in this field is to generate molecules with specific properties while also producing a wide range of diverse candidates. Although advanced technologies such as transformer models and reinforcement learning have been applied in drug design, their potential has not been fully realized. Therefore, we propose MolRL-MGPT, a reinforcement learning algorithm with multiple GPT agents for drug molecular generation. To promote molecular diversity, we encourage the agents to collaborate in searching for desirable molecules in diverse directions. Our algorithm has shown promising results on the GuacaMol benchmark and exhibits efficacy in designing inhibitors against SARS-CoV-2 protein targets. The codes are available at: https://github.com/HXYfighter/MolRL-MGPT.

**Abstract(Chinese)**: 新药设计是药理学中的一个关键问题，也是人工智能科学研究中的一个新的关注领域。这一领域的一个核心挑战是在生成具有特定性质的分子的同时，也能够产生多样化的候选分子。尽管先进技术如变压器模型和强化学习已被应用于药物设计，但其潜力尚未完全发挥。因此，我们提出了MolRL-MGPT，这是一个具有多个GPT代理的强化学习算法，用于药物分子生成。为了促进分子多样性，我们鼓励代理人合作，在不同方向寻找理想的分子。我们的算法在GuacaMol基准测试中展现出了有希望的结果，并在设计抗击SARS-CoV-2蛋白靶标的抑制剂方面表现出了有效性。代码可在以下网址获取: https://github.com/HXYfighter/MolRL-MGPT。

**URL**: https://nips.cc/virtual/2023/poster/73033

---

## DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models
**Author**: Ying Fan · Olivia Watkins · Yuqing Du · Yuqing Du · Hao Liu · Moonkyung Ryu · Craig Boutilier · Pieter Abbeel · Mohammad Ghavamzadeh · Kangwook Lee · Kimin Lee

**Abstract**: Learning from human feedback has been shown to improve text-to-image models. These techniques first learn a reward function that captures what humans care about in the task and then improve the models based on the learned reward function. Even though relatively simple approaches (e.g., rejection sampling based on reward scores) have been investigated, fine-tuning text-to-image models with the reward function remains challenging. In this work, we propose using online reinforcement learning (RL) to fine-tune text-to-image models. We focus on diffusion models, defining the fine-tuning task as an RL problem, and updating the pre-trained text-to-image diffusion models using policy gradient to maximize the feedback-trained reward. Our approach, coined DPOK, integrates policy optimization with KL regularization. We conduct an analysis of KL regularization for both RL fine-tuning and supervised fine-tuning. In our experiments, we show that DPOK is generally superior to supervised fine-tuning with respect to both image-text alignment and image quality. Our code is available at https://github.com/google-research/google-research/tree/master/dpok.

**Abstract(Chinese)**: 从人类反馈中学习已被证明可以改善文本到图像模型。这些技术首先学习一个奖励函数，捕捉了人类在任务中关心的内容，然后根据所学的奖励函数改善模型。尽管已经调查了相对简单的方法（例如，基于奖励分数的拒绝抽样），但使用奖励函数对文本到图像模型进行微调仍然具有挑战性。在这项工作中，我们建议使用在线强化学习（RL）来微调文本到图像模型。我们专注于扩散模型，将微调任务定义为一个RL问题，并使用策略梯度来最大化经过反馈训练的奖励来更新预训练的文本到图像扩散模型。我们的方法，命名为DPOK，将策略优化与KL正则化相结合。我们对RL微调和监督微调的KL正则化进行了分析。在我们的实验中，我们展示了DPOK在图像文本对齐和图像质量方面通常优于监督微调。我们的代码可以在https://github.com/google-research/google-research/tree/master/dpok找到。

**URL**: https://nips.cc/virtual/2023/poster/72652

---

## Adjustable Robust Reinforcement Learning for Online 3D Bin Packing
**Author**: Yuxin Pan · Yize Chen · Fangzhen Lin

**Abstract**: Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints.  While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound  by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.

**Abstract(Chinese)**: 为在线3D装箱问题（3D-BPP）设计有效政策一直是一个长期的挑战，主要是由于输入箱序列的不可预测性和严格的物理约束。尽管当前用于在线3D-BPP的深度强化学习（DRL）方法在优化基础箱序列分布上的平均性能方面取得了令人鼓舞的结果，但它们在一些最坏情况出现的真实世界设置中通常会失败。标准的鲁棒DRL算法往往过分优先考虑在正常问题实例分布下性能的优化，以牺牲在最坏情况下的性能。为了解决这些问题，我们首先引入一个基于排列的攻击者，以调查为解决在线3D-BPP提出的DRL和启发式方法的实际鲁棒性。然后，我们提出了一种可调节的鲁棒强化学习（AR2L）框架，该框架允许有效地调整鲁棒性权重，以实现策略在平均和最坏环境中的性能平衡。具体来说，我们将目标函数制定为期望和最坏情况返回的加权和，并通过与混合动力学的返回相关来推导出性能的下限。为了实现这一下限，我们采用一个迭代过程，该过程搜索相关的混合动力学并改进相应的策略。我们将此过程集成到两种流行的鲁棒对抗算法中，以开发精确和近似的AR2L算法。实验证明，AR2L在改善策略鲁棒性的同时，仍然保持了在正常情况下的可接受性能水平，具有多功能性。

**URL**: https://nips.cc/virtual/2023/poster/72997

---

## Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric
**Author**: Yiming Wang · Ming Yang · Renzhi Dong · Binbin Sun · Furui Liu · Leong Hou U

**Abstract**: Reward shaping is an effective technique for integrating domain knowledge into reinforcement learning (RL). However, traditional approaches like potential-based reward shaping totally rely on manually designing shaping reward functions, which significantly restricts exploration efficiency and introduces human cognitive biases.While a number of RL methods have been proposed to boost exploration by designing an intrinsic reward signal as exploration bonus. Nevertheless, these methods heavily rely on the count-based episodic term in their exploration bonus which falls short in scalability. To address these limitations, we propose a general end-to-end potential-based exploration bonus for deep RL via potentials of state discrepancy, which motivates the agent to discover novel states and provides them with denser rewards without manual intervention. Specifically, we measure the novelty of adjacent states by calculating their distance using the bisimulation metric-based potential function, which enhances agent's exploration and ensures policy invariance. In addition, we offer a theoretical guarantee on our inverse dynamic bisimulation metric, bounding the value difference and ensuring that the agent explores states with higher TD error, thus significantly improving training efficiency. The proposed approach is named \textbf{LIBERTY} (exp\textbf{L}oration v\textbf{I}a \textbf{B}isimulation m\textbf{E}t\textbf{R}ic-based s\textbf{T}ate discrepanc\textbf{Y}) which is comprehensively evaluated on the MuJoCo and the Arcade Learning Environments. Extensive experiments have verified the superiority and scalability of our algorithm compared with other competitive methods.

**Abstract(Chinese)**: 奖励塑造是将领域知识融入强化学习(RL)的有效技术。然而，传统方法如基于潜势的奖励塑造完全依赖于手动设计塑造奖励函数，这显著限制了探索效率并引入人类认知偏见。虽然已经提出了许多RL方法来通过设计内在奖励信号作为探索奖励来增强探索，但这些方法在其探索奖励中严重依赖于计数的情节项，因此在可扩展性方面存在短板。为了解决这些局限性，我们提出了一个基于潜在的端到端的探索奖励，用于深度RL，通过状态差异的潜在函数来激励代理程序发现新颖状态，并为其提供更密集的奖励，无需人工干预。具体来说，我们通过计算其距离使用双模拟度量为基础的潜在函数来衡量邻近状态的新颖性，从而增强代理的探索并确保策略的不变性。此外，我们在我们的逆动态双模拟测度上提供了一个理论保证，限制了值的差异，并确保代理探索具有更高的TD误差的状态，从而显着提高了训练效率。提出的方法被命名为	extbf{自由}（通过双模拟度量的状态差异）在MuJoCo和Arcade Learning Environments上进行了全面评估。广泛的实验已经验证了我们的算法与其他竞争方法相比的优越性和可扩展性。

**URL**: https://nips.cc/virtual/2023/poster/73067

---

## Structured State Space Models for In-Context Reinforcement Learning
**Author**: Chris Lu · Yannick Schroecker · Albert Gu · Emilio Parisotto · Jakob Foerster · Satinder Singh · Feryal Behbahani

**Abstract**: Structured state space sequence (S4) models have recently achieved state-of-the-art performance on long-range sequence modeling tasks. These models also have fast inference speeds and parallelisable training, making them potentially useful in many reinforcement learning settings. We propose a  modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks. We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task. We evaluate our modified architecture on a set of partially-observable environments and find that, in practice, our model outperforms RNN's while also running over five times faster. Then, by leveraging the model’s ability to handle long-range sequences, we achieve strong performance on a challenging meta-learning task in which the agent is given a randomly-sampled continuous control environment, combined with a randomly-sampled linear projection of the environment's observations and actions. Furthermore, we show the resulting model can adapt to out-of-distribution held-out tasks. Overall, the results presented in this paper show that structured state space models are fast and performant for in-context reinforcement learning tasks. We provide code at https://github.com/luchris429/s5rl.

**Abstract(Chinese)**: 结构化状态空间序列（S4）模型最近在长程序列建模任务中取得了最先进的性能。这些模型还具有快速的推理速度和可并行化的训练，使它们在许多强化学习设置中可能会很有用。我们提出了S4变体的修改，使我们能够并行初始化和重置隐藏状态，从而能够处理强化学习任务。我们展示了我们修改后的架构在序列长度方面的渐近速度比变压器快，并且在简单的基于内存的任务上表现比循环神经网络更好。我们在一组部分可观测环境上评估了我们修改后的架构，并发现实际上我们的模型表现优于循环神经网络，同时速度也快五倍以上。然后，通过利用模型处理长程序列的能力，我们在具有挑战性的元学习任务上取得了良好的表现，其中代理程序给定了一个随机抽样的连续控制环境，结合了环境观察和动作的随机抽样线性投影。此外，我们展示了所得模型可以适应分布之外的保留任务。总的来说，本文提出的结果表明，结构化状态空间模型在上下文强化学习任务中快速而高效。我们在 https://github.com/luchris429/s5rl 提供了代码。

**URL**: https://nips.cc/virtual/2023/poster/72850

---

## Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning
**Author**: Akhil Bagaria · Ben Abbatematteo · Omer Gottesman · Matt Corsaro · Sreehari Rammohan · George Konidaris

**Abstract**: An agent learning an option in hierarchical reinforcement learning must solve three problems: identify the option's subgoal (termination condition), learn a policy, and learn where that policy will succeed (initiation set). The termination condition is typically identified first, but the option policy and initiation set must be learned simultaneously, which is challenging because the initiation set depends on the option policy, which changes as the agent learns. Consequently, data obtained from option execution becomes invalid over time, leading to an inaccurate initiation set that subsequently harms downstream task performance. We highlight three issues---data non-stationarity, temporal credit assignment, and pessimism---specific to learning initiation sets, and propose to address them using tools from off-policy value estimation and classification. We show that our method learns higher-quality initiation sets faster than existing methods (in MiniGrid and Montezuma's Revenge), can automatically discover promising grasps for robot manipulation (in Robosuite), and improves the performance of a state-of-the-art option discovery method in a challenging maze navigation task in MuJoCo.

**Abstract(Chinese)**: 在分层强化学习中学习选项的代理必须解决三个问题:确定选项的子目标（终止条件）、学习策略以及确定该策略成功的条件（初始集）。通常首先确定终止条件，但必须同时学习选项策略和初始集，这是具有挑战性的，因为初始集取决于选项策略，而代理学习过程中选项策略会发生变化。因此，从选项执行中获得的数据随时间变化，导致初始集不准确，随后影响下游任务性能。我们强调了三个问题——数据的非稳态性、时间上的信用分配和悲观主义——这些特定于学习初始集，并提出使用离策略值估计和分类工具来解决这些问题。我们展示了我们的方法比现有方法更快地学习到更高质量的初始集（在MiniGrid和Montezuma's Revenge中），可以自动发现机器人操作中有前景的抓取点（在Robosuite中），并且在 MuJoCo 中挑战迷宫导航任务中改进了最先进的选项发现方法的性能。

**URL**: https://nips.cc/virtual/2023/poster/72866

---

## Robust Knowledge Transfer in Tiered Reinforcement Learning
**Author**: Jiawei Huang · Niao He

**Abstract**: In this paper, we study the Tiered Reinforcement Learning setting, a parallel transfer learning framework, where the goal is to transfer knowledge from the low-tier (source) task to the high-tier (target) task to reduce the exploration risk of the latter while solving the two tasks in parallel. Unlike previous work, we do not assume the low-tier and high-tier tasks share the same dynamics or reward functions, and focus on robust knowledge transfer without prior knowledge on the task similarity. We identify a natural and necessary condition called the ``Optimal Value Dominance'' for our objective. Under this condition, we propose novel online learning algorithms such that, for the high-tier task, it can achieve constant regret on partial states depending on the task similarity and retain near-optimal regret when the two tasks are dissimilar, while for the low-tier task, it can keep near-optimal without making sacrifice. Moreover, we further study the setting with multiple low-tier tasks, and propose a novel transfer source selection mechanism, which can ensemble the information from all low-tier tasks and allow provable benefits on a much larger state-action space.

**Abstract(Chinese)**: 在本文中，我们研究了分层强化学习设置，这是一个并行传输学习框架，其目标是从低层（源）任务向高层（目标）任务转移知识，以减少后者的探索风险，同时并行解决这两个任务。与先前的工作不同，我们不假设低层和高层任务共享相同的动态或奖励函数，并侧重于在没有关于任务相似性的先验知识的情况下进行强大的知识转移。我们确定了一个称为“最优值支配”的自然和必要条件来实现我们的目标。在这种条件下，我们提出了新颖的在线学习算法，使得对于高层任务，根据任务相似性，它可以在部分状态上实现恒定的后悔，并在两个任务不相似时保持近乎最优的后悔，而对于低层任务，它可以保持近乎最优而不做出牺牲。此外，我们进一步研究了具有多个低层任务的环境，并提出了一种新颖的传输源选择机制，它可以集成来自所有低层任务的信息，并在更大的状态-动作空间中实现可证实的收益。

**URL**: https://nips.cc/virtual/2023/poster/73015

---

## No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions
**Author**: Tiancheng Jin · Junyan Liu · Chloé Rouyer · William Chang · Chen-Yu Wei · Haipeng Luo

**Abstract**: Existing online learning algorithms for adversarial Markov Decision Processes achieve $\mathcal{O}(\sqrt{T})$ regret after $T$ rounds of interactions even if the loss functions are chosen arbitrarily by an adversary, with the caveat that the transition function has to be fixed.This is because it has been shown that adversarial transition functions make no-regret learning impossible.Despite such impossibility results, in this work, we develop algorithms that can handle both adversarial losses and adversarial transitions, with regret increasing smoothly in the degree of maliciousness of the adversary.More concretely, we first propose an algorithm that enjoys $\widetilde{\mathcal{O}}(\sqrt{T} + C^{P})$ regret where $C^{P}$ measures how adversarial the transition functions are and can be at most $\mathcal{O}(T)$.While this algorithm itself requires knowledge of $C^{P}$, we further develop a black-box reduction approach that removes this requirement.Moreover, we also show that further refinements of the algorithm not only maintains the same regret bound, but also simultaneously adapts to easier environments (where losses are generated in a certain stochastically constrained manner as in [Jin et al. 2021]) and achieves $\widetilde{\mathcal{O}}(U + \sqrt{UC^{L}}  + C^{P})$ regret, where $U$ is some standard gap-dependent coefficient and $C^{L}$ is the amount of corruption on losses.

**Abstract(Chinese)**: 现有的用于对抗性马尔可夫决策过程的在线学习算法在进行了T轮交互之后，即使损失函数是由对手任意选择的，也能实现O(√T)的后悔，前提是转移函数是固定的。这是因为已经证明对抗性转移函数使得无后悔学习变得不可能。尽管存在这样的不可能结果，在这项工作中，我们开发了可以处理对抗性损失和对抗性转移的算法，并且后悔在对手恶意程度上升时会平滑增加。更具体地说，我们首先提出了一种算法，其后悔度为∼O(√T+C^P)，其中C^P衡量了转移函数的对抗性，并且最多可以为O(T)。虽然这个算法本身需要了解C^P，但我们进一步开发了一种黑盒化简方法来去除这一要求。此外，我们还表明算法的进一步改进不仅保持了相同的后悔界限，而且还同时适应了更容易的环境（在此环境中损失是按一定的随机约束方式生成的，就像[Jin et al. 2021]中一样），并且实现了∼O(U+√(UC^L)+C^P)的后悔，其中U是某个标准的间隙相关系数，C^L是损失上的破坏程度。

**URL**: https://nips.cc/virtual/2023/poster/73055

---

## General Munchausen Reinforcement Learning with Tsallis Kullback-Leibler Divergence
**Author**: Lingwei Zhu · Zheng Chen · Matthew Schlegel · Martha White

**Abstract**: Many policy optimization approaches in reinforcement learning incorporate a Kullback-Leilbler (KL) divergence to the previous policy, to prevent the policy from changing too quickly. This idea was initially proposed in a seminal paper on Conservative Policy Iteration, with approximations given by algorithms like TRPO and Munchausen Value Iteration (MVI). We continue this line of work by investigating a generalized KL divergence---called the Tsallis KL divergence. Tsallis KL defined by the $q$-logarithm is a strict generalization, as $q = 1$ corresponds to the standard KL divergence; $q > 1$ provides a range of new options. We characterize the types of policies learned under the Tsallis KL, and motivate when $q >1$ could be beneficial.  To obtain a practical algorithm that incorporates Tsallis KL regularization, we extend MVI, which is one of the simplest approaches to incorporate KL regularization. We show that this generalized MVI($q$) obtains significant improvements over the standard MVI($q = 1$) across 35 Atari games.

**Abstract(Chinese)**: 摘要：在强化学习中，许多政策优化方法都包含了对先前政策的Kullback-Leilbler (KL)散度，以防止政策变化过快。这个想法最初是在一篇关于保守政策迭代的重要论文中提出的，算法如TRPO和Munchausen Value Iteration (MVI)给出了近似。我们继续这一研究方向，通过研究一个称为Tsallis KL散度的广义KL散度。通过$q$-对数定义的Tsallis KL是一个严格的泛化，因为$q=1$对应于标准的KL散度；$q>1$提供了一系列新选项。我们表征了在Tsallis KL下学习的政策类型，并阐明了在$q>1$时可能有益的动机。为了获得一个包含Tsallis KL正则化的实用算法，我们扩展了MVI，这是一种包含KL正则化的最简单的方法之一。我们证明，这种广义MVI($q$)相对于标准MVI($q=1$)在35个Atari游戏中获得了显著的改进。

**URL**: https://nips.cc/virtual/2023/poster/72977

---

## Multi-Modal Inverse Constrained Reinforcement Learning from a Mixture of Demonstrations
**Author**: Guanren Qiao · Guiliang Liu · Pascal Poupart · Zhiqiang Xu

**Abstract**: Inverse Constraint Reinforcement Learning (ICRL) aims to recover the underlying constraints respected by expert agents in a data-driven manner. Existing ICRL algorithms typically assume that the demonstration data is generated by a single type of expert. However, in practice, demonstrations often comprise a mixture of trajectories collected from various expert agents respecting different constraints, making it challenging to explain expert behaviors with a unified constraint function. To tackle this issue, we propose a Multi-Modal Inverse Constrained Reinforcement Learning (MMICRL) algorithm for simultaneously estimating multiple constraints corresponding to different types of experts. MMICRL constructs a flow-based density estimator that enables unsupervised expert identification from demonstrations, so as to infer the agent-specific constraints. Following these constraints, MMICRL imitates expert policies with a novel multi-modal constrained policy optimization objective that minimizes the agent-conditioned policy entropy and maximizes the unconditioned one. To enhance robustness, we incorporate this objective into the contrastive learning framework. This approach enables imitation policies to capture the diversity of behaviors among expert agents. Extensive experiments in both discrete and continuous environments show that MMICRL outperforms other baselines in terms of constraint recovery and control performance.

**Abstract(Chinese)**: 逆向约束强化学习（ICRL）旨在以数据驱动的方式恢复专家代理人遵守的潜在约束。现有的ICRL算法通常假设示范数据由单一类型的专家生成。然而，在实践中，演示通常包括从尊重不同约束的各种专家代理人收集的轨迹的混合，这使得用统一的约束函数解释专家行为具有挑战性。为了解决这个问题，我们提出了一种用于同时估计不同类型专家对应的多模态逆向约束强化学习（MMICRL）算法。MMICRL构建了一种基于流的密度估计器，能够从示范中无监督地识别专家，以推断特定代理人的约束。根据这些约束，MMICRL使用一种新型多模态约束策略优化目标来模仿专家策略，该目标最小化代理人条件策略熵并最大化无条件策略熵。为增强稳健性，我们将这一目标纳入对比学习框架。这一方法使模仿策略能够捕获专家代理人之间的行为多样性。在离散和连续环境中的大量实验表明，MMICRL在约束恢复和控制性能方面优于其他基线。

**URL**: https://nips.cc/virtual/2023/poster/72837

---

## Mutual Information Regularized Offline Reinforcement Learning
**Author**: Xiao Ma · Bingyi Kang · Zhongwen Xu · Min Lin · Shuicheng Yan

**Abstract**: The major challenge of offline RL is the distribution shift that appears when out-of-distribution actions are queried, which makes the policy improvement direction biased by extrapolation errors. Most existing methods address this problem by penalizing the policy or value for deviating from the behavior policy during policy improvement or evaluation. In this work, we propose a novel MISA framework to approach offline RL from the perspective of Mutual Information between States and Actions in the dataset by directly constraining the policy improvement direction. MISA constructs lower bounds of mutual information parameterized by the policy and Q-values. We show that optimizing this lower bound is equivalent to maximizing the likelihood of a one-step improved policy on the offline dataset. Hence, we constrain the policy improvement direction to lie in the data manifold. The resulting algorithm simultaneously augments the policy evaluation and improvement by adding mutual information regularizations. MISA is a general framework that unifies conservative Q-learning (CQL) and behavior regularization methods (e.g., TD3+BC) as special cases. We introduce 3 different variants of MISA, and empirically demonstrate that tighter mutual information lower bound gives better offline RL performance. In addition, our extensive experiments show MISA significantly outperforms a wide range of baselines on various tasks of the D4RL benchmark, e.g., achieving 742.9 total points on gym-locomotion tasks. Our code is attached and will be released upon publication.

**Abstract(Chinese)**: 离线RL的主要挑战是当询问超出分布的动作时出现的分布偏移，这使得策略改进方向受到外推误差的偏差。大部分现有方法通过在策略改进或评估过程中对策略或价值进行惩罚来解决这个问题。在这项工作中，我们提出了一种新颖的MISA框架，从数据集中状态和动作之间的互信息的角度直接限制策略改进方向，以解决离线RL问题。MISA构建了由策略和Q值参数化的互信息的下界。我们展示了优化这个下界等同于最大化离线数据集上一步改进策略的可能性。因此，我们限制策略改进方向位于数据流形中。由此产生的算法通过添加互信息正则化同时增强了策略评估和改进。MISA是一个通用框架，将保守的Q-learning（CQL）和行为正则化方法（例如TD3+BC）统一为特殊情况。我们引入了MISA的3个不同变体，并从经验上证明了更紧的互信息下界可以提供更好的离线RL性能。此外，我们广泛的实验表明，MISA显著优于各种D4RL基准任务的基线，例如在gym-locomotion任务上达到了742.9的总分。我们的代码已附上，将在发表后发布。

**URL**: https://nips.cc/virtual/2023/poster/72924

---

## Efficient Diffusion Policies For Offline Reinforcement Learning
**Author**: Bingyi Kang · Xiao Ma · Chao Du · Tianyu Pang · Shuicheng Yan

**Abstract**: Offline reinforcement learning (RL) aims to learn optimal policies from offline datasets, where the parameterization of policies is crucial but often overlooked. Recently, Diffsuion-QL significantly boosts the performance of offline RL by representing a policy with a diffusion model, whose success relies on a parametrized Markov Chain with hundreds of steps for sampling. However, Diffusion-QL suffers from two critical limitations. 1) It is computationally inefficient to forward and backward through the whole Markov chain during training. 2) It is incompatible with maximum likelihood-based RL algorithms (e.g., policy gradient methods) as the likelihood of diffusion models is intractable. Therefore, we propose efficient diffusion policy (EDP) to overcome these two challenges. EDP approximately constructs actions from corrupted ones at training to avoid running the sampling chain. We conduct extensive experiments on the D4RL benchmark. The results show that EDP can reduce the diffusion policy training time from 5 days to 5 hours on gym-locomotion tasks. Moreover, we show that EDP is compatible with various offline RL algorithms (TD3, CRR, and IQL) and achieves new state-of-the-art on D4RL by large margins over previous methods.

**Abstract(Chinese)**: 离线强化学习（RL）旨在从离线数据集中学习最优策略，其中策略的参数化至关重要，但经常被忽视。最近，Diffusion-QL通过使用扩散模型表示策略，显著提升了离线RL的性能，其成功依赖于一个带有数百步用于采样的参数化马尔可夫链。然而，Diffusion-QL存在两个关键限制。1）在训练期间，通过整个马尔可夫链进行前向和后向传递的计算效率低下。2）它与基于最大似然的RL算法（例如，策略梯度方法）不兼容，因为扩散模型的似然是难以处理的。因此，我们提出了高效扩散策略（EDP）来克服这两个挑战。EDP在训练期间通过从损坏的动作近似构建动作，以避免运行采样链。我们在D4RL基准测试上进行了大量实验。结果表明，EDP可以将gym-locomotion任务的扩散策略训练时间从5天缩短至5小时。此外，我们展示了EDP与各种离线RL算法（TD3、CRR和IQL）兼容，并在D4RL上大幅领先于以前的方法，实现了新的最优性能。

**URL**: https://nips.cc/virtual/2023/poster/73058

---

## Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets
**Author**: Anirudhan Badrinath · Yannis Flet-Berliac · Allen Nie · Emma Brunskill

**Abstract**: Despite the recent advancements in offline reinforcement learning via supervised learning (RvS) and the success of the decision transformer (DT) architecture in various domains, DTs have fallen short in several challenging benchmarks. The root cause of this underperformance lies in their inability to seamlessly connect segments of suboptimal trajectories. To overcome this limitation, we present a novel approach to enhance RvS methods by integrating intermediate targets. We introduce the Waypoint Transformer (WT), using an architecture that builds upon the DT framework and  conditioned on automatically-generated waypoints. The results show a significant increase in the final return compared to existing RvS methods, with performance on par or greater than existing state-of-the-art temporal difference learning-based methods. Additionally, the performance and stability improvements are largest in the most challenging environments and data configurations, including AntMaze Large Play/Diverse and Kitchen Mixed/Partial.

**Abstract(Chinese)**: 尽管最近通过监督学习 (RvS) 在离线强化学习方面取得了一些进展，并且决策 Transformer (DT) 架构在各个领域取得了成功，但是 DT 在一些具有挑战性的基准测试中表现不佳。此次性能不佳的根本原因在于它们无法无缝连接次优轨迹的各个片段。为克服这一限制，我们提出了一种新颖的方法，通过整合中间目标来增强 RvS 方法。我们引入了 Waypoint Transformer (WT) ，使用一个基于 DT 框架构建的结构，并且受到自动生成的航路点的条件控制。结果显示，与现有的 RvS 方法相比，最终回报大幅增加，并且在与现有最先进的基于时序差异学习的方法相比性能相当甚至更好。此外，在最具挑战性的环境和数据配置中，包括 AntMaze Large Play/Diverse 以及 Kitchen Mixed/Partial，性能和稳定性的改进最为显著。

**URL**: https://nips.cc/virtual/2023/poster/72475

---

## Adversarial Model for Offline Reinforcement Learning
**Author**: Mohak Bhardwaj · Tengyang Xie · Byron Boots · Nan Jiang · Ching-An Cheng

**Abstract**: We propose a novel model-based offline Reinforcement Learning (RL) framework, called Adversarial Model for Offline Reinforcement Learning (ARMOR), which can robustly learn policies to improve upon an arbitrary reference policy regardless of data coverage. ARMOR is designed to optimize policies for the worst-case performance relative to the reference policy through adversarially training a Markov decision process model. In theory, we prove that ARMOR, with a well-tuned hyperparameter, can compete with the best policy within data coverage when the reference policy is supported by the data. At the same time, ARMOR is robust to hyperparameter choices: the policy learned by ARMOR, with any admissible hyperparameter, would never degrade the performance of the reference policy, even when the reference policy is not covered by the dataset. To validate these properties in practice, we design a scalable implementation of ARMOR, which by adversarial training, can optimize policies without using model ensembles in contrast to typical model-based methods. We show that ARMOR achieves competent performance with both state-of-the-art offline model-free and model-based RL algorithms and can robustly improve the reference policy over various hyperparameter choices.

**Abstract(Chinese)**: 我们提出了一种新颖的基于模型的离线强化学习（RL）框架，名为Adversarial Model for Offline Reinforcement Learning（ARMOR），它可以稳健地学习策略，以改进任意参考策略，而不考虑数据覆盖范围。 ARMOR 旨在通过对马尔可夫决策过程模型进行对抗训练，针对参考策略的最坏情况表现来优化策略。理论上，我们证明了在参考策略受数据支持时，经过良好调整的超参数，ARMOR 可以与数据覆盖范围内的最佳策略竞争。同时，ARMOR 对超参数选择具有鲁棒性：ARMOR 学习的策略，无论使用任何可接受的超参数，都不会降低参考策略的性能，即使参考策略不在数据集的覆盖范围内。为了验证这些特性，在实践中，我们设计了 ARMOR 的可扩展实现，通过对抗训练，可以优化策略，而无需使用模型集成，这与典型的基于模型的方法相比。我们展示了 ARMOR 在状态良好的离线无模型和基于模型的 RL 算法中表现出了良好的性能，并能够稳健地改进参考策略，对各种超参数选择进行了验证。

**URL**: https://nips.cc/virtual/2023/poster/72755

---

## Conservative State Value Estimation for Offline Reinforcement Learning
**Author**: Liting Chen · Jie Yan · Zhengdao Shao · Lu Wang · Qingwei Lin · Saravanakumar Rajmohan · Thomas Moscibroda · Dongmei Zhang

**Abstract**: Offline reinforcement learning faces a significant challenge of value over-estimation due to the distributional drift between the dataset and the current learned policy, leading to learning failure in practice. The common approach is to incorporate a penalty term to reward or value estimation in the Bellman iterations. Meanwhile, to avoid extrapolation on out-of-distribution (OOD) states and actions, existing methods focus on conservative Q-function estimation. In this paper, we propose Conservative State Value Estimation (CSVE), a new approach that learns conservative V-function via directly imposing penalty on OOD states. Compared to prior work, CSVE allows more effective state value estimation with conservative guarantees and further better policy optimization. Further, we apply CSVE and develop a practical actor-critic algorithm in which the critic does the conservative value estimation by additionally sampling and penalizing the states around the dataset, and the actor applies advantage weighted updates extended with state exploration to improve the policy. We evaluate in classic continual control tasks of D4RL, showing that our method performs better than the conservative Q-function learning methods and is strongly competitive among recent SOTA methods.

**Abstract(Chinese)**: 离线强化学习面临着一个重大挑战，即由于数据集和当前学习策略之间的分布漂移，导致值高估，进而在实践中导致学习失败。常见的方法是在贝尔曼迭代中加入惩罚项以奖励或价值估计。同时，为了避免在分布之外的状态和行为上进行外推，现有方法侧重于保守的 Q 函数估计。在本文中，我们提出了保守状态值估计（CSVE），这是一种新方法，通过直接对分布之外的状态施加惩罚来学习保守的 V 函数。与先前的工作相比，CSVE 能够更有效地估计状态值，并提供保守的保证，进一步改进策略优化。此外，我们应用 CSVE 并开发了一个实用的演员-评论家算法，其中评论家通过额外采样和惩罚数据集周围的状态进行保守的值估计，而演员利用带有状态探索的优势加权更新来改进策略。我们在 D4RL 的经典连续控制任务中进行评估，结果显示我们的方法表现优于保守的 Q 函数学习方法，并且在最近的 SOTA 方法中具有很强的竞争力。

**URL**: https://nips.cc/virtual/2023/poster/72661

---

## Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning
**Author**: Jianzhun Shao · Yun Qu · Chen Chen · Hongchang Zhang · Xiangyang Ji

**Abstract**: Offline multi-agent reinforcement learning is challenging due to the coupling effect of both distribution shift issue common in offline setting and the high dimension issue common in multi-agent setting, making the action out-of-distribution (OOD) and value overestimation phenomenon excessively severe. To mitigate this problem, we propose a novel multi-agent offline RL algorithm, named CounterFactual Conservative Q-Learning (CFCQL) to conduct conservative value estimation. Rather than regarding all the agents as a high dimensional single one and directly applying single agent conservative methods to it, CFCQL calculates conservative regularization for each agent separately in a counterfactual way and then linearly combines them to realize an overall conservative value estimation. We prove that it still enjoys the underestimation property and the performance guarantee as those single agent conservative methods do, but the induced regularization and safe policy improvement bound are independent of the agent number, which is therefore theoretically superior to the direct treatment referred to above, especially when the agent number is large. We further conduct experiments on four environments including both discrete and continuous action settings on both existing and our man-made datasets, demonstrating that CFCQL outperforms existing methods on most datasets and even with a remarkable margin on some of them.

**Abstract(Chinese)**: 离线多智能体强化学习具有挑战性，因为它同时受到离线环境中常见的分布偏移问题和多智能体环境中常见的高维问题的耦合影响，使得动作脱离分布（OOD）和价值高估现象过于严重。为了缓解这一问题，我们提出了一种新颖的多智能体离线强化学习算法，名为反事实保守Q学习（CFCQL），用于进行保守价值估计。CFCQL不同于将所有智能体视为高维度的单个智能体，并直接应用单智能体的保守方法，而是以反事实的方式分别为每个智能体计算保守正则化，然后线性组合它们以实现整体保守价值估计。我们证明它仍然具有低估性质和性能保证，类似于单一智能体的保守方法，但引入的正则化和安全策略改进上限与智能体数量无关，因此在理论上优于上述直接处理方式，特别是当智能体数量较大时。我们在包括离散和连续动作设置的四个环境中进行了实验，涵盖了现有数据集和我们自己的人造数据集，结果表明CFCQL在大多数数据集上优于现有方法，甚至在其中一些数据集上表现明显优越。

**URL**: https://nips.cc/virtual/2023/poster/72777

---

## Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning
**Author**: Jialong Wu · Haoyu Ma · Chaoyi Deng · Mingsheng Long

**Abstract**: Unsupervised pre-training methods utilizing large and diverse datasets have achieved tremendous success across a range of domains. Recent work has investigated such unsupervised pre-training methods for model-based reinforcement learning (MBRL) but is limited to domain-specific or simulated data. In this paper, we study the problem of pre-training world models with abundant in-the-wild videos for efficient learning of downstream visual control tasks. However, in-the-wild videos are complicated with various contextual factors, such as intricate backgrounds and textured appearance, which precludes a world model from extracting shared world knowledge to generalize better. To tackle this issue, we introduce Contextualized World Models (ContextWM) that explicitly separate context and dynamics modeling to overcome the complexity and diversity of in-the-wild videos and facilitate knowledge transfer between distinct scenes. Specifically, a contextualized extension of the latent dynamics model is elaborately realized by incorporating a context encoder to retain contextual information and empower the image decoder, which encourages the latent dynamics model to concentrate on essential temporal variations. Our experiments show that in-the-wild video pre-training equipped with ContextWM can significantly improve the sample efficiency of MBRL in various domains, including robotic manipulation, locomotion, and autonomous driving. Code is available at this repository: https://github.com/thuml/ContextWM.

**Abstract(Chinese)**: 使用大规模和多样化数据集的无监督预训练方法在各个领域取得了巨大成功。最近的研究探讨了针对基于模型的强化学习（MBRL）的无监督预训练方法，但仅限于特定领域或模拟数据。在本文中，我们研究了利用丰富的野外视频预训练世界模型，以有效学习下游视觉控制任务的问题。然而，野外视频受到各种背景和纹理外观等各种情境因素的影响，这使得世界模型无法提取共享的世界知识以实现更好的泛化。为了解决这个问题，我们引入了具有上下文的世界模型（ContextWM），明确分离了上下文和动态建模，以克服野外视频的复杂性和多样性，并促进不同场景之间的知识传递。具体而言，通过将上下文编码器纳入，精心实现了潜在动态模型的上下文化扩展，以保留上下文信息并增强图像解码器，从而鼓励潜在动态模型专注于基本的时间变化。我们的实验表明，配备ContextWM的野外视频预训练可以显著提高MBRL在各个领域（包括机器人操作、运动和自动驾驶）的样本效率。代码可以在此存储库找到：https://github.com/thuml/ContextWM。

**URL**: https://nips.cc/virtual/2023/poster/72660

---

## Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration
**Author**: Dongyoung Kim · Jinwoo Shin · Pieter Abbeel · Younggyo Seo

**Abstract**: A promising technique for exploration is to maximize the entropy of visited state distribution, i.e., state entropy, by encouraging uniform coverage of visited state space. While it has been effective for an unsupervised setup, it tends to struggle in a supervised setup with a task reward, where an agent prefers to visit high-value states to exploit the task reward. Such a preference can cause an imbalance between the distributions of high-value states and low-value states, which biases exploration towards low-value state regions as a result of the state entropy increasing when the distribution becomes more uniform. This issue is exacerbated when high-value states are narrowly distributed within the state space, making it difficult for the agent to complete the tasks. In this paper, we present a novel exploration technique that maximizes the value-conditional state entropy, which separately estimates the state entropies that are conditioned on the value estimates of each state, then maximizes their average. By only considering the visited states with similar value estimates for computing the intrinsic bonus, our method prevents the distribution of low-value states from affecting exploration around high-value states, and vice versa. We demonstrate that the proposed alternative to the state entropy baseline significantly accelerates various reinforcement learning algorithms across a variety of tasks within MiniGrid, DeepMind Control Suite, and Meta-World benchmarks. Source code is available at https://sites.google.com/view/rl-vcse.

**Abstract(Chinese)**: 探索的一种有前途的技术是通过最大化访问状态分布的熵，即状态熵，鼓励访问状态空间的均匀覆盖。虽然这在无监督设置下很有效，在有任务奖励的监督设置中往往会遇到困难，因为代理倾向于访问高价值状态以利用任务奖励。这种偏好可能导致高价值状态和低价值状态的分布不均衡，这会使探索偏向低价值状态区域，因为当分布变得更加均匀时，状态熵会增加。当高价值状态在状态空间内分布狭窄时，这个问题会加剧，这会使得代理很难完成任务。在本文中，我们提出了一种新的探索技术，它最大化了值条件状态熵，分别估计了每个状态的值估计条件下的状态熵，然后最大化它们的平均值。通过仅考虑具有类似值估计的访问状态来计算内在奖励，我们的方法可以防止低价值状态的分布影响高价值状态周围的探索，反之亦然。我们证明了我们所提出的替代状态熵基准明显加速了各种强化学习算法在MiniGrid、DeepMind Control Suite和Meta-World基准上的任务。源代码可在https://sites.google.com/view/rl-vcse找到。

**URL**: https://nips.cc/virtual/2023/poster/72613

---

## Two Heads are Better Than One: A Simple Exploration Framework for Efficient Multi-Agent Reinforcement Learning
**Author**: Jiahui Li · Kun Kuang · Baoxiang Wang · Xingchen Li · Fei Wu · Jun Xiao · Long Chen

**Abstract**: Exploration strategy plays an important role in reinforcement learning, especially in sparse-reward tasks. In cooperative multi-agent reinforcement learning~(MARL), designing a suitable exploration strategy is much more challenging due to the large state space and the complex interaction among agents. Currently, mainstream exploration methods in MARL either contribute to exploring the unfamiliar states which are large and sparse, or measuring the interaction among agents with high computational costs. We found an interesting phenomenon that different kinds of exploration plays a different role in different MARL scenarios, and choosing a suitable one is often more effective than designing an exquisite algorithm. In this paper, we propose a exploration method that incorporate the \underline{C}uri\underline{O}sity-based and \underline{IN}fluence-based exploration~(COIN) which is simple but effective in various situations. First, COIN measures the influence of each agent on the other agents based on mutual information theory and designs it as intrinsic rewards which are applied to each individual value function. Moreover, COIN computes the curiosity-based intrinsic rewards via prediction errors which are added to the extrinsic reward. For integrating the two kinds of intrinsic rewards, COIN utilizes a novel framework in which they complement each other and lead to a sufficient and effective exploration on cooperative MARL tasks. We perform extensive experiments on different challenging benchmarks, and results across different scenarios show the superiority of our method.

**Abstract(Chinese)**: 探索策略在强化学习中扮演着重要角色，特别是在稀疏奖励任务中。在合作多智能体强化学习（MARL）中，由于庞大的状态空间和智能体之间复杂的相互作用，设计合适的探索策略变得更加具有挑战性。目前，MARL 中主流的探索方法要么有助于探索庞大且稀疏的陌生状态，要么要付出高计算代价来衡量智能体之间的相互作用。我们发现一个有趣的现象，即不同种类的探索在不同的 MARL 场景中扮演不同的角色，并且选择合适的探索策略往往比设计精妙的算法更加有效。在本文中，我们提出了一种探索方法，将基于好奇心和基于影响力的探索（COIN）结合起来，简单而有效地适用于各种情况。首先，COIN 根据互信息理论衡量每个智能体对其他智能体的影响，并将其设计为内在奖励，应用于每个个体价值函数。此外，COIN 通过预测误差计算基于好奇心的内在奖励，并将其添加到外在奖励中。为了整合这两种内在奖励，COIN 利用了一种新颖的框架，使它们相互补充，并在合作 MARL 任务中实现了充分有效的探索。我们在不同具有挑战性的基准测试中进行了大量实验，结果表明我们的方法在不同场景下具有优越性。

**URL**: https://nips.cc/virtual/2023/poster/72536

---

## Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning
**Author**: Guanlin Liu · Lifeng LAI

**Abstract**: Due to the broad range of applications of multi-agent reinforcement learning (MARL), understanding the effects of adversarial attacks against MARL model is essential for the safe applications of this model. Motivated by this, we investigate the impact of adversarial attacks on MARL. In the considered setup, there is an exogenous attacker who is able to modify the rewards before the agents receive them or manipulate the actions before the environment receives them. The attacker aims to guide each agent into a target policy or maximize the cumulative rewards under some specific reward function chosen by the attacker, while minimizing the amount of the manipulation on feedback and action. We first show the limitations of the action poisoning only attacks and the reward poisoning only attacks. We then introduce a mixed attack strategy with both the action poisoning and reward poisoning. We show that the mixed attack strategy can efficiently attack MARL agents even if the attacker has no prior information about the underlying environment and the agents’ algorithms.

**Abstract(Chinese)**: 由于多智能体强化学习（MARL）具有广泛的应用范围，理解对抗攻击对MARL模型的影响对于该模型的安全应用至关重要。出于这个原因，我们调查了对MARL的对抗攻击的影响。在考虑的设置中，存在一个外部攻击者，他能够在智能体接收奖励之前修改奖励或在环境接收动作之前操纵动作。攻击者的目标是引导每个智能体进入特定策略或在攻击者选择的特定奖励函数下最大化累积奖励，同时最大程度地减少对反馈和动作的操纵。我们首先展示了仅对动作操纵和仅对奖励操纵的攻击的局限性。然后我们引入了一个混合攻击策略，包括动作操纵和奖励操纵。我们表明，即使攻击者没有关于底层环境和智能体算法的先验信息，混合攻击策略也可以有效地攻击MARL智能体。

**URL**: https://nips.cc/virtual/2023/poster/72574

---

## Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning
**Author**: Yifan Zang · Jinmin He · Kai Li · Haobo Fu · Qiang Fu · Junliang Xing · Jian Cheng

**Abstract**: Grouping is ubiquitous in natural systems and is essential for promoting efficiency in team coordination. This paper proposes a novel formulation of Group-oriented Multi-Agent Reinforcement Learning (GoMARL), which learns automatic grouping without domain knowledge for efficient cooperation. In contrast to existing approaches that attempt to directly learn the complex relationship between the joint action-values and individual utilities, we empower subgroups as a bridge to model the connection between small sets of agents and encourage cooperation among them, thereby improving the learning efficiency of the whole team. In particular, we factorize the joint action-values as a combination of group-wise values, which guide agents to improve their policies in a fine-grained fashion. We present an automatic grouping mechanism to generate dynamic groups and group action-values. We further introduce a hierarchical control for policy learning that drives the agents in the same group to specialize in similar policies and possess diverse strategies for various groups. Experiments on the StarCraft II micromanagement tasks and Google Research Football scenarios verify our method's effectiveness. Extensive component studies show how grouping works and enhances performance.

**Abstract(Chinese)**: 分组在自然系统中无处不在，对于促进团队协调的效率至关重要。本文提出了一种新颖的Group-oriented Multi-Agent Reinforcement Learning (GoMARL)的组态方案，该方案能够在没有领域知识的情况下学习自动分组，以实现高效的合作。与现有方法直接学习联合动作价值和个体效用之间复杂关系的做法不同，我们将子群体作为桥梁，来建模一小组代理之间的关联，并鼓励它们之间的合作，从而提高整个团队的学习效率。具体而言，我们将联合动作价值因子化为组内价值的组合，指导代理以精细的方式改进其策略。我们提出了一种自动生成动态小组和小组动作价值的自动分组机制。此外，我们引入了针对策略学习的分层控制，推动同一组中的代理专注于类似策略，并为不同的小组提供多样化的策略。对StarCraft II微观管理任务和Google Research Football场景的实验验证了我们方法的有效性。广泛的组件研究展示了分组的工作方式并增强了性能。

**URL**: https://nips.cc/virtual/2023/poster/72436

---

## Instructing Goal-Conditioned Reinforcement Learning Agents with Temporal Logic Objectives
**Author**: Wenjie Qiu · Wensen Mao · He Zhu

**Abstract**: Goal-conditioned reinforcement learning (RL) is a powerful approach for learning general-purpose skills by reaching diverse goals. However, it has limitations when it comes to task-conditioned policies, where goals are specified by temporally extended instructions written in the Linear Temporal Logic (LTL) formal language. Existing approaches for finding LTL-satisfying policies rely on sampling a large set of LTL instructions during training to adapt to unseen tasks at inference time. However, these approaches do not guarantee generalization to out-of-distribution LTL objectives, which may have increased complexity. In this paper, we propose a novel approach to address this challenge. We show that simple goal-conditioned RL agents can be instructed to follow arbitrary LTL specifications without additional training over the LTL task space. Unlike existing approaches that focus on LTL specifications expressible as regular expressions, our technique is unrestricted and generalizes to $\omega$-regular expressions. Experiment results demonstrate the effectiveness of our approach in adapting goal-conditioned RL agents to satisfy complex temporal logic task specifications zero-shot.

**Abstract(Chinese)**: 抽象：目标条件强化学习（RL）是一种强大的方法，通过实现多样化的目标来学习通用技能。然而，在任务条件策略中，存在一些限制，其中目标由在线时序逻辑（LTL）形式语言编写的时间扩展指令指定。现有的寻找满足LTL的策略的方法依赖于在训练期间对大量LTL指令进行采样，以适应推理时的未见任务。然而，这些方法不能保证泛化到分布外的LTL目标，这可能增加了复杂性。在本文中，我们提出了一种解决这一挑战的新方法。我们表明，简单的目标条件RL代理可以被指示遵循任意LTL规范，而无需对LTL任务空间进行额外训练。与现有关注作为正规表达式可表达的LTL规范的方法不同，我们的技术是无限制的，并且可以推广到ω-正规表达式。实验结果表明了我们的方法在使目标条件RL代理适应复杂的时序逻辑任务规范的零-shot方面的有效性。

**URL**: https://nips.cc/virtual/2023/poster/73035

---

## Improved Bayesian Regret Bounds for Thompson Sampling in Reinforcement Learning
**Author**: Ahmadreza Moradipari · Mohammad Pedramfar · Modjtaba Shokrian Zini · Vaneet Aggarwal

**Abstract**: In this paper, we prove state-of-the-art Bayesian regret bounds for Thompson Sampling in reinforcement learning in a multitude of settings. We present a refined analysis of the information ratio, and show an upper bound of order $\widetilde{O}(H\sqrt{d_{l_1}T})$ in the time inhomogeneous reinforcement learning problem where $H$ is the episode length and $d_{l_1}$ is the Kolmogorov $l_1-$dimension of the space of environments. We then find concrete bounds of $d_{l_1}$ in a variety of settings, such as tabular, linear and finite mixtures, and discuss how our results improve the state-of-the-art.

**Abstract(Chinese)**: 摘要：在本文中，我们证明了在多种设置中强化学习中 Thompson Sampling 的贝叶斯遗憾界的最新成果。我们对信息比率进行了精细的分析，并在时间不均匀的强化学习问题中表明了一个 $\widetilde{O}(H\sqrt{d_{l_1}T})$ 的上界，其中 $H$ 是每一轮的长度，$d_{l_1}$ 是环境空间 Kolmogorov $l_1-$维度的阶数。然后，我们在各种设置中找到了 $d_{l_1}$ 的具体界限，比如表格、线性和有限混合，讨论了我们结果如何改进了最新的研究成果。

**URL**: https://nips.cc/virtual/2023/poster/72963

---

## Replicability in Reinforcement Learning
**Author**: Amin Karbasi · Grigoris Velegkas · Lin Yang · Felix Zhou

**Abstract**: We initiate the mathematical study of replicability as an   algorithmic property in the context of reinforcement learning (RL).  We focus on the fundamental setting of discounted tabular MDPs with access to a generative model.  Inspired by Impagliazzo et al. [2022], we say that an RL algorithm is replicable if,  with high probability,  it outputs the exact same policy  after two executions on i.i.d. samples drawn from the generator  when its internal randomness  is the same.  We first provide   an efficient $\rho$-replicable algorithm for $(\varepsilon, \delta)$-optimal policy estimation  with sample and time complexity $\widetilde O\left(\frac{N^3\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$,  where $N$ is the number of state-action pairs.  Next,  for the subclass of deterministic algorithms,  we provide a lower bound of order $\Omega\left(\frac{N^3}{(1-\gamma)^3\cdot\varepsilon^2\cdot\rho^2}\right)$.  Then, we study a relaxed version of replicability proposed  by Kalavasis et al. [2023] called TV indistinguishability.  We design a computationally efficient TV indistinguishable algorithm for policy estimation  whose sample complexity is $\widetilde O\left(\frac{N^2\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.  At the cost of $\exp(N)$ running time,  we transform these TV indistinguishable algorithms to $\rho$-replicable ones without increasing their sample complexity.  Finally,  we introduce the notion of approximate-replicability  where we only require that two outputted policies are close  under an appropriate statistical divergence (e.g., Renyi)  and show an improved sample complexity of $\widetilde O\left(\frac{N\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.

**Abstract(Chinese)**: 我们在强化学习（RL）的算法性质中，首次将可复制性的数学研究纳入讨论。我们关注具有生成模型的折扣表格MDP的基本环境。受Impagliazzo等人 [2022] 的启发，我们称RL算法为可复制的，如果在内部随机性相同时，从生成器中绘制的i.i.d.样本上进行两次执行后，以很高的概率输出完全相同的策略。我们首先提供了一个有效的$\rho$-可复制算法，用于$(\varepsilon, \delta)$-最优策略估计，其样本和时间复杂度为$\widetilde O\left(\frac{N^3\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$，其中$N$是状态-动作对的数量。接下来，对于确定性算法的子类，我们提供了一个阶为$\Omega\left(\frac{N^3}{(1-\gamma)^3\cdot\varepsilon^2\cdot\rho^2}\right)$的下界。然后，我们研究了Kalavasis等人 [2023] 提出的可复制性的放松版本，称为TV不可辨识。我们设计了一个计算效率高的TV不可辨识算法，其策略估计的样本复杂度为$\widetilde O\left(\frac{N^2\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$。以$\exp(N)$的运行时间为代价，我们将这些TV不可辨识算法转换为$\rho$-可复制算法，而不增加它们的样本复杂度。最后，我们引入了近似可复制性的概念，其中我们只要求两个输出的策略在适当的统计散度（例如，Renyi）下是接近的，并展示了$\widetilde O\left(\frac{N\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$的改进样本复杂度。

**URL**: https://nips.cc/virtual/2023/poster/72792

---

## Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning
**Author**: Hanlin Zhu · Paria Rashidinejad · Jiantao Jiao

**Abstract**: We propose A-Crab (Actor-Critic Regularized by Average Bellman error), a new practical algorithm for offline reinforcement learning (RL) in complex environments with insufficient data coverage. Our algorithm combines the marginalized importance sampling framework with the actor-critic paradigm, where the critic returns evaluations of the actor (policy) that are pessimistic relative to the offline data and have a small average (importance-weighted) Bellman error. Compared to existing methods, our algorithm simultaneously offers a number of advantages:(1) It achieves the optimal statistical rate of $1/\sqrt{N}$---where $N$ is the size of offline dataset---in converging to the best policy covered in the offline dataset, even when combined with general function approximators.(2) It relies on a weaker \textit{average} notion of policy coverage (compared to the $\ell_\infty$ single-policy concentrability) that exploits the structure of policy visitations.(3) It outperforms the data-collection behavior policy over a wide range of specific hyperparameters. We provide both theoretical analysis and experimental results to validate the effectiveness of our proposed algorithm. The code is available at https://github.com/zhuhl98/ACrab.

**Abstract(Chinese)**: 我们提出A-Crab（Actor-Critic由平均贝尔曼误差正则化），这是一种新的实用算法，用于在数据覆盖不足的复杂环境中进行离线强化学习（RL）。我们的算法将边际化重要性抽样框架与演员-评论家范式相结合，其中评论家返回对演员（策略）的评估，这些评估相对于线下数据而言是悲观的，并且具有较小的平均（重要性加权）贝尔曼误差。与现有方法相比，我们的算法同时提供了多个优势：（1）即使与一般的函数逼近器结合使用，它也能实现收敛到线下数据集中涵盖的最佳策略的最佳统计速率为$1/\sqrt{N}$，其中$N$是线下数据集的大小。（2）它依赖于较弱的\textit{平均}策略覆盖概念（而不是$\ell_\infty$单策略聚集性），利用策略访问的结构。（3）在各种特定超参数范围内，它优于数据收集行为策略。我们提供理论分析和实验结果，以验证我们提出的算法的有效性。代码可在https://github.com/zhuhl98/ACrab 上获得。

**URL**: https://nips.cc/virtual/2023/poster/72845

---

## Model-Free Reinforcement Learning with the Decision-Estimation Coefficient
**Author**: Dylan J Foster · Noah Golowich · Jian Qian · Alexander Rakhlin · Ayush Sekhari

**Abstract**: We consider the problem of interactive decision making, encompassing structured bandits and reinforcementlearning with general function approximation. Recently, Foster et al. (2021) introduced theDecision-Estimation Coefficient, a measure of statistical complexity that lower bounds the optimal regret for interactive decisionmaking, as well as a meta-algorithm, Estimation-to-Decisions, which achieves upperbounds in terms of the same quantity. Estimation-to-Decisions is a reduction, which liftsalgorithms for (supervised) online estimation into algorithms fordecision making. In this paper, we show that by combining Estimation-to-Decisions witha specialized form of "optimistic" estimation introduced byZhang (2022), it is possible to obtain guaranteesthat improve upon those of Foster et al. (2021) byaccommodating more lenient notions of estimation error. We use this approach to derive regret bounds formodel-free reinforcement learning with value function approximation, and give structural results showing when it can and cannot help more generally.

**Abstract(Chinese)**: 我们考虑交互式决策问题，包括结构化的赌博机和具有一般函数逼近的强化学习。最近，Foster等人(2021)引入了决策估计系数，这是一个衡量统计复杂性的指标，它下界了交互式决策的最优遗憾，以及一个元算法——估计到决策，它在相同数量方面实现了上界。估计到决策是一种缩减，它将用于(监督)在线估计的算法提升为用于决策的算法。在这篇论文中，我们展示了通过将估计到决策与张(2022)引入的一种特殊形式的“乐观”估计相结合，可以获得保证，这些保证可以改进Foster等人(2021)的保证，因为可以容纳更宽松的估计误差概念。我们使用这种方法得出了无模型强化学习的遗憾边界，并提供了结构性结果，显示它在何时能够以及何时不能够更普遍地帮助。

**URL**: https://nips.cc/virtual/2023/poster/72518

---

## MAG-GNN: Reinforcement Learning Boosted Graph Neural Network
**Author**: Lecheng Kong · Jiarui Feng · Hao Liu · Dacheng Tao · Yixin Chen · Muhan Zhang

**Abstract**: While Graph Neural Networks (GNNs) recently became powerful tools in graph learning tasks, considerable efforts have been spent on improving GNNs' structural encoding ability. A particular line of work proposed subgraph GNNs that use subgraph information to improve GNNs' expressivity and achieved great success. However, such effectivity sacrifices the efficiency of GNNs by enumerating all possible subgraphs. In this paper, we analyze the necessity of complete subgraph enumeration and show that a model can achieve a comparable level of expressivity by considering a small subset of the subgraphs. We then formulate the identification of the optimal subset as a combinatorial optimization problem and propose Magnetic Graph Neural Network (MAG-GNN), a reinforcement learning (RL) boosted GNN, to solve the problem. Starting with a candidate subgraph set, MAG-GNN employs an RL agent to iteratively update the subgraphs to locate the most expressive set for prediction. This reduces the exponential complexity of subgraph enumeration to the constant complexity of a subgraph search algorithm while keeping good expressivity. We conduct extensive experiments on many datasets, showing that MAG-GNN achieves competitive performance to state-of-the-art methods and even outperforms many subgraph GNNs. We also demonstrate that MAG-GNN effectively reduces the running time of subgraph GNNs.

**Abstract(Chinese)**: 尽管图神经网络（GNNs）最近成为图学习任务中的强大工具，但人们已经付出了相当大的努力来改进GNNs的结构编码能力。一个特定的研究方向提出了子图GNNs，利用子图信息来提高GNNs的表达能力并取得了巨大成功。然而，这种有效性是通过列举所有可能的子图来牺牲GNNs的效率。在本文中，我们分析了完整子图列举的必要性，并表明一个模型可以通过考虑子图的一个小子集来达到可比较的表达能力。然后，我们将最佳子集的识别问题以组合优化问题的形式提出，并提出了磁性图神经网络（MAG-GNN），这是一个通过强化学习（RL）增强的GNN，用于解决这个问题。从候选子图集开始，MAG-GNN利用RL代理来迭代更新子图，以找到用于预测的最具表现力的子集。这将子图列举的指数复杂性降低到了子图搜索算法的常数复杂性，同时保持了良好的表达能力。我们在许多数据集上进行了大量实验证明，MAG-GNN达到了与最先进方法竞争的性能，甚至胜过了许多子图GNNs。我们还证明了MAG-GNN有效地减少了子图GNNs的运行时间。

**URL**: https://nips.cc/virtual/2023/poster/72031

---

## Ensemble-based Deep Reinforcement Learning for Vehicle Routing Problems under Distribution Shift
**Author**: YUAN JIANG · Zhiguang Cao · Yaoxin Wu · Wen Song · Jie Zhang

**Abstract**: While performing favourably on the independent and identically distributed (i.i.d.) instances, most of the existing neural methods for vehicle routing problems (VRPs) struggle to generalize in the presence of a distribution shift. To tackle this issue, we propose an ensemble-based deep reinforcement learning method for VRPs, which learns a group of diverse sub-policies to cope with various instance distributions. In particular, to prevent convergence of the parameters to the same one, we enforce diversity across sub-policies by leveraging Bootstrap with random initialization. Moreover, we also explicitly pursue inequality between sub-policies by exploiting regularization terms during training to further enhance diversity. Experimental results show that our method is able to outperform the state-of-the-art neural baselines on randomly generated instances of various distributions, and also generalizes favourably on the benchmark instances from TSPLib and CVRPLib, which confirmed the effectiveness of the whole method and the respective designs.

**Abstract(Chinese)**: 在独立同分布（i.i.d.）实例上表现良好的情况下，大多数现有的车辆路径问题（VRP）的神经方法在存在分布转移的情况下很难泛化。为了解决这个问题，我们提出了一种基于集成的深度强化学习方法，用于VRP，该方法学习一组不同的子策略以应对各种实例分布。特别是，为了防止参数收敛到相同的参数值，我们利用随机初始化的Bootstrap来实现子策略之间的多样性。此外，我们还在训练过程中明确追求子策略之间的不平等，通过利用正则化项进一步增强多样性。实验结果表明，我们的方法能够在各种分布的随机生成实例上胜过最先进的神经基线，并且在TSPLib和CVRPLib的基准实例上表现良好，从而证实了整个方法及各自设计的有效性。

**URL**: https://nips.cc/virtual/2023/poster/72153

---

## Goal-Conditioned Predictive Coding for Offline Reinforcement Learning
**Author**: Zilai Zeng · Ce Zhang · Shijie Wang · Chen Sun

**Abstract**: Recent work has demonstrated the effectiveness of formulating decision making as supervised learning on offline-collected trajectories. Powerful sequence models, such as GPT or BERT, are often employed to encode the trajectories. However, the benefits of performing sequence modeling on trajectory data remain unclear. In this work, we investigate whether sequence modeling has the ability to condense trajectories into useful representations that enhance policy learning. We adopt a two-stage framework that first leverages sequence models to encode trajectory-level representations, and then learns a goal-conditioned policy employing the encoded representations as its input. This formulation allows us to consider many existing supervised offline RL methods as specific instances of our framework. Within this framework, we introduce Goal-Conditioned Predictive Coding (GCPC), a sequence modeling objective that yields powerful trajectory representations and leads to performant policies. Through extensive empirical evaluations on AntMaze, FrankaKitchen and Locomotion environments, we observe that sequence modeling can have a significant impact on challenging decision making tasks. Furthermore, we demonstrate that GCPC learns a goal-conditioned latent representation encoding the future trajectory, which enables competitive performance on all three benchmarks.

**Abstract(Chinese)**: 最近的工作表明，将决策制定为对离线收集的轨迹进行监督学习的有效性。强大的序列模型，如GPT或BERT，经常被用来对轨迹进行编码。然而，对于在轨迹数据上执行序列建模的益处仍不清楚。在这项工作中，我们调查序列建模是否具有将轨迹压缩为有用表示以增强策略学习的能力。我们采用了一个两阶段框架，首先利用序列模型对轨迹级表示进行编码，然后学习一个以编码表示作为输入的目标条件策略。这种制定方式使我们能够将许多现有的监督离线RL方法视为我们框架的特定实例。在这个框架内，我们引入了目标条件预测编码（GCPC），这是一个产生强大轨迹表示并导致性能良好的策略的序列建模目标。通过在AntMaze、FrankaKitchen和Locomotion环境上进行了广泛的实证评估，我们观察到序列建模对具有挑战性的决策任务可能产生重大影响。此外，我们证明GCPC学习了一个以目标为条件的潜在表示，对未来轨迹进行编码，从而在所有三个基准测试中实现了竞争性能。

**URL**: https://nips.cc/virtual/2023/poster/72131

---

## Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback
**Author**: Minyoung Hwang · Gunmin Lee · Hogun Kee · Chan Woo Kim · Kyungjae Lee · Songhwai Oh

**Abstract**: Reinforcement learning from human feedback (RLHF) alleviates the problem of designing a task-specific reward function in reinforcement learning by learning it from human preference. However, existing RLHF models are considered inefficient as they produce only a single preference data from each human feedback. To tackle this problem, we propose a novel RLHF framework called SeqRank, that uses sequential preference ranking to enhance the feedback efficiency. Our method samples trajectories in a sequential manner by iteratively selecting a defender from the set of previously chosen trajectories $\mathcal{K}$ and a challenger from the set of unchosen trajectories $\mathcal{U}\setminus\mathcal{K}$, where $\mathcal{U}$ is the replay buffer. We propose two trajectory comparison methods with different defender sampling strategies: (1) sequential pairwise comparison that selects the most recent trajectory and (2) root pairwise comparison that selects the most preferred trajectory from $\mathcal{K}$. We construct a data structure and rank trajectories by preference to augment additional queries. The proposed method results in at least 39.2% higher average feedback efficiency than the baseline and also achieves a balance between feedback efficiency and data dependency. We examine the convergence of the empirical risk and the generalization bound of the reward model with Rademacher complexity. While both trajectory comparison methods outperform conventional pairwise comparison, root pairwise comparison improves the average reward in locomotion tasks and the average success rate in manipulation tasks by 29.0% and 25.0%, respectively. The source code and the videos are provided in the supplementary material.

**Abstract(Chinese)**: 从人类反馈中进行强化学习（RLHF）通过从人类喜好中学习来减轻强化学习中设计任务特定奖励函数的问题。然而，现有的RLHF模型被认为效率低，因为它们仅从每个人类反馈中产生单一的偏好数据。为了解决这个问题，我们提出了一种称为SeqRank的新型RLHF框架，该框架使用连续偏好排序来增强反馈效率。我们的方法通过从先前选择的轨迹集$\mathcal{K}$中迭代地选择一个守卫者和从未选择的轨迹集$\mathcal{U}\setminus\mathcal{K}$中选择一个挑战者的方式按顺序采样轨迹，其中$\mathcal{U}$是重放缓冲区。我们提出了两种不同的守卫者采样策略的轨迹比较方法：（1）选择最近轨迹的顺序配对比较和（2）选择$\mathcal{K}$中最优先轨迹的顺序配对比较。我们构建了一个数据结构来按偏好对轨迹进行排序以增加额外的查询。所提出的方法使反馈效率至少比基线提高了39.2%，同时实现了反馈效率和数据依赖性之间的平衡。我们检验了奖励模型的经验风险收敛性和Rademacher复杂度的泛化界限。虽然两种轨迹比较方法都优于传统的配对比较，但是$\mathcal{K}$中的顺序配对比较能将轨迹的平均奖励提高了29.0%，并且在操作任务的平均成功率上提高了25.0%。附录中提供了源代码和视频。

**URL**: https://nips.cc/virtual/2023/poster/71915

---

## Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning
**Author**: Yihang Yao · ZUXIN LIU · Zhepeng Cen · Jiacheng Zhu · Wenhao Yu · Tingnan Zhang · DING ZHAO

**Abstract**: Safe reinforcement learning (RL) focuses on training reward-maximizing agents subject to pre-defined safety constraints. Yet, learning versatile safe policies that can adapt to varying safety constraint requirements during deployment without retraining remains a largely unexplored and challenging area. In this work, we formulate the versatile safe RL problem and consider two primary requirements: training efficiency and zero-shot adaptation capability. To address them, we introduce the Conditioned Constrained Policy Optimization (CCPO) framework, consisting of two key modules: (1) Versatile Value Estimation (VVE) for approximating value functions under unseen threshold conditions, and (2) Conditioned Variational Inference (CVI) for encoding arbitrary constraint thresholds during policy optimization. Our extensive experiments demonstrate that CCPO outperforms the baselines in terms of safety and task performance while preserving zero-shot adaptation capabilities to different constraint thresholds data-efficiently. This makes our approach suitable for real-world dynamic applications.

**Abstract(Chinese)**: 安全强化学习（RL）侧重于训练在预定义安全约束条件下最大化奖励的代理，然而，学习能够适应部署过程中不断变化的安全约束要求，而无需重新训练的多才多艺的安全策略仍然是一个未经深入探讨且具有挑战性的领域。在这项工作中，我们制定了多才多艺的安全RL问题，并考虑了两个主要要求：训练效率和零-shot适应能力。为了解决这些问题，我们引入了条件约束策略优化（CCPO）框架，包括两个关键模块：（1）多才多艺价值估计（VVE），用于在未见阈值条件下逼近价值函数，以及（2）条件变分推断（CVI），用于在策略优化过程中对任意约束阈值进行编码。我们的广泛实验表明，CCPO在安全性和任务性能方面优于基准线，并且在数据有效性上保留了对不同约束阈值的零-shot适应能力。这使得我们的方法适用于现实世界的动态应用。

**URL**: https://nips.cc/virtual/2023/poster/72253

---

## Prediction and Control in Continual Reinforcement Learning
**Author**: Nishanth Anand · Doina Precup

**Abstract**: Temporal difference (TD) learning is often used to update the estimate of the value function which is used by RL agents to extract useful policies. In this paper, we focus on value function estimation in continual reinforcement learning. We propose to decompose the value function into two components which update at different timescales: a permanent value function, which holds general knowledge that persists over time, and a transient value function, which allows quick adaptation to new situations. We establish theoretical results showing that our approach is well suited for continual learning and draw connections to the complementary learning systems (CLS) theory from neuroscience. Empirically, this approach improves performance significantly on both prediction and control problems.

**Abstract(Chinese)**: 时间差异（TD）学习常用于更新值函数的估计，该估计由强化学习代理使用以提取有用的策略。在本文中，我们专注于持续强化学习中的值函数估计。我们建议将值函数分解为两个组成部分，这两个组成部分在不同时间尺度上进行更新：一个是永久值函数，它保存随时间持续存在的一般知识；另一个是瞬时值函数，它允许快速适应新情况。我们建立了理论结果，表明我们的方法非常适合持续学习，并与神经科学中的互补学习系统（CLS）理论建立了联系。从经验上看，这种方法显著改善了预测和控制问题的性能。

**URL**: https://nips.cc/virtual/2023/poster/72001

---

## Sample Efficient Reinforcement Learning in Mixed Systems through Augmented Samples and Its Applications to Queueing Networks
**Author**: Honghao Wei · Honghao Wei · Xin Liu · Weina Wang · Lei Ying

**Abstract**: This paper considers a class of reinforcement learning problems, which involve systems with two types of states: stochastic and pseudo-stochastic. In such systems, stochastic states follow a stochastic transition kernel while the transitions of pseudo-stochastic states are deterministic {\em given} the stochastic states/transitions. We refer to such systems as mixed systems, which are widely used in various applications, including Manufacturing systems, communication networks, and queueing networks. We propose a sample-efficient RL method that accelerates learning by generating augmented data samples. The proposed algorithm is data-driven (model-free), but it learns the policy from data samples from both real and augmented samples. This method significantly improves learning by reducing the sample complexity such that the dataset only needs to have sufficient coverage of the stochastic states. We analyze the sample complexity of the proposed method under Fitted Q Iteration (FQI) and demonstrate that the optimality gap decreases as  $O\left(\sqrt{\frac{1}{n}}+\sqrt{\frac{1}{m}}\right),$ where $n$ represents the number of real samples, and $m$ is the number of augmented samples per real sample. It is important to note that without augmented samples, the optimality gap is $O(1)$ due to the insufficient data coverage of the pseudo-stochastic states. Our experimental results on multiple queueing network applications confirm that the proposed method indeed significantly accelerates both deep Q-learning and deep policy gradient.

**Abstract(Chinese)**: 这篇论文考虑了一类强化学习问题，涉及具有两种状态的系统：随机状态和伪随机状态。在这样的系统中，随机状态遵循随机转移核，而伪随机状态的转移是确定的，即给定随机状态/转移。我们将这样的系统称为混合系统，在各种应用中广泛使用，包括制造系统、通信网络和排队网络。我们提出了一种样本高效的强化学习方法，通过生成增强数据样本来加速学习。所提出的算法是数据驱动的（无模型），但它从实际和增强样本数据中学习策略。这种方法通过减少样本复杂性显著改善了学习，使得数据集只需足够覆盖随机状态。在 Fitted Q Iteration（FQI）下，我们分析了所提出方法的样本复杂性，并证明最优性差距随之减小为 $O\left(\sqrt{\frac{1}{n}}+\sqrt{\frac{1}{m}}\right)$，其中 $n$ 表示真实样本的数量，$m$ 表示每个真实样本的增强样本的数量。值得注意的是，如果没有增强样本，由于伪随机状态的数据覆盖不足，最优性差距是 $O(1)$。我们在多个排队网络应用上的实验结果证实，所提出的方法确实显著加速了深度 Q 学习和深度策略梯度。

**URL**: https://nips.cc/virtual/2023/poster/71961

---

## CORL: Research-oriented Deep Offline Reinforcement Learning Library
**Author**: Denis Tarasov · Alexander Nikulin · Dmitry Akimov · Vladislav Kurenkov · Sergey Kolesnikov

**Abstract**: CORL is an open-source library that provides thoroughly benchmarked single-file implementations of both deep offline and offline-to-online reinforcement learning algorithms. It emphasizes a simple developing experience with a straightforward codebase and a modern analysis tracking tool. In CORL, we isolate methods implementation into separate single files, making performance-relevant details easier to recognize. Additionally, an experiment tracking feature is available to help log metrics, hyperparameters, dependencies, and more to the cloud. Finally, we have ensured the reliability of the implementations by benchmarking commonly employed D4RL datasets providing a transparent source of results that can be reused for robust evaluation tools such as performance profiles, probability of improvement, or expected online performance.

**Abstract(Chinese)**: CORL是一个开源库，提供深度离线和离线到在线强化学习算法的经过彻底基准测试的单文件实现。它强调简单的开发体验，具有直观的代码库和现代化的分析跟踪工具。在CORL中，我们将方法实现隔离为单独的文件，使性能相关的细节更容易识别。此外，还提供了实验跟踪功能，可帮助记录指标、超参数、依赖项等到云端。最后，我们通过基准测试了常用的D4RL数据集，确保了实现的可靠性，提供了可重复使用的透明结果源，可用于稳健评估工具，如性能概况、改进概率或预期在线性能。

**URL**: https://nips.cc/virtual/2023/poster/73613

---

## Latent exploration for Reinforcement Learning
**Author**: Alberto Silvio Chiappa · Alessandro Marin Vargas · Ann Huang · Alexander Mathis

**Abstract**: In Reinforcement Learning, agents learn policies by exploring and interacting with the environment. Due to the curse of dimensionality, learning policies that map high-dimensional sensory input to motor output is particularly challenging. During training, state of the art methods (SAC, PPO, etc.) explore the environment by perturbing the actuation with independent Gaussian noise. While this unstructured exploration has proven successful in numerous tasks, it can be suboptimal for overactuated systems. When multiple actuators, such as motors or muscles, drive behavior, uncorrelated perturbations risk diminishing each other's effect, or modifying the behavior in a task-irrelevant way. While solutions to introduce time correlation across action perturbations exist, introducing correlation across actuators has been largely ignored. Here, we propose LATent TIme-Correlated Exploration (Lattice), a method to inject temporally-correlated noise into the latent state of the policy network, which can be seamlessly integrated with on- and off-policy algorithms. We demonstrate that the noisy actions generated by perturbing the network's activations can be modeled as a multivariate Gaussian distribution with a full covariance matrix. In the PyBullet locomotion tasks, Lattice-SAC achieves state of the art results, and reaches 18\% higher reward than unstructured exploration in the Humanoid environment. In the musculoskeletal control environments of MyoSuite, Lattice-PPO achieves higher reward in most reaching and object manipulation tasks, while also finding more energy-efficient policies with reductions of 20-60\%. Overall, we demonstrate the effectiveness of structured action noise in time and actuator space for complex motor control tasks. The code is available at: https://github.com/amathislab/lattice.

**Abstract(Chinese)**: 在强化学习中，智能体通过探索和与环境进行交互来学习策略。由于维度灾难的存在，学习将高维感知输入映射到运动输出的策略尤其具有挑战性。在训练过程中，最先进的方法（如SAC、PPO等）通过对执行器施加独立的高斯噪声来探索环境。虽然这种非结构化的探索在许多任务中被证明是成功的，但对于多执行系统来说可能是次优的。当多个执行器（如电机或肌肉）驱动行为时，不相关的扰动可能会降低彼此的作用，或者以与任务无关的方式修改行为。尽管存在引入时间相关性的动作扰动的解决方案，但跨执行器引入相关性却被大多数人忽视。在这里，我们提出了LATent TIme-Correlated Exploration（Lattice），这是一种方法，可以将临时相关噪声注入策略网络的潜在状态中，它可以与在线和离线算法无缝集成。我们证明，通过扰动网络的激活所生成的嘈杂动作可以被建模为具有完整协方差矩阵的多元高斯分布。在PyBullet运动任务中，Lattice-SAC取得了最先进的结果，在Humanoid环境中的奖励比非结构化探索高出18％。在MyoSuite的肌肉骨骼控制环境中，Lattice-PPO在大多数到达和物体操作任务中获得更高的奖励，同时还找到了更节能的策略，能效提高了20-60％。总的来说，我们展示了结构化动作噪声在复杂的动作控制任务中的有效性。代码位于:https://github.com/amathislab/lattice。

**URL**: https://nips.cc/virtual/2023/poster/72059

---

## Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning
**Author**: Changyu CHEN · Ramesha Karunasena · Thanh Nguyen · Arunesh Sinha · Pradeep Varakantham

**Abstract**: Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists. In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy — the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.

**Abstract(Chinese)**: 在强化学习（RL）中，许多问题寻求具有大离散多维但无序行动空间的最优策略；其中包括随机分配资源的问题，比如多个安全资源和应急响应单位的部署等。在这种情况下的一个挑战是，基础行动空间是分类的（离散且无序）且很大，现有的强化学习方法效果不佳。此外，这些问题需要实际行动（分配）的有效性；这种有效性约束通常难以用紧凑的数学形式来表达。问题的分配性质还倾向于随机最优策略，如果存在的话。在这项工作中，我们通过（1）应用（状态）条件正规化流来紧凑表示随机策略——紧凑性是由于网络只生成一个抽样行动和相应的行动对数概率，然后由演员-评论家方法使用；以及（2）使用无效行动拒绝方法（通过有效行动oracle）来更新基本策略来解决这些挑战。通过我们导出的修改后的策略梯度实现了行动拒绝。最后，我们进行了大量实验，以展示我们的方法相对于先前方法的可伸缩性以及在任何状态下强制执行任意状态条件约束的分布支持的能力。

**URL**: https://nips.cc/virtual/2023/poster/72301

---

## Conditional Mutual Information for Disentangled Representations in Reinforcement Learning
**Author**: Mhairi Dunion · Trevor McInroe · Kevin Luck · Kevin Sebastian Luck · Josiah Hanna · Stefano Albrecht

**Abstract**: Reinforcement Learning (RL) environments can produce training data with spurious correlations between features due to the amount of training data or its limited feature coverage. This can lead to RL agents encoding these misleading correlations in their latent representation, preventing the agent from generalising if the correlation changes within the environment or when deployed in the real world. Disentangled representations can improve robustness, but existing disentanglement techniques that minimise mutual information between features require independent features, thus they cannot disentangle correlated features. We propose an auxiliary task for RL algorithms that learns a disentangled representation of high-dimensional observations with correlated features by minimising the conditional mutual information between features in the representation. We demonstrate experimentally, using continuous control tasks, that our approach improves generalisation under correlation shifts, as well as improving the training performance of RL algorithms in the presence of correlated features.

**Abstract(Chinese)**: 强化学习（RL）环境可能由于训练数据的数量或其有限的特征覆盖而产生特征之间的虚假相关性。这可能导致RL代理程序在其潜在表示中对这些误导性相关性进行编码，当环境中的相关性发生变化或在真实世界中部署时，会阻止代理程序进行泛化。分离的表示可以提高鲁棒性，但现有的最小化特征之间互信息的分离技术需要独立的特征，因此无法分离相关特征。我们为RL算法提出了一种辅助任务，通过最小化表示中特征之间的条件互信息来学习具有相关特征的高维观测的分离表示。我们通过连续控制任务的实验证明，我们的方法在相关性转移下改善了泛化能力，并改善了RL算法在存在相关特征时的训练性能。

**URL**: https://nips.cc/virtual/2023/poster/72294

---

## RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability
**Author**: Chuning Zhu · Max Simchowitz · Siri Gadipudi · Abhishek Gupta

**Abstract**: Visual model-based RL methods typically encode image observations into low-dimensional representations in a manner that does not eliminate redundant information. This leaves them susceptible to spurious variations -- changes in task-irrelevant components such as background distractors or lighting conditions. In this paper, we propose a visual model-based RL method that learns a latent representation resilient to such spurious variations. Our training objective encourages the representation to be maximally predictive of dynamics and reward, while constraining the information flow from the observation to the latent representation. We demonstrate that this objective significantly bolsters the resilience of visual model-based RL methods to visual distractors, allowing them to operate in dynamic environments. We then show that while the learned encoder is able to operate in dynamic environments, it is not invariant under significant distribution shift. To address this, we propose a simple reward-free alignment procedure that enables test time adaptation of the encoder. This allows for quick adaptation to widely differing environments without having to relearn the dynamics and policy. Our effort is a step towards making model-based RL a practical and useful tool for dynamic, diverse domains and we show its effectiveness in simulation tasks with significant spurious variations.

**Abstract(Chinese)**: 视觉模型基础的强化学习方法通常将图像观测编码为低维表示，以一种不消除冗余信息的方式。这使它们容易受到虚假变化的影响，比如与任务无关的组成部分的变化，如背景干扰或光照条件的变化。在本文中，我们提出了一种视觉模型基础的强化学习方法，该方法学习了一种对这种虚假变化具有韧性的潜在表示。我们的训练目标鼓励该表示对动态和奖励具有最大的预测性，同时限制从观测到潜在表示的信息流。我们证明了这一目标显著增强了视觉模型基础的强化学习方法对视觉干扰的韧性，使其能够在动态环境中运作。然后我们展示了，虽然所学的编码器能够在动态环境中运作，但它并不具有显著的分布转移不变性。为了解决这个问题，我们提出了一个简单的无奖励对齐过程，使得在测试时能够对该编码器进行适应。这使得能够快速适应大不相同的环境，而无需重新学习动态和策略。我们的努力是向着使基于模型的强化学习成为动态、多样领域中实际和有用工具的一步，并且我们展示了它在具有明显虚假变化的仿真任务中的有效性。

**URL**: https://nips.cc/virtual/2023/poster/71822

---

## Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition
**Author**: Suyoung Lee · Myungsik Cho · Youngchul Sung

**Abstract**: Meta-reinforcement learning (meta-RL) techniques have demonstrated remarkable success in generalizing deep reinforcement learning across a range of tasks. Nevertheless, these methods often struggle to generalize beyond tasks with parametric variations. To overcome this challenge, we propose Subtask Decomposition and Virtual Training (SDVT), a novel meta-RL approach that decomposes each non-parametric task into a collection of elementary subtasks and parameterizes the task based on its decomposition. We employ a  Gaussian mixture VAE to meta-learn the decomposition process, enabling the agent to reuse policies acquired from common subtasks. Additionally, we propose a virtual training procedure, specifically designed for non-parametric task variability, which generates hypothetical subtask compositions, thereby enhancing generalization to previously unseen subtask compositions. Our method significantly improves performance on the Meta-World ML-10 and ML-45 benchmarks, surpassing current state-of-the-art techniques.

**Abstract(Chinese)**: 元元强化学习（Meta-RL）技术已经在多个任务上展现了通用深度强化学习的显著成功。然而，这些方法通常难以推广到具有参数变化的任务之外。为了克服这一挑战，我们提出了子任务分解和虚拟训练（SDVT），这是一种新颖的元强化学习方法，它将每个非参数化任务分解成一系列基本子任务，并基于其分解对任务进行参数化。我们采用高斯混合VAE来元学习分解过程，从而使代理能够重复使用从常见子任务中获得的策略。此外，我们提出了一种针对非参数化任务变化的虚拟训练程序，该程序生成假设的子任务组成，从而增强对之前未见的子任务组成的推广能力。我们的方法显著改善了在 Meta-World ML-10 和 ML-45 基准上的性能，并超过了当前最先进的技术。

**URL**: https://nips.cc/virtual/2023/poster/72053

---

## A Long $N$-step Surrogate Stage Reward for Deep Reinforcement Learning
**Author**: Junmin Zhong · Ruofan Wu · Jennie Si

**Abstract**: We introduce a new stage reward estimator  named the long $N$-step surrogate stage (LNSS) reward for deep reinforcement learning (RL). It aims at mitigating the high variance problem, which has shown impeding successful convergence of learning, hurting task performance, and hindering applications of deep RL in continuous control problems. In this paper we show that LNSS, which utilizes a long reward trajectory of  rewards of future steps, provides consistent performance improvement measured by average reward, convergence speed, learning success rate,and variance reduction in $Q$ values and rewards.  Our evaluations are based on a variety of environments in DeepMind Control Suite and OpenAI Gym  by using  LNSS in baseline deep RL algorithms such as DDPG, D4PG, and TD3. We show  that LNSS reward has enabled good results that have been challenging to obtain by deep RL previously. Our analysis also shows that  LNSS exponentially reduces the upper bound on the variances of $Q$ values from respective single-step methods.

**Abstract(Chinese)**: 摘要：我们引入了一种名为长$N$步替代阶段（LNSS）奖励的新阶段奖励估计器，用于深度强化学习（RL）。它旨在减轻高方差问题，该问题已经显示出阻碍学习成功收敛，损害任务性能，并阻碍深度RL在连续控制问题中的应用。在本文中，我们展示了利用未来步骤的奖励长路径的LNSS，通过平均奖励、收敛速度、学习成功率和$Q$值和奖励的方差减少，提供了一致的性能改进。我们的评估基于DeepMind控制套件和OpenAI Gym中的各种环境，通过在基线深度RL算法（如DDPG、D4PG和TD3）中使用LNSS。我们展示了LNSS奖励使得之前深度RL难以获得的良好结果成为可能。我们的分析还表明，LNSS指数级地减少了$Q$值的方差上界，从而比相应的单步方法降低了许多。

**URL**: https://nips.cc/virtual/2023/poster/72325

---

## Video Prediction Models as Rewards for Reinforcement Learning
**Author**: Alejandro Escontrela · Ademi Adeniji · Wilson Yan · Ajay Jain · Xue Bin Peng · Ken Goldberg · Youngwoon Lee · Danijar Hafner · Pieter Abbeel

**Abstract**: Specifying reward signals that allow agents to learn complex behaviors is a long-standing challenge in reinforcement learning.A promising approach is to extract preferences for behaviors from unlabeled videos, which are widely available on the internet. We present Video Prediction Rewards (VIPER), an algorithm that leverages pretrained video prediction models as action-free reward signals for reinforcement learning. Specifically, we first train an autoregressive transformer on expert videos and then use the video prediction likelihoods as reward signals for a reinforcement learning agent. VIPER enables expert-level control without programmatic task rewards across a wide range of DMC, Atari, and RLBench tasks. Moreover, generalization of the video prediction model allows us to derive rewards for an out-of-distribution environment where no expert data is available, enabling cross-embodiment generalization for tabletop manipulation. We see our work as starting point for scalable reward specification from unlabeled videos that will benefit from the rapid advances in generative modeling. Source code and datasets are available on the project website: https://ViperRL.com

**Abstract(Chinese)**: 通过从未标记的视频中提取行为偏好，这是强化学习中长期存在的挑战。一种有前途的方法是利用互联网上广泛可获得的未标记视频，为行为提取偏好。我们提出了Video Prediction Rewards（VIPER）算法，它利用预先训练的视频预测模型作为无需动作的奖励信号来进行强化学习。具体而言，我们首先对专家视频训练自回归变压器，然后使用视频预测概率作为强化学习代理的奖励信号。VIPER可以在一系列DMC、Atari和RLBench任务中实现专家级控制，而无需编程任务奖励。此外，视频预测模型的泛化能力使我们能够为无专家数据可用的分布外环境导出奖励，从而实现桌面操作的跨体现泛化。我们认为我们的工作是从未标记视频中可扩展奖励规范的起点，将受益于生成建模的快速进展。项目网站上提供了源代码和数据集：https://ViperRL.com

**URL**: https://nips.cc/virtual/2023/poster/72159

---

## $\texttt{TACO}$: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning
**Author**: Ruijie Zheng · Xiyao Wang · Yanchao Sun · Shuang Ma · Jieyu Zhao · Huazhe Xu · Hal Daumé III · Furong Huang

**Abstract**: Despite recent progress in reinforcement learning (RL) from raw pixel data, sample inefficiency continues to present a substantial obstacle. Prior works have attempted to address this challenge by creating self-supervised auxiliary tasks, aiming to enrich the agent's learned representations with control-relevant information for future state prediction.However, these objectives are often insufficient to learn representations that can represent the optimal policy or value function, and they often consider tasks with small, abstract discrete action spaces and thus overlook the importance of action representation learning in continuous control.In this paper, we introduce $\texttt{TACO}$: $\textbf{T}$emporal $\textbf{A}$ction-driven $\textbf{CO}$ntrastive Learning, a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. $\texttt{TACO}$ simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states. Theoretically, $\texttt{TACO}$ can be shown to learn state and action representations that encompass sufficient information for control, thereby improving sample efficiency.For online RL, $\texttt{TACO}$ achieves 40% performance boost after one million environment interaction steps on average across nine challenging visual continuous control tasks from Deepmind Control Suite. In addition, we show that $\texttt{TACO}$ can also serve as a plug-and-play module adding to existing offline visual RL methods to establish the new state-of-the-art performance for offline visual RL across offline datasets with varying quality.

**Abstract(Chinese)**: 摘要：尽管最近在从原始像素数据中进行强化学习（RL）方面取得了进展，但样本效率依然是一个重大障碍。先前的研究尝试通过创建自监督辅助任务来解决这一挑战，旨在丰富代理程序学到的表示，以获得未来状态预测的与控制相关信息。然而，这些目标通常不足以学习能够表示最优策略或值函数的表示，并且它们经常考虑具有小型、抽象的离散动作空间的任务，从而忽视了在连续控制中动作表示学习的重要性。在本文中，我们引入了$	exttt{TACO}$：$	extbf{T}$emporal $	extbf{A}$ction-driven $	extbf{CO}$ntrastive Learning，这是一种简单但功能强大的时间对比学习方法，可促进代理程序同时获得潜在的状态和动作表示。$	exttt{TACO}$通过优化当前状态的表示与动作序列的表示以及相应未来状态的表示之间的互信息，同时学习状态和动作表示。从理论上讲，$	exttt{TACO}$可以被证明学习包含足够控制信息的状态和动作表示，从而提高样本效率。对于在线RL，在Deepmind Control Suite的九个具有挑战性的视觉连续控制任务中，在一百万个环境交互步骤后，$	exttt{TACO}$的性能平均提高了40％。此外，我们展示了$	exttt{TACO}$还可以作为即插即用模块添加到现有的离线视觉RL方法中，以在具有不同质量的离线数据集上建立新的离线视觉RL的最新性能。

**URL**: https://nips.cc/virtual/2023/poster/70929

---

## For SALE: State-Action Representation Learning for Deep Reinforcement Learning
**Author**: Scott Fujimoto · Wei-Di Chang · Edward Smith · Shixiang (Shane) Gu · Doina Precup · David Meger

**Abstract**: In reinforcement learning (RL), representation learning is a proven tool for complex image-based tasks, but is often overlooked for environments with low-level states, such as physical control problems. This paper introduces SALE, a novel approach for learning embeddings that model the nuanced interaction between state and action, enabling effective representation learning from low-level states. We extensively study the design space of these embeddings and highlight important design considerations. We integrate SALE and an adaptation of checkpoints for RL into TD3 to form the TD7 algorithm, which significantly outperforms existing continuous control algorithms. On OpenAI gym benchmark tasks, TD7 has an average performance gain of 276.7% and 50.7% over TD3 at 300k and 5M time steps, respectively, and works in both the online and offline settings.

**Abstract(Chinese)**: 在强化学习（RL）中，表示学习是复杂基于图像的任务的一种验证工具，但通常被忽视了对于低级状态的环境，比如物理控制问题。本文介绍了SALE，一种新颖的方法，用于学习嵌入，以建模状态和动作之间的微妙交互，从而实现从低级状态进行有效的表示学习。我们广泛研究了这些嵌入的设计空间，并强调了重要的设计考虑。我们将SALE与RL的检查点适应整合到TD3中，形成了TD7算法，该算法显着优于现有的连续控制算法。在OpenAI gym基准任务中，TD7在300k和5M时间步骤上的平均性能提升分别为276.7%和50.7%，比TD3高，而且在在线和离线设置中均有效。

**URL**: https://nips.cc/virtual/2023/poster/69999

---

## Sample Complexity of Goal-Conditioned Hierarchical Reinforcement Learning
**Author**: Arnaud Robert · Ciara Pike-Burke · Aldo Faisal

**Abstract**: Hierarchical Reinforcement Learning (HRL) algorithms can perform planning at multiple levels of abstraction. Empirical results have shown that state or temporal abstractions might significantly improve the sample efficiency of algorithms. Yet, we still do not have a complete understanding of the basis of those efficiency gains nor any theoretically grounded design rules. In this paper, we derive a lower bound on the sample complexity for the considered class of goal-conditioned HRL algorithms. The proposed lower bound empowers us to quantify the benefits of hierarchical decomposition and leads to the design of a simple Q-learning-type algorithm that leverages hierarchical decompositions. We empirically validate our theoretical findings by investigating the sample complexity of the proposed hierarchical algorithm on a spectrum of tasks (hierarchical $n$-rooms, Gymnasium's Taxi). The hierarchical $n$-rooms tasks were designed to allow us to dial their complexity over multiple orders of magnitude. Our theory and algorithmic findings provide a step towards answering the foundational question of quantifying the improvement hierarchical decomposition offers over monolithic solutions in reinforcement learning.

**Abstract(Chinese)**: 摘要：

分层强化学习（HRL）算法可以在多个抽象层次上进行规划。经验证实验结果表明，状态或时间抽象可能显著提高算法的样本效率。然而，我们仍然没有完全理解这些效率增益的基础，也没有任何理论上的设计规则。在本文中，我们推导了对于所考虑的目标条件HRL算法的样本复杂度下限。所提出的下限使我们能够量化分层分解的好处，并导致设计了一种简单的Q学习类型算法，利用分层分解。我们通过在一系列任务上（分层$n$间房，体育馆的出租车）调查所提出的分层算法的样本复杂度，从实证上验证了我们的理论发现。分层$n$间房任务的设计允许我们在多个数量级上调整其复杂性。我们的理论和算法发现为回答量化强化学习中分层分解提供了比整体解决方案提供了多大改进的基础性问题迈出了一步。

**URL**: https://nips.cc/virtual/2023/poster/72289

---

## RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization
**Author**: Siqi Shen · Chennan Ma · Chao Li · Weiquan Liu · Yongquan Fu · Songzhu Mei · Xinwang Liu · Cheng Wang

**Abstract**: Multi-agent systems are characterized by environmental uncertainty, varying policies of agents, and partial observability, which result in significant risks. In the context of Multi-Agent Reinforcement Learning (MARL), learning coordinated and decentralized policies that are sensitive to risk is challenging. To formulate the coordination requirements in risk-sensitive MARL, we introduce the Risk-sensitive Individual-Global-Max (RIGM) principle as a generalization of the Individual-Global-Max (IGM) and Distributional IGM (DIGM) principles. This principle requires that the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy. Current MARL value factorization methods do not satisfy the RIGM principle for common risk metrics such as the Value at Risk (VaR) metric or distorted risk measurements. Therefore, we propose RiskQ to address this limitation, which models the joint return distribution by modeling quantiles of it as weighted quantile mixtures of per-agent return distribution utilities. RiskQ satisfies the RIGM principle for the VaR and distorted risk metrics. We show that RiskQ can obtain promising performance through extensive experiments. The source code of RiskQ is available in https://github.com/xmu-rl-3dv/RiskQ.

**Abstract(Chinese)**: 多Agent系统的特点是环境不确定性、agent的政策变化以及部分可观察性，这导致了重大风险。在多Agent强化学习（MARL）的背景下，学习协调和去中心化政策对风险敏感的问题具有挑战性。为了在风险敏感MARL中制定协调要求，我们引入了风险敏感个体-全局最大（RIGM）原则，作为个体-全局最大（IGM）和分布式IGM（DIGM）原则的一般化。该原则要求每个Agent的风险敏感行动选择集合应等同于中央政策的风险敏感行动选择。当前的MARL值因子分解方法不能满足常见风险度量（如风险值（VaR）度量或扭曲风险度量）的RIGM原则。因此，我们提出RiskQ来解决这一限制，它通过将每个Agent的回报分布效用的加权分位数混合来对联合回报分布进行建模。RiskQ满足VaR和扭曲风险度量的RIGM原则。我们展示了RiskQ通过广泛的实验可以获得有希望的性能。RiskQ的源代码可在 https://github.com/xmu-rl-3dv/RiskQ 上获得。

**URL**: https://nips.cc/virtual/2023/poster/72240

---

## Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms
**Author**: Alexander Bukharin · Yan Li · Yue Yu · Qingru Zhang · Zhehui Chen · Simiao Zuo · Chao Zhang · Songan Zhang · Tuo Zhao

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown promising results across several domains. Despite this promise, MARL policies often lack robustness and are therefore sensitive to small changes in their environment. This presents a serious concern for the real world deployment of MARL algorithms, where the testing environment may slightly differ from the training environment. In this work we show that we can gain robustness by controlling a policy’s Lipschitz constant, and under mild conditions, establish the existence of a Lipschitz and close-to-optimal policy. Motivated by these insights, we propose a new robust MARL framework, ERNIE, that promotes the Lipschitz continuity of the policies with respect to the state observations and actions by adversarial regularization. The ERNIE framework provides robustness against noisy observations, changing transition dynamics, and malicious actions of agents. However, ERNIE’s adversarial regularization may introduce some training instability. To reduce this instability, we reformulate adversarial regularization as a Stackelberg game. We demonstrate the effectiveness of the proposed framework with extensive experiments in traffic light control and particle environments. In addition, we extend ERNIE to mean-field MARL with a formulation based on distributionally robust optimization that outperforms its non-robust counterpart and is of independent interest. Our code is available at https://github.com/abukharin3/ERNIE.

**Abstract(Chinese)**: 多智体强化学习（MARL）在多个领域取得了令人期待的成果。尽管如此，MARL策略通常缺乏鲁棒性，因此对其环境的微小变化非常敏感。这对于将MARL算法部署到现实世界中来说是一个严重的问题，因为测试环境可能与训练环境略有不同。在本研究中，我们展示了通过控制策略的Lipschitz常数，我们可以获得鲁棒性，并在温和条件下，建立了Lipschitz和接近最优策略的存在性。受到这些见解的启发，我们提出了一种新的鲁棒MARL框架ERNIE，通过对抗正则化促进策略相对于状态观察和动作的Lipschitz连续性。ERNIE框架提供了对于嘈杂观测、变化的转移动态以及智体恶意行为的鲁棒性。然而，ERNIE的对抗正则化可能会引入一些训练不稳定性。为了减少这种不稳定性，我们将对抗正则化重新构造为斯塔克尔伯格博弈。通过在交通灯控制和粒子环境中进行广泛实验，我们展示了所提出框架的有效性。此外，我们将ERNIE扩展到基于分布鲁棒优化的均场MARL，其公式优于其非鲁棒对应物，并具有独立的兴趣。我们的代码可在 https://github.com/abukharin3/ERNIE 上获得。

**URL**: https://nips.cc/virtual/2023/poster/72245

---

## Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning
**Author**: Matthias Gerstgrasser · Tom Danino · Sarah Keren

**Abstract**: We present a novel multi-agent RL approach, Selective Multi-Agent Prioritized Experience Relay, in which agents share with other agents a limited number of transitions they observe during training. The intuition behind this is that even a small number of relevant experiences from other agents could help each agent learn. Unlike many other multi-agent RL algorithms, this approach allows for largely decentralized training, requiring only a limited communication channel between agents. We show that our approach outperforms baseline no-sharing decentralized training and state-of-the art multi-agent RL algorithms. Further, sharing only a small number of highly relevant experiences outperforms sharing all experiences between agents, and the performance uplift from selective experience sharing is robust across a range of hyperparameters and DQN variants.

**Abstract(Chinese)**: 我们提出了一种新的多智能体强化学习方法，即选择性多智能体优先经验中继，其中智能体在训练过程中与其他智能体共享他们观察到的有限数量的过渡。这背后的直觉是，即使少量来自其他智能体的相关经验也可以帮助每个智能体学习。与许多其他多智能体强化学习算法不同，这种方法允许在很大程度上去中心化训练，只需要智能体之间的有限通信渠道。我们表明，我们的方法胜过基线无分享去中心化训练和最先进的多智能体强化学习算法。此外，仅分享少量高度相关的经验胜过在智能体之间分享所有经验，选择性经验分享所带来的性能提升在一系列超参数和DQN变体中都是稳健的。

**URL**: https://nips.cc/virtual/2023/poster/72350

---

## Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms
**Author**: Yashaswini Murthy · Mehrdad Moharrami · R. Srikant

**Abstract**: Many policy-based reinforcement learning (RL) algorithms can be viewed as instantiations of approximate policy iteration (PI), i.e., where policy improvement and policy evaluation are both performed approximately. In applications where the average reward objective is the meaningful performance metric, often discounted reward formulations are used with the discount factor being close to $1,$ which is equivalent to making the expected horizon very large. However, the corresponding theoretical bounds for error performance scale with the square of the horizon. Thus, even after dividing the total reward by the length of the horizon, the corresponding performance bounds for average reward problems go to infinity. Therefore, an open problem has been to obtain meaningful performance bounds for approximate PI and RL algorithms for the average-reward setting.  In this paper, we solve this open problem by obtaining the first non-trivial finite time error bounds for average-reward MDPs which go to zero in the limit as policy evaluation and policy improvement errors go to zero.

**Abstract(Chinese)**: 摘要：许多基于策略的强化学习（RL）算法可以被视为近似策略迭代（PI）的实例化，即策略改进和策略评估都是近似完成的。在平均回报目标是有意义的性能指标的应用中，通常会使用折现回报公式，其中折现因子接近 1，这相当于使预期视野非常大。然而，对应的误差性能的理论界限随着视野的平方而增加。因此，即使将总回报除以视野的长度，平均回报问题的相应性能界限也会趋于无穷大。因此，获得近似 PI 和 RL 算法对于平均回报设置的有意义性能界限一直是一个未解决的问题。在本文中，我们通过获得第一个非平凡的有限时间误差界限来解决这个未解决的问题，这些误差界限在策略评估和策略改进的误差趋于零时趋于零。

**URL**: https://nips.cc/virtual/2023/poster/71759

---

## Reward-agnostic Fine-tuning: Provable Statistical Benefits of Hybrid Reinforcement Learning
**Author**: Gen Li · Wenhao Zhan · Jason Lee · Yuejie Chi · Yuxin Chen

**Abstract**: This paper studies tabular reinforcement learning (RL) in the hybrid setting, which assumes access to both an offline dataset and online interactions with the unknown environment. A central question boils down to how to efficiently utilize online data to strengthen and complement the offline dataset and enable effective policy fine-tuning. Leveraging recent advances in reward-agnostic exploration and offline RL, we design a three-stage hybrid RL algorithm that beats the best of both worlds --- pure offline RL and pure online RL --- in terms of sample complexities. The proposed algorithm does not require any reward information during data collection. Our theory is developed based on a new notion called single-policy partial concentrability, which captures the trade-off between distribution mismatch and miscoverage and guides the interplay between offline and online data.

**Abstract(Chinese)**: 这篇论文研究了表格强化学习（RL）在混合环境中的情况，该情况假定可以同时访问离线数据集和与未知环境的在线交互。一个核心问题在于如何有效利用在线数据来加强和补充离线数据集，并实现有效的策略微调。借助最近的奖励无关探索和离线RL的进展，我们设计了一个三阶段混合RL算法，其在样本复杂性方面击败了最优秀的两种世界——纯离线RL和纯在线RL。所提出的算法在数据收集过程中不需要任何奖励信息。我们的理论是基于一种称为单策略部分可集中性的新概念发展而来，该概念捕捉了分布不匹配和覆盖不足之间的权衡，并指导了离线和在线数据之间的相互作用。

**URL**: https://nips.cc/virtual/2023/poster/71851

---

## Anytime-Competitive Reinforcement Learning with Policy Prior
**Author**: Jianyi Yang · Pengfei Li · Tongxin Li · Adam Wierman · Shaolei Ren

**Abstract**: This paper studies the problem of Anytime-Competitive Markov Decision Process (A-CMDP). Existing works on Constrained Markov Decision Processes (CMDPs) aim to optimize the expected reward while constraining the expected cost over random dynamics, but the cost in a specific episode can still be unsatisfactorily high. In contrast, the goal of A-CMDP is to optimize the expected reward while guaranteeing a bounded cost in each round of any episode against a policy prior. We propose a new algorithm, called Anytime-Competitive Reinforcement Learning (ACRL), which provably guarantees the anytime cost constraints. The regret analysis shows the policy asymptotically matches the optimal reward achievable under the anytime competitive constraints. Experiments on the application of carbon-intelligent computing verify the reward performance and cost constraint guarantee of ACRL.

**Abstract(Chinese)**: 本文研究了Anytime-Competitive Markov决策过程（A-CMDP）的问题。现有的关于约束马尔可夫决策过程（CMDP）的研究旨在优化预期奖励，同时约束随机动态下的预期成本，但在特定情节中的成本仍可能不令人满意。相反，A-CMDP的目标是在对策略优先的任何情节的每一轮中保证有界成本的同时优化预期奖励。我们提出了一种新的算法，称为Anytime-Competitive Reinforcement Learning（ACRL），它可以证明保证了任意时间成本约束。遗憾分析表明，该策略在渐近条件下与任意竞争性约束下可实现的最优奖励相匹配。对碳智能计算应用的实验验证了ACRL的奖励性能和成本约束保证。

**URL**: https://nips.cc/virtual/2023/poster/72273

---

## Corruption-Robust Offline Reinforcement Learning with General Function Approximation
**Author**: Chenlu Ye · Rui Yang · Quanquan Gu · Tong Zhang

**Abstract**: We investigate the problem of corruption robustness in offline reinforcement learning (RL) with general function approximation, where an adversary can corrupt each sample in the offline dataset, and the corruption level $\zeta\geq0$ quantifies the cumulative corruption amount over $n$ episodes and $H$ steps. Our goal is to find a policy that is robust to such corruption and minimizes the suboptimality gap with respect to the optimal policy for the uncorrupted Markov decision processes (MDPs). Drawing inspiration from the uncertainty-weighting technique from the robust online RL setting \citep{he2022nearly,ye2022corruptionrobust}, we design a new uncertainty weight iteration procedure to efficiently compute on batched samples and propose a corruption-robust algorithm for offline RL. Notably, under the assumption of single policy coverage and the knowledge of $\zeta$, our proposed algorithm achieves a suboptimality bound that is worsened by an additive factor of $\mathcal O(\zeta \cdot (\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H))^{1/2} (C(\hat{\mathcal F},\mu))^{-1/2} n^{-1})$ due to the corruption. Here $\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H)$ is the coverage coefficient that depends on the regularization parameter $\lambda$, the confidence set $\hat{\mathcal F}$, and the dataset $\mathcal Z_n^H$, and $C(\hat{\mathcal F},\mu)$ is a coefficient that depends on $\hat{\mathcal F}$ and the underlying data distribution $\mu$. When specialized to linear MDPs, the corruption-dependent error term reduces to $\mathcal O(\zeta d n^{-1})$ with $d$ being the dimension of the feature map, which matches the existing lower bound for corrupted linear MDPs. This suggests that our analysis is tight in terms of the corruption-dependent term.

**Abstract(Chinese)**: 我们研究了离线强化学习中通用函数逼近的腐败鲁棒性问题，其中敌对方可以破坏离线数据集中的每个样本，而腐败程度$\zeta\geq0$量化了$n$个周期和$H$步中的累积腐败量。我们的目标是找到一个对这种腐败具有鲁棒性的策略，并最小化与未腐败马尔可夫决策过程（MDPs）的最优策略的次优差距。灵感来自于鲁棒在线强化学习环境中的不确定性加权技术 \citep{he2022nearly,ye2022corruptionrobust}，我们设计了一种新的不确定性权重迭代过程，以便在批处理样本上高效计算，并提出了一种用于离线强化学习的腐败鲁棒算法。值得注意的是，在单策略覆盖和对$\zeta$的了解的假设下，我们提出的算法实现了一个次优性界限，该界限因腐败而恶化了一个附加因子$\mathcal O(\zeta \cdot (\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H))^{1/2} (C(\hat{\mathcal F},\mu))^{-1/2} n^{-1})$。这里$\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H)$是依赖于正则化参数$\lambda$、置信区间$\hat{\mathcal F}$和数据集$\mathcal Z_n^H$的覆盖系数，而$C(\hat{\mathcal F},\mu)$是一个依赖于$\hat{\mathcal F}$和基础数据分布$\mu$的系数。当特化为线性MDPs时，腐败依赖的误差项减小为$\mathcal O(\zeta d n^{-1})$，其中$d$是特征映射的维数，这与已有的腐败线性MDPs的下界相一致。这表明我们的分析在腐败依赖项方面是紧凑的。

**URL**: https://nips.cc/virtual/2023/poster/72027

---

## A Partially-Supervised Reinforcement Learning Framework for Visual Active Search
**Author**: Anindya Sarkar · Nathan Jacobs · Yevgeniy Vorobeychik

**Abstract**: Visual active search (VAS) has been proposed as a  modeling framework in which visual cues are used to guide exploration, with the goal of identifying regions of interest in a large geospatial area. Its potential applications include identifying hot spots of rare wildlife poaching activity, search-and-rescue scenarios, identifying illegal trafficking of weapons, drugs, or people, and many others. State of the art approaches to VAS include applications of deep reinforcement learning (DRL), which yield end-to-end search policies, and traditional active search, which combines predictions with custom algorithmic approaches. While the DRL framework has been shown to greatly outperform traditional active search in such domains, its end-to-end nature does not make full use of supervised information attained either during training, or during actual search, a significant limitation if search tasks differ significantly from those in the training distribution. We propose an approach that combines the strength of both DRL and conventional active search approaches by decomposing the search policy into a prediction module, which produces a geospatial distribution of regions of interest based on task embedding and search history, and a search module, which takes the predictions and search history as input and outputs the search distribution. In addition, we develop a novel meta-learning approach for jointly learning the resulting combined policy that can make effective use of supervised information obtained both at training and decision time. Our extensive experiments demonstrate that the proposed representation and meta-learning frameworks significantly outperform state of the art in visual active search on several problem domains.

**Abstract(Chinese)**: 可视化主动搜索（VAS）已被提出作为一种建模框架，在这种框架中，利用视觉线索来引导探索，以识别大型地理空间区域中的感兴趣区域为目标。其潜在应用包括识别罕见的野生动物偷猎热点、搜救场景、识别非法武器、毒品或人口贩运等。VAS的最新方法包括深度强化学习（DRL）的应用，可以产生端到端搜索策略，以及传统的主动搜索，结合了预测和定制算法方法。尽管在这些领域，DRL框架已被证明远远优于传统的主动搜索，但它的端到端性质并未充分利用在训练过程中或实际搜索中获得的监督信息，若搜索任务与训练分布中的任务差异显著，则会受到较大限制。我们提出了一种方法，通过将搜索策略分解为预测模块和搜索模块的结合来充分发挥DRL和传统主动搜索方法的优势。预测模块根据任务嵌入和搜索历史生成感兴趣区域的地理空间分布，搜索模块根据预测和搜索历史作为输入，并输出搜索分布。此外，我们开发了一种新颖的元学习方法，可联合学习所得的复合策略，并能够有效利用在训练和决策时间中获得的监督信息。我们广泛的实验证明了所提出的表示和元学习框架在几个问题领域的可视化主动搜索中远远优于最新技术。

**URL**: https://nips.cc/virtual/2023/poster/71640

---

## Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning
**Author**: Fan Feng · Sara Magliacane

**Abstract**: In many reinforcement learning tasks, the agent has to learn to interact with many objects of different types and generalize to unseen combinations and numbers of objects. Often a task is a composition of previously learned tasks (e.g. block stacking).These are examples of compositional generalization, in which we compose object-centric representations to solve complex tasks. Recent works have shown the benefits of object-factored representations and hierarchical abstractions for improving sample efficiency in these settings. On the other hand, these methods do not fully exploit the benefits of factorization in terms of object attributes. In this paper, we address this opportunity and introduce the Dynamic Attribute FacTored RL (DAFT-RL) framework. In DAFT-RL, we leverage object-centric representation learning to extract objects from visual inputs. We learn to classify them into classes and infer their latent parameters. For each class of object, we learn a class template graph that describes how the dynamics and reward of an object of this class factorize according to its attributes. We also learn an interaction pattern graph that describes how objects of different classes interact with each other at the attribute level. Through these graphs and a dynamic interaction graph that models the interactions between objects, we can learn a policy that can then be directly applied in a new environment by estimating the interactions and latent parameters.We evaluate DAFT-RL in three benchmark datasets and show our framework outperforms the state-of-the-art in generalizing across unseen objects with varying attributes and latent parameters, as well as in the composition of previously learned tasks.

**Abstract(Chinese)**: 在许多强化学习任务中，代理程序必须学会与许多不同类型的对象进行交互，并且推广到看不见的对象组合和数量。通常任务是先前学习任务的组合（例如，积木堆叠）。这些是组合泛化的例子，其中我们组合以对象为中心的表示来解决复杂的任务。最近的研究表明，在这些设置中，采用对象分解表示和分层抽象有助于提高样本效率。另一方面，这些方法并未充分利用因子分解在对象属性方面的益处。在本文中，我们着眼于这一机会，并引入了动态属性分解强化学习（DAFT-RL）框架。在DAFT-RL中，我们利用以对象为中心的表示学习从视觉输入中提取对象。我们学会对它们进行分类，并推断它们的潜在参数。对于每个对象类别，我们学习一个描述该类对象的动态模板图，该图描述了该类对象的动态和奖励是如何根据其属性进行分解的。我们还学习了一个交互模式图，描述了不同类别的对象如何在属性级别上相互作用。通过这些图以及描述对象之间相互作用的动态交互图，我们可以学习一种策略，然后通过估计交互和潜在参数，直接应用于新环境。我们在三个基准数据集中评估了DAFT-RL，并展示了我们的框架在泛化到具有不同属性和潜在参数的未见对象以及在以前学习的任务组合方面的表现优于最新技术成果。

**URL**: https://nips.cc/virtual/2023/poster/71123

---

## Offline Reinforcement Learning for Mixture-of-Expert Dialogue Management
**Author**: Dhawal Gupta · Yinlam Chow · Azamat Tulepbergenov · Mohammad Ghavamzadeh · Craig Boutilier

**Abstract**: Reinforcement learning (RL) has shown great promise for developing agents for dialogue management (DM) that are non-myopic, conduct rich conversations, and maximize overall user satisfaction. Despite the advancements in RL and language models (LMs), employing RL to drive conversational chatbots still poses significant challenges. A primary issue stems from RL’s dependency on online exploration for effective learning, a process that can be costly. Moreover, engaging in online interactions with humans during the training phase can raise safety concerns, as the LM can potentially generate unwanted outputs. This issue is exacerbated by the combinatorial action spaces facing these algorithms, as most LM agents generate responses at the word level. We develop various RL algorithms, specialized in dialogue planning, that leverage recent Mixture-of-Expert Language Models (MoE-LMs)---models that capture diverse semantics, generate utterances reflecting different intents, and are amenable for multi-turn DM. By exploiting the MoE-LM structure, our methods significantly reduce the size of the action space and improve the efficacy of RL-based DM. We evaluate our methods in open-domain dialogue to demonstrate their effectiveness with respect to the diversity of intent in generated utterances and overall DM performance.

**Abstract(Chinese)**: 强化学习（RL）已经显示出为发展非短视、进行丰富对话并最大化用户满意度的对话管理代理方面具有巨大潜力。尽管RL和语言模型（LMs）方面有所进展，但是在驱动对话聊天机器人上仍然存在着很大的挑战。主要问题之一源自RL对于有效学习所需进行的在线探索的依赖，这是一个可能代价高昂的过程。此外，在训练阶段与人类进行在线交互可能会引发安全问题，因为LM可能会产生不需要的输出。这一问题受到算法面临的组合动作空间的挑战的加剧，因为大多数LM代理以单词级别生成响应。我们开发了各种专门用于对话规划的RL算法，利用最近的专家混合语言模型（MoE-LMs）---这些模型能够捕捉多样的语义，生成反映不同意图的话语，并且适用于多轮对话管理。通过利用MoE-LM结构，我们的方法显著减小了动作空间的大小，并提高了基于RL的对话管理的效果。我们在开放领域对话中评估了我们的方法，以展示它们在生成话语中意图多样性和整体对话管理性能方面的有效性。

**URL**: https://nips.cc/virtual/2023/poster/71578

---

## Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation
**Author**: Nikki Lijing Kuang · Ming Yin · Mengdi Wang · Yu-Xiang Wang · Yian Ma

**Abstract**: Recent studies in reinforcement learning (RL) have made significant progress by leveraging function approximation to alleviate the sample complexity hurdle for better performance. Despite the success, existing provably efficient algorithms typically rely on the accessibility of immediate feedback upon taking actions. The failure to account for the impact of delay in observations can significantly degrade the performance of real-world systems due to the regret blow-up. In this work, we tackle the challenge of delayed feedback in RL with linear function approximation by employing posterior sampling, which has been shown to empirically outperform the popular UCB algorithms in a wide range of regimes. We first introduce \textit{Delayed-PSVI}, an optimistic value-based algorithm that effectively explores the value function space via noise perturbation with posterior sampling. We provide the first analysis for posterior sampling algorithms with delayed feedback in RL and show our algorithm achieves $\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 \mathbb{E}[\tau])$ worst-case regret in the presence of unknown stochastic delays. Here $\mathbb{E}[\tau]$ is the expected delay. To further improve its computational efficiency and to expand its applicability in high-dimensional RL problems, we incorporate a gradient-based approximate sampling scheme via Langevin dynamics for \textit{Delayed-LPSVI}, which maintains the same order-optimal regret guarantee with $\widetilde{O}(dHK)$ computational cost. Empirical evaluations are performed to demonstrate the statistical and computational efficacy of our algorithms.

**Abstract(Chinese)**: 在最近的强化学习（RL）研究中，通过利用函数逼近来显著提高进展，以减轻样本复杂性障碍，从而获得更好的性能。尽管取得了成功，现有的可以证明有效的算法通常依赖于在采取行动后能够立即获得反馈的可访问性。未能考虑观察延迟的影响可能会导致真实世界系统性能的显著下降，因为后悔会急剧增加。在本文中，我们通过使用后验抽样来解决强化学习中延迟反馈的挑战，后验抽样在各种范围内的实证表现已经超越了流行的UCB算法。我们首先介绍\textit {Delayed-PSVI}，这是一种乐观的基于价值的算法，通过后验抽样的噪声扰动有效地探索价值函数空间。我们为具有延迟反馈的后验抽样算法提供了首次分析，并展示了我们的算法在未知随机延迟存在的情况下实现了$\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 \mathbb{E}[\tau])$的最坏情况后悔。这里$\mathbb{E}[\tau]$是预期延迟。为了进一步提高其计算效率并扩大其在高维RL问题中的适用性，我们通过Langevin动力学引入了基于梯度的近似抽样方案，用于\textit {Delayed-LPSVI}，它保持相同顺序最优的后悔保证，并且计算成本为$\widetilde{O}(dHK)$。进行了经验评估，以展示我们算法的统计和计算效力。

**URL**: https://nips.cc/virtual/2023/poster/71646

---

## Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning
**Author**: Stefan Stojanovic · Yassir Jedra · Yassir Jedra · Alexandre Proutiere

**Abstract**: We study matrix estimation problems arising in reinforcement learning with low-rank structure. In low-rank bandits, the matrix to be recovered specifies the expected arm rewards, and for low-rank Markov Decision Processes (MDPs), it characterizes the transition kernel of the MDP. In both cases, each entry of the matrix carries important information, and we seek estimation methods with low entry-wise prediction error. Importantly, these methods further need to accommodate for inherent correlations in the available data (e.g. for MDPs, the data consists of system trajectories). We investigate the performance of  simple spectral-based matrix estimation approaches: we show that they efficiently recover the singular subspaces of the matrix and exhibit nearly-minimal entry-wise prediction error. These new results on low-rank matrix estimation make it possible to devise reinforcement learning algorithms that fully exploit the underlying low-rank structure. We provide two examples of such algorithms: a regret minimization algorithm for low-rank bandit problems, and a best policy identification algorithm for low-rank MDPs. Both algorithms yield state-of-the-art performance guarantees.

**Abstract(Chinese)**: 我们研究在具有低秩结构的强化学习中出现的矩阵估计问题。在低秩赌博机中，要恢复的矩阵指定了预期的臂奖励，而对于低秩马尔可夫决策过程（MDPs），它表征了MDP的转移核。在这两种情况下，矩阵的每个条目都携带重要信息，我们寻求具有低逐条条目预测误差的估计方法。重要的是，这些方法需要进一步适应可用数据中的固有相关性（例如对于MDPs，数据由系统轨迹组成）。我们研究了简单的基于谱的矩阵估计方法的性能：我们表明它们能够有效恢复矩阵的奇异子空间，并表现出几乎最小的逐项预测误差。这些关于低秩矩阵估计的新结果使得可能设计全面利用基础低秩结构的强化学习算法。我们提供了两个这样算法的示例：一种针对低秩赌博机问题的遗憾最小化算法，以及一种针对低秩MDPs的最佳策略识别算法。这两种算法都提供了最先进的性能保证。

**URL**: https://nips.cc/virtual/2023/poster/71206

---

## A Definition of Continual Reinforcement Learning
**Author**: David Abel · Andre Barreto · Benjamin Van Roy · Doina Precup · Hado van Hasselt · Satinder Singh

**Abstract**: In a standard view of the reinforcement learning problem, an agent’s goal is to efficiently identify a policy that maximizes long-term reward. However, this perspective is based on a restricted view of learning as finding a solution, rather than treating learning as endless adaptation. In contrast, continual reinforcement learning refers to the setting in which the best agents never stop learning. Despite the importance of continual reinforcement learning, the community lacks a simple definition of the problem that highlights its commitments and makes its primary concepts precise and clear. To this end, this paper is dedicated to carefully defining the continual reinforcement learning problem. We formalize the notion of agents that “never stop learning” through a new mathematical language for analyzing and cataloging agents. Using this new language, we define a continual learning agent as one that can be understood as carrying out an implicit search process indefinitely, and continual reinforcement learning as the setting in which the best agents are all continual learning agents. We provide two motivating examples, illustrating that traditional views of multi-task reinforcement learning and continual supervised learning are special cases of our definition. Collectively, these definitions and perspectives formalize many intuitive concepts at the heart of learning, and open new research pathways surrounding continual learning agents.

**Abstract(Chinese)**: 在标准的强化学习问题视角中，代理的目标是有效地确定最大化长期回报的策略。然而，这一观点基于将学习视为寻找解决方案的受限观点，而非将学习视为无休止的适应。相比之下，持续强化学习指的是最优代理永不停止学习的情境。尽管持续强化学习的重要性不言而喻，但学术界缺乏一个突出其承诺并使其主要概念准确清晰的简单问题定义。因此，本文致力于仔细定义持续强化学习问题。我们通过一种新的数学语言对“永不停止学习”的代理概念进行了形式化。利用这种新语言，我们将持续学习代理定义为能被理解为无限实施隐式搜索过程的代理，并将持续强化学习定义为最优代理均为持续学习代理的情境。我们提供了两个激励性例子，说明传统的多任务强化学习和持续监督学习视角是我们定义的特例。总的来说，这些定义和观点将学习核心直觉概念形式化，并为围绕持续学习代理的新研究路径打开了大门。

**URL**: https://nips.cc/virtual/2023/poster/71231

---

## Inverse Reinforcement Learning with the Average Reward Criterion
**Author**: Feiyang Wu · Jingyang Ke · Anqi Wu

**Abstract**: We study the problem of Inverse Reinforcement Learning (IRL) with an average-reward criterion. The goal is to recover an unknown policy and a reward function when the agent only has samples of states and actions from an experienced agent. Previous IRL methods assume that the expert is trained in a discounted environment, and the discount factor is known. This work alleviates this assumption by proposing an average-reward framework with efficient learning algorithms. We develop novel stochastic first-order methods to solve the IRL problem under the average-reward setting, which requires solving an Average-reward Markov Decision Process (AMDP) as a subproblem. To solve the subproblem, we develop a Stochastic Policy Mirror Descent (SPMD) method under general state and action spaces that needs $\mathcal{O}(1/\varepsilon)$ steps of gradient computation. Equipped with SPMD, we propose the Inverse Policy Mirror Descent (IPMD) method for solving the IRL problem with a $\mathcal{O}(1/\varepsilon^2)$ complexity. To the best of our knowledge, the aforementioned complexity results are new in IRL with the average reward criterion. Finally, we corroborate our analysis with numerical experiments using the MuJoCo benchmark and additional control tasks.

**Abstract(Chinese)**: 我们研究了具有平均奖励标准的逆强化学习（IRL）问题。目标是在只有来自有经验的代理的状态和行为样本的情况下，恢复未知策略和奖励函数。先前的IRL方法假设专家在折扣环境中接受训练，并且折扣因子是已知的。这项工作通过提出一个具有高效学习算法的平均奖励框架来减轻这一假设。我们开发了新颖的随机一阶方法来解决平均奖励环境下的IRL问题，这需要解决作为子问题的平均奖励马尔可夫决策过程（AMDP）。为了解决子问题，我们开发了一种随机策略镜像下降（SPMD）方法，适用于一般的状态和行为空间，需要$\mathcal{O}(1/\varepsilon)$步的梯度计算。利用SPMD，我们提出了逆策略镜像下降（IPMD）方法，用于解决具有$\mathcal{O}(1/\varepsilon^2)$复杂度的IRL问题。据我们所知，上述复杂度结果在具有平均奖励标准的IRL中是新的。最后，我们通过使用MuJoCo基准和额外的控制任务进行数值实验来证实我们的分析。

**URL**: https://nips.cc/virtual/2023/poster/71307

---

## Context Shift Reduction for Offline Meta-Reinforcement Learning
**Author**: Yunkai Gao · Rui Zhang · Jiaming Guo · Fan Wu · Qi Yi · Shaohui Peng · Siming Lan · Ruizhi Chen · Zidong Du · Xing Hu · Qi Guo · Ling Li · Yunji Chen

**Abstract**: Offline meta-reinforcement learning (OMRL) utilizes pre-collected offline datasets to enhance the agent's generalization ability on unseen tasks. However, the context shift problem arises due to the distribution discrepancy between the contexts used for training (from the behavior policy) and testing (from the exploration policy). The context shift problem leads to incorrect task inference and further deteriorates the generalization ability of the meta-policy. Existing OMRL methods either overlook this problem or attempt to mitigate it with additional information. In this paper, we propose a novel approach called Context Shift Reduction for OMRL (CSRO) to address the context shift problem with only offline datasets. The key insight of CSRO is to minimize the influence of policy in context during both the meta-training and meta-test phases.  During meta-training, we design a max-min mutual information representation learning mechanism to diminish the impact of the behavior policy on task representation. In the meta-test phase, we introduce the non-prior context collection strategy to reduce the effect of the exploration policy. Experimental results demonstrate that CSRO significantly reduces the context shift and improves the generalization ability, surpassing previous methods across various challenging domains.

**Abstract(Chinese)**: 离线元强化学习（OMRL）利用预先收集的离线数据集来增强代理程序在未见任务上的泛化能力。然而，由于训练（来自行为策略）和测试（来自探索策略）中的上下文分布差异，出现了上下文转移问题。上下文转移问题导致了错误的任务推断，并进一步损害了元策略的泛化能力。现有的OMRL方法要么忽视这个问题，要么试图通过额外信息来减轻它。在本文中，我们提出了一种称为离线元强化学习上下文转移减少（CSRO）的新方法，以仅利用离线数据集来解决上下文转移问题。CSRO的关键见解是在元训练和元测试阶段都要最小化策略在上下文中的影响。在元训练期间，我们设计了一个最大-最小互信息表示学习机制，以减少行为策略对任务表示的影响。在元测试阶段，我们引入非先验上下文收集策略来减少探索策略的影响。实验结果表明，CSRO显著减少了上下文转移并改善了泛化能力，优于先前方法在各种具有挑战性的领域中。

**URL**: https://nips.cc/virtual/2023/poster/71558

---

## Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning
**Author**: Jing Zhang · Chi Zhang · Wenjia Wang · Bingyi Jing

**Abstract**: Due to the inability to interact with the environment, offline reinforcement learning (RL) methods face the challenge of estimating the Out-of-Distribution (OOD) points. Existing methods for addressing this issue either control policy to exclude the OOD action or make the $Q$ function pessimistic. However, these methods can be overly conservative or fail to identify OOD areas accurately. To overcome this problem, we propose a Constrained Policy optimization with Explicit Behavior density (CPED) method that utilizes a flow-GAN model to explicitly estimate the density of behavior policy. By estimating the explicit density, CPED can accurately identify the safe region and enable exploration within the region, resulting in less conservative learning policies.  We further provide theoretical results for both the flow-GAN estimator and performance guarantee for CPED by showing that CPED can find the optimal $Q$-function value. Empirically, CPED outperforms existing alternatives on various standard offline reinforcement learning tasks, yielding higher expected returns.

**Abstract(Chinese)**: 由于无法与环境交互，离线强化学习（RL）方法面临着估计分布外（OOD）点的挑战。现有解决此问题的方法要么控制策略以排除OOD动作，要么让$Q$函数保守。然而，这些方法可能过分保守或无法准确识别OOD区域。为了克服这一问题，我们提出了一种利用流-GAN模型显式估计行为策略密度的受约束策略优化方法（CPED）。通过估计显式密度，CPED能够准确识别安全区域，并使探索在该区域内，从而导致更不保守的学习策略。我们进一步提供了流-GAN估计器的理论结果以及CPED的性能保证，通过表明CPED可以找到最优$Q$-函数值。在经验上，CPED在各种标准离线强化学习任务上表现优于现有替代方案，产生更高的期望回报。

**URL**: https://nips.cc/virtual/2023/poster/71027

---

## Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents
**Author**: Woojun Kim · Yongjae Shin · Jongeui Park · Youngchul Sung

**Abstract**: Deep reinforcement learning (RL) has achieved remarkable success in solving complex tasks through its integration with deep neural networks (DNNs) as function approximators. However, the reliance on DNNs has introduced a new challenge called primacy bias, whereby these function approximators tend to prioritize early experiences, leading to overfitting. To alleviate this bias, a reset method has been proposed, which involves periodic resets of a portion or the entirety of a deep RL agent while preserving the replay buffer. However, the use of this method can result in performance collapses after executing the reset, raising concerns from the perspective of safe RL and regret minimization. In this paper, we propose a novel reset-based method that leverages deep ensemble learning to address the limitations of the vanilla reset method and enhance sample efficiency. The effectiveness of the proposed method is validated through various experiments including those in the domain of safe RL. Numerical results demonstrate its potential for real-world applications requiring high sample efficiency and safety considerations.

**Abstract(Chinese)**: 深度强化学习（RL）通过与深度神经网络（DNNs）的集成作为函数逼近器，在解决复杂任务方面取得了显著成功。然而，对于DNNs的依赖引入了一个新的挑战，称为首要偏差，其中这些函数逼近器倾向于优先考虑早期经验，导致过度拟合。为了减轻这种偏差，提出了一种重置方法，其中涉及周期性重置深度RL代理的一部分或全部，并保留回放缓冲区。然而，使用该方法可能会导致重置后的性能下降，从安全RL和遗憾最小化的角度引发担忧。本文提出了一种利用深度集成学习的新重置方法，以解决普通重置方法的局限性，提高样本效率。所提出的方法的有效性通过各种实验证明，包括在安全RL领域的实验。数值结果证明了它在需要高样本效率和安全考虑的真实世界应用中的潜力。

**URL**: https://nips.cc/virtual/2023/poster/71142

---

## PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning
**Author**: Hojoon Lee · Hanseul Cho · HYUNSEUNG KIM · DAEHOON GWAK · Joonkee Kim · Jaegul Choo · Se-Young Yun · Chulhee Yun

**Abstract**: In Reinforcement Learning (RL), enhancing sample efficiency is crucial, particularly in scenarios when data acquisition is costly and risky. In principle, off-policy RL algorithms can improve sample efficiency by allowing multiple updates per environment interaction. However, these multiple updates often lead the model to overfit to earlier interactions, which is referred to as the loss of plasticity. Our study investigates the underlying causes of this phenomenon by dividing plasticity into two aspects. Input plasticity, which denotes the model's adaptability to changing input data, and label plasticity, which denotes the model's adaptability to evolving input-output relationships. Synthetic experiments on the CIFAR-10 dataset reveal that finding smoother minima of loss landscape enhances input plasticity, whereas refined gradient propagation improves label plasticity. Leveraging these findings, we introduce the PLASTIC algorithm, which harmoniously combines techniques to address both concerns. With minimal architectural modifications, PLASTIC achieves competitive performance on benchmarks including Atari-100k and Deepmind Control Suite. This result emphasizes the importance of preserving the model's plasticity to elevate the sample efficiency in RL. The code is available at https://github.com/dojeon-ai/plastic.

**Abstract(Chinese)**: 在强化学习（RL）中，提高样本效率至关重要，特别是在数据获取成本高昂且风险高的情况下。原则上，离线策略RL算法可以通过允许每个环境交互进行多次更新来提高样本效率。然而，这些多次更新通常会导致模型过度拟合先前的交互，这被称为可塑性的丢失。我们的研究通过将可塑性分为两个方面来调查这一现象。输入可塑性表示模型对不断变化的输入数据的适应能力，标签可塑性表示模型对不断演变的输入输出关系的适应能力。对CIFAR-10数据集的合成实验表明，寻找更平缓的损失景观最小值可以增强输入可塑性，而精细的梯度传播则改善了标签可塑性。利用这些发现，我们推出了PLASTIC算法，它和谐地结合了解决这两个问题的技术。通过最小的架构修改，PLASTIC在包括Atari-100k和Deepmind Control Suite在内的基准测试上取得了竞争性的表现。这一结果强调了保持模型的可塑性对提高RL中的样本效率的重要性。代码可在https://github.com/dojeon-ai/plastic 上获得。

**URL**: https://nips.cc/virtual/2023/poster/71114

---

## Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning
**Author**: Seungyong Moon · Junyoung Yeom · Bumsoo Park · Hyun Oh Song

**Abstract**: Discovering achievements with a hierarchical structure in procedurally generated environments presents a significant challenge.This requires an agent to possess a broad range of abilities, including generalization and long-term reasoning. Many prior methods have been built upon model-based or hierarchical approaches, with the belief that an explicit module for long-term planning would be advantageous for learning hierarchical dependencies. However, these methods demand an excessive number of environment interactions or large model sizes, limiting their practicality. In this work, we demonstrate that proximal policy optimization (PPO), a simple yet versatile model-free algorithm, outperforms previous methods when optimized with recent implementation practices. Moreover, we find that the PPO agent can predict the next achievement to be unlocked to some extent, albeit with limited confidence. Based on this observation, we introduce a novel contrastive learning method, called achievement distillation, which strengthens the agent's ability to predict the next achievement. Our method exhibits a strong capacity for discovering hierarchical achievements and shows state-of-the-art performance on the challenging Crafter environment in a sample-efficient manner while utilizing fewer model parameters.

**Abstract(Chinese)**: 在程序生成的环境中发现具有分层结构的成就是一个重大挑战。这需要一个代理具备广泛的能力，包括泛化和长期推理。许多先前的方法都建立在基于模型或分层方法的基础上，他们认为一个明确的长期规划模块对学习分层依赖关系是有利的。然而，这些方法需要大量的环境交互或大型模型尺寸，限制了它们的实际应用性。在这项工作中，我们证明了近端策略优化（PPO）这一简单而多才多艺的无模型算法，在最新的实施实践下优于先前的方法。此外，我们发现 PPO 代理可以在一定程度上预测接下来需要解锁的成就，尽管信心有限。基于这一观察，我们引入了一种新颖的对比学习方法，称为成就提炼，它增强了代理预测下一个成就的能力。我们的方法在发现分层成就方面表现出强大的能力，并展示了在 Crafter 环境中以高效的方式展现出最新性能，同时利用更少的模型参数。

**URL**: https://nips.cc/virtual/2023/poster/71091

---

## STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning
**Author**: Weipu Zhang · Gang Wang · Jian Sun · Yetian Yuan · Gao Huang

**Abstract**: Recently, model-based reinforcement learning algorithms have demonstrated remarkable efficacy  in visual input environments. These approaches begin by constructing a parameterized simulation world model of the real environment through self-supervised learning. By leveraging the imagination of the world model, the agent's policy is enhanced without the constraints of sampling from the real environment. The performance of these algorithms heavily relies on the sequence modeling and generation capabilities of the world model. However, constructing a perfectly accurate model of a complex unknown environment is nearly impossible. Discrepancies between the model and reality may cause the agent to pursue virtual goals, resulting in subpar performance in the real environment. Introducing random noise into model-based reinforcement learning has been proven beneficial.In this work, we introduce Stochastic Transformer-based wORld Model (STORM), an efficient world model architecture that combines the strong sequence modeling and generation capabilities of Transformers with the stochastic nature of variational autoencoders. STORM achieves a mean human performance of $126.7\%$ on the Atari $100$k benchmark, setting a new record among state-of-the-art methods that do not employ lookahead search techniques. Moreover, training an agent with $1.85$ hours of real-time interaction experience on a single NVIDIA GeForce RTX 3090 graphics card requires only $4.3$ hours, showcasing improved efficiency compared to previous methodologies.

**Abstract(Chinese)**: 最近，基于模型的强化学习算法在视觉输入环境中表现出了显著的功效。这些方法首先通过自监督学习构建真实环境的参数化仿真世界模型。通过利用世界模型的想象力，增强了 agent 的策略，而不受来自真实环境的采样的约束。这些算法的性能在很大程度上依赖于世界模型的序列建模和生成能力。然而，构建一个对复杂未知环境完全准确的模型几乎是不可能的。模型和现实之间的差异可能会导致 agent 追求虚拟目标，在真实环境中表现不佳。在基于模型的强化学习中引入随机噪声已被证明是有益的。在这项工作中，我们介绍了一种称为 STORM（Stochastic Transformer-based wORld Model）的高效世界模型架构，它结合了 Transformer 强大的序列建模和生成能力以及变分自动编码器的随机特性。STORM 在 Atari 100k 基准上达到了 126.7% 的人类平均表现，创下了不使用前瞻搜索技术的最新方法的新记录。此外，在单张 NVIDIA GeForce RTX 3090 显卡上，仅需 4.3 小时就可以对一个具有 1.85 小时实时交互经验的 agent 进行训练，相比以往的方法，其效率有了显著的提高。

**URL**: https://nips.cc/virtual/2023/poster/71385

---

## Large Language Models Are Semi-Parametric Reinforcement Learning Agents
**Author**: Danyang Zhang · Lu Chen · Situo Zhang · Hongshen Xu · Zihan Zhao · Kai Yu

**Abstract**: Inspired by the insights in cognitive science with respect to human memory and reasoning mechanism, a novel evolvable LLM-based (Large Language Model) agent framework is proposed as Rememberer. By equipping the LLM with a long-term experience memory, Rememberer is capable of exploiting the experiences from the past episodes even for different task goals, which excels an LLM-based agent with fixed exemplars or equipped with a transient working memory. We further introduce Reinforcement Learning with Experience Memory (RLEM) to update the memory. Thus, the whole system can learn from the experiences of both success and failure, and evolve its capability without fine-tuning the parameters of the LLM. In this way, the proposed Rememberer constitutes a semi-parametric RL agent. Extensive experiments are conducted on two RL task sets to evaluate the proposed framework. The average results with different initialization and training sets exceed the prior SOTA by 4% and 2% for the success rate on two task sets and demonstrate the superiority and robustness of Rememberer.

**Abstract(Chinese)**: 受认知科学对人类记忆和推理机制的启发，提出了一种新颖的可进化的基于LLM（大型语言模型）的代理框架，名为Rememberer。通过给LLM配备长期经验记忆，Rememberer能够利用过去的经历，即使是对不同的任务目标，这优于具有固定示例或配备瞬态工作记忆的基于LLM的代理。我们进一步介绍了具有经验记忆的强化学习（RLEM），以更新记忆。因此，整个系统可以从成功和失败的经验中学习，并且在不进行LLM参数微调的情况下提高其能力。因此，所提出的Rememberer构成了半参数化RL代理。对两个RL任务集进行了大量实验以评估所提出的框架。在不同初始化和训练集的平均结果超过先前任务上的SOTA 4%和2%，展示了Rememberer的优越性和鲁棒性。

**URL**: https://nips.cc/virtual/2023/poster/71228

---

## Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning
**Author**: Zikang Tian · Ruizhi Chen · Xing Hu · Ling Li · Rui Zhang · Fan Wu · Shaohui Peng · Jiaming Guo · Zidong Du · Qi Guo · Yunji Chen

**Abstract**: In recent years, Multi-Agent Reinforcement Learning (MARL) techniques have made significant strides in achieving high asymptotic performance in single task. However, there has been limited exploration of model transferability across tasks. Training a model from scratch for each task can be time-consuming and expensive, especially for large-scale Multi-Agent Systems. Therefore, it is crucial to develop methods for generalizing the model across tasks. Considering that there exist task-independent subtasks across MARL tasks, a model that can decompose such subtasks from the source task could generalize to target tasks. However, ensuring true task-independence of subtasks poses a challenge. In this paper, we propose to \textbf{d}ecompose a \textbf{t}ask in\textbf{to} a series of \textbf{g}eneralizable \textbf{s}ubtasks (DT2GS), a novel framework that addresses this challenge by utilizing a scalable subtask encoder and an adaptive subtask semantic module. We show that these components endow subtasks with two properties critical for task-independence: avoiding overfitting to the source task and maintaining consistent yet scalable semantics across tasks. Empirical results demonstrate that DT2GS possesses sound zero-shot generalization capability across tasks, exhibits sufficient transferability, and outperforms existing methods in both multi-task and single-task problems.

**Abstract(Chinese)**: 在近年来，多智体强化学习（MARL）技术在单一任务中取得了显著的渐近性能。然而，对跨任务的模型可转移性的探索有限。为每个任务从头开始训练模型可能耗时且昂贵，尤其是对于大规模多智体系统而言。因此，对于跨任务泛化模型的发展非常关键。考虑到在MARL任务中存在任务无关的子任务，一个能够从源任务中分解这种子任务并泛化到目标任务的模型可能会更具普适性。然而，确保子任务的真正任务无关性具有挑战性。在本文中，我们提出将任务分解为一系列可泛化子任务（DT2GS），这是一个新颖的框架，通过利用可扩展的子任务编码器和自适应子任务语义模块来解决这一挑战。我们展示了这些组件赋予子任务两个关键属性：避免过度拟合源任务以及在任务之间保持一致且可扩展的语义。实证结果表明，DT2GS具有跨任务零-shot泛化能力，展现了足够的可转移性，并且在多任务和单任务问题上优于现有方法。

**URL**: https://nips.cc/virtual/2023/poster/71183

---

## Loss Dynamics of Temporal Difference Reinforcement Learning
**Author**: Blake Bordelon · Paul Masset · Henry Kuo · Cengiz Pehlevan

**Abstract**: Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent dynamics. We study how learning dynamics and plateaus depend on feature structure, learning rate, discount factor, and reward function. We then analyze how strategies like learning rate annealing and reward shaping can favorably alter learning dynamics and plateaus. To conclude, our work introduces new tools to open a new direction towards developing a theory of learning dynamics in reinforcement learning.

**Abstract(Chinese)**: 强化学习在几个应用中取得了成功，其中代理需要学习在反馈稀疏的环境中行动。然而，尽管取得了实证成功，但对于强化学习模型的参数以及用于表示状态的特征如何相互作用以控制学习动态，仍然存在理论上的理解不足。在这项工作中，我们借鉴了统计物理学的概念，研究了具有线性函数逼近器的时间差分学习价值函数的典型情况学习曲线。我们的理论是基于高斯等价假设推导出来的，其中对于随机轨迹的平均值被替换为时间相关的高斯特征平均值，并且我们验证了我们的假设在小规模马尔可夫决策过程中的情况。我们发现由于对可能的情节空间进行子采样而导致的随机半梯度噪音会导致值误差中出现显著的高原，这与传统的梯度下降动态不同。我们研究了学习动态和高原如何取决于特征结构、学习率、折现因子和奖励函数。然后，我们分析了学习率退火和奖励塑造等策略如何有利地改变学习动态和高原。总之，我们的工作引入了新工具，开辟了强化学习中学习动态理论发展的新方向。

**URL**: https://nips.cc/virtual/2023/poster/71541

---

## Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets
**Author**: Zhang-Wei Hong · Aviral Kumar · Sathwik Karnik · Abhishek Bhandwaldar · Akash Srivastava · Joni Pajarinen · Romain Laroche · Abhishek Gupta · Pulkit Agrawal

**Abstract**: Offline reinforcement learning (RL) enables learning a decision-making policy without interaction with the environment. This makes it particularly beneficial in situations where such interactions are costly. However, a known challenge for offline RL algorithms is the distributional mismatch between the state-action distributions of the learned policy and the dataset, which can significantly impact performance. State-of-the-art algorithms address it by constraining the policy to align with the state-action pairs in the dataset. However, this strategy struggles on datasets that predominantly consist of trajectories collected by low-performing policies and only a few trajectories from high-performing ones. Indeed, the constraint to align with the data leads the policy to imitate low-performing behaviors predominating the dataset. Our key insight to address this issue is to constrain the policy to the policy that collected the good parts of the dataset rather than all data. To this end, we optimize the importance sampling weights to emulate sampling data from a data distribution generated by a nearly optimal policy. Our method exhibits considerable performance gains (up to five times better) over the existing approaches in state-of-the-art offline RL algorithms over 72 imbalanced datasets with varying types of imbalance.

**Abstract(Chinese)**: 离线强化学习（RL）使得学习决策策略成为可能，而无需与环境进行交互。这在那些交互成本较高的情况下尤为有益。然而，离线RL算法面临的一个已知挑战是学到的策略的状态-动作分布与数据集之间的分布不匹配，这可能会显著影响性能。最先进的算法通过将策略限制为与数据集中的状态-动作对齐来解决这个问题。然而，在主要由低性能策略收集的轨迹组成的数据集上，这种策略很难应用，并且只包含少量来自高性能策略的轨迹。实际上，与数据对齐的约束会使策略模仿主导数据集的低性能行为。我们解决这个问题的关键见解是将策略约束为采集了数据集中优质部分的策略，而非所有数据。为此，我们优化重要性抽样权重，以模拟从几乎最优策略生成的数据分布中抽样数据。我们的方法在72个不同类型不平衡的数据集上表现出了显著的性能提升（最高提高了五倍），超过了现有的最先进离线RL算法。

**URL**: https://nips.cc/virtual/2023/poster/71552

---

## Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage
**Author**: Jose Blanchet · Miao Lu · Tong Zhang · Han Zhong

**Abstract**: We study distributionally robust offline reinforcement learning (RL), which seeks to find an optimal robust policy purely from an offline dataset that can perform well in perturbed environments. We propose a generic algorithm framework Doubly Pessimistic Model-based Policy Optimization ($\texttt{P}^2\texttt{MPO}$) for robust offline RL, which features a novel combination of a flexible model estimation subroutine and a doubly pessimistic policy optimization step. Here the double pessimism principle is crucial to overcome the distribution shift incurred by i) the mismatch between behavior policy and the family of target policies; and ii) the perturbation of the nominal model. Under certain accuracy assumptions on the model estimation subroutine, we show that $\texttt{P}^2\texttt{MPO}$ is provably sample-efficient with robust partial coverage data, which means that the offline dataset has good coverage of the distributions induced by the optimal robust policy and perturbed models around the nominal model. By tailoring specific model estimation subroutines for concrete examples including tabular Robust Markov Decision Process (RMDP), factored RMDP, and RMDP with kernel and neural function approximations, we show that $\texttt{P}^2\texttt{MPO}$ enjoys a $\tilde{\mathcal{O}}(n^{-1/2})$ convergence rate, where $n$ is the number of trajectories in the offline dataset. Notably, these models, except for the tabular case, are first identified and proven tractable by this paper. To the best of our knowledge, we first propose a general learning principle --- double pessimism --- for robust offline RL and show that it is provably efficient in the context of general function approximations.

**Abstract(Chinese)**: 我们研究分布鲁棒的离线强化学习（RL），其旨在从离线数据集中纯粹找到一个能够在扰动环境中表现良好的最佳鲁棒策略。我们提出了一个通用的算法框架——双重悲观模型政策优化（$\texttt{P}^2\texttt{MPO}$），用于鲁棒离线RL，其特点是灵活的模型估计子例程和双重悲观政策优化步骤的新颖组合。这里的双重悲观原则对于克服由i）行为策略与目标策略系列之间的不匹配和ii）名义模型的扰动所引起的分布偏移至关重要。在模型估计子例程的某些准确性假设下，我们表明$\texttt{P}^2\texttt{MPO}$在具有鲁棒部分覆盖数据的情况下在样本效率上是可以证明的，这意味着离线数据集很好地覆盖了由最佳鲁棒策略和名义模型周围扰动模型所引起的分布。通过为具体示例定制特定的模型估计子例程，包括表格式鲁棒马尔可夫决策过程（RMDP）、分解RMDP以及具有核和神经函数逼近的RMDP，我们表明$\texttt{P}^2\texttt{MPO}$具有$\tilde{\mathcal{O}}(n^{-1/2})$的收敛速率，其中$n$是离线数据集中的轨迹数。值得注意的是，除了表格式情况外，这些模型在本文中是首次被识别并证明可处理的。据我们所知，我们首次提出了一个通用的学习原则——双重悲观——用于鲁棒的离线RL，并表明在一般函数逼近的背景下，其是可以证明高效的。

**URL**: https://nips.cc/virtual/2023/poster/71427

---

## Efficient Exploration in Continuous-time Model-based Reinforcement Learning
**Author**: Lenart Treven · Jonas Hübotter · Bhavya · Florian Dorfler · Andreas Krause

**Abstract**: Reinforcement learning algorithms typically consider discrete-time dynamics, even though the underlying systems are often continuous in time. In this paper, we introduce a model-based reinforcement learning algorithm that represents continuous-time dynamics using nonlinear ordinary differential equations (ODEs). We capture epistemic uncertainty using well-calibrated probabilistic models, and use the optimistic principle for exploration. Our regret bounds surface the importance of the measurement selection strategy (MSS), since in continuous time we not only must decide how to explore, but also when to observe the underlying system. Our analysis demonstrates that the regret is sublinear when modeling ODEs with Gaussian Processes (GP) for common choices of MSS, such as equidistant sampling. Additionally, we propose an adaptive, data-dependent, practical MSS that, when combined with GP dynamics, also achieves sublinear regret with significantly fewer samples. We showcase the benefits of continuous-time modeling over its discrete-time counterpart, as well as our proposed adaptive MSS over standard baselines, on several applications.

**Abstract(Chinese)**: 强化学习算法通常考虑离散时间动态，即使底层系统通常是连续时间的。在本文中，我们介绍了一种基于模型的强化学习算法，该算法利用非线性常微分方程（ODEs）表示连续时间动态。我们利用良好校准的概率模型捕获认知不确定性，并利用乐观原则进行探索。我们的后悔界突��了测量选择策略（MSS）的重要性，因为在连续时间中，我们不仅必须决定如何进行探索，还必须决定何时观察底层系统。我们的分析表明，在使用高斯过程（GP）对ODE进行建模时，对于常见的MSS选择，例如等距取样，后悔是次线性的。此外，我们提出了一种自适应的、数据依赖的实际MSS，当与GP动态结合时，也能以显著较少的样本实现次线性后悔。我们展示了连续时间建模相对于其离散时间对应物的优势，以及我们提出的自适应MSS相对于标准基准的优势在几个应用场景中。

**URL**: https://nips.cc/virtual/2023/poster/71440

---

## Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning
**Author**: Tyler Kastner · Murat Erdogdu · Amir-massoud Farahmand

**Abstract**: We consider the problem of learning models for risk-sensitive reinforcement learning. We theoretically demonstrate that proper value equivalence, a method of learning models which can be used to plan optimally in the risk-neutral setting, is not sufficient to plan optimally in the risk-sensitive setting. We leverage distributional reinforcement learning to introduce two new notions of model equivalence, one which is general and can be used to plan for any risk measure, but is intractable; and a practical variation which allows one to choose which risk measures they may plan optimally for. We demonstrate how our models can be used to augment any model-free risk-sensitive algorithm, and provide both tabular and large-scale experiments to demonstrate our method’s ability.

**Abstract(Chinese)**: 我们考虑学习用于风险敏感强化学习的模型的问题。我们在理论上证明，适当的价值等价性，一种学习模型的方法，可以用于在风险中性设置下进行最优规划，但不足以在风险敏感设置下进行最优规划。我们利用分布式强化学习引入了两种模型等价性的新概念，一种是通用的，并可用于为任何风险测度进行最优规划，但是难以处理；还有一种实用的变体，允许选择为哪些风险测度进行最优规划。我们展示了我们的模型如何用于增强任何无模型风险敏感算法，并提供了表格和大规模实验来展示我们方法的能力。

**URL**: https://nips.cc/virtual/2023/poster/71612

---

## Swarm Reinforcement Learning for Adaptive Mesh Refinement
**Author**: Niklas Freymuth · Philipp Dahlinger · Tobias Würth · Simon Reisch · Luise Kärger · Gerhard Neumann

**Abstract**: The Finite Element Method, an important technique in engineering, is aided by Adaptive Mesh Refinement (AMR), which dynamically refines mesh regions to allow for a favorable trade-off between computational speed and simulation accuracy. Classical methods for AMR depend on task-specific heuristics or expensive error estimators, hindering their use for complex simulations. Recent learned AMR methods tackle these problems, but so far scale only to simple toy examples. We formulate AMR as a novel Adaptive Swarm Markov Decision Process in which a mesh is modeled as a system of simple collaborating agents that may split into multiple new agents. This framework allows for a spatial reward formulation that simplifies the credit assignment problem, which we combine with Message Passing Networks to propagate information between neighboring mesh elements. We experimentally validate the effectiveness of our approach, Adaptive Swarm Mesh Refinement (ASMR), showing that it learns reliable, scalable, and efficient refinement strategies on a set of challenging problems. Our approach significantly speeds up computation, achieving up to 30-fold improvement compared to uniform refinements in complex simulations. Additionally, we outperform learned baselines and achieve a refinement quality that is on par with a traditional error-based AMR strategy without expensive oracle information about the error signal.

**Abstract(Chinese)**: 有限元方法是工程中的一项重要技术，自适应网格细化（AMR）辅助了这一方法，动态地细化网格区域，以实现计算速度和模拟精度之间的有利权衡。传统的AMR方法依赖于特定任务的启发式方法或昂贵的误差估计器，这阻碍了它们在复杂模拟中的使用。最近的学习型AMR方法解决了这些问题，但目前仅能扩展到简单的示例。我们将AMR构建为一种新颖的自适应群体马尔可夫决策过程，在这种过程中，网格被建模为一组简单协作的代理，它们可能分裂成多个新代理。该框架允许进行空间奖励制定，简化了信用分配问题，我们将其与消息传递网络相结合，以在相邻网格单元之间传播信息。我们通过实验证实了我们方法的有效性，自适应群体网格细化（ASMR）显示出它学习了一套在一系列具有挑战性问题上可靠、可扩展和高效的细化策略。我们的方法显著加速了计算，与复杂模拟中的均匀细化相比，实现了长达30倍的改进。此外，我们胜过了学习基线，并且达到了与传统基于误差的AMR策略一样的细化质量，而不需要昂贵的有关误差信号的知识。

**URL**: https://nips.cc/virtual/2023/poster/70308

---

## Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective
**Author**: Zeyu Zhang · Yi Su · Hui Yuan · Yiran Wu · Rishab Balasubramanian · Qingyun Wu · Huazheng Wang · Mengdi Wang

**Abstract**: Off-policy Learning to Rank (LTR) aims to optimize a ranker from data collected by a deployed logging policy. However, existing off-policy learning to rank methods often make strong assumptions about how users generate the click data, i.e., the click model, and hence need to tailor their methods specifically under different click models. In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking could be learned with offline reinforcement learning (RL) directly. Building upon this, we leverage offline RL techniques for off-policy LTR and propose the Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR) method, which could be easily applied to a wide range of click models. Through a dedicated formulation of the MDP, we show that offline RL algorithms can adapt to various click models without complex debiasing techniques and prior knowledge of the model. Results on various large-scale datasets demonstrate that CUOLR consistently outperforms the state-of-the-art off-policy learning to rank algorithms while maintaining consistency and robustness under different click models.

**Abstract(Chinese)**: 离线学习排序（LTR）旨在通过已部署的记录策略收集的数据优化排名器。然而，现有的离线学习排序方法往往对用户生成点击数据的方式（即点击模型）做出了很强的假设，因此需要根据不同的点击模型专门调整它们的方法。在本文中，我们将通用随机点击模型下的排名过程统一视为马尔可夫决策过程（MDP），并可通过离线强化学习（RL）直接学习最优排名。在此基础上，我们利用离线RL技术进行离线LTR，并提出了点击模型不可知的统一离线学习排序方法（CUOLR），可轻松应用于各种点击模型。通过对MDP的专门制定，我们表明离线RL算法可以适应各种点击模型，无需复杂的去偏差技术和先验知识的模型。在各种大规模数据集上的结果表明，CUOLR在保持一致性和稳健性的同时，始终优于最先进的离线学习排序算法。

**URL**: https://nips.cc/virtual/2023/poster/70469

---

## Prioritizing Samples in Reinforcement Learning with Reducible Loss
**Author**: Shivakanth Sujit · Somjit Nath · Pedro Braga · Samira Ebrahimi Kahou

**Abstract**: Most reinforcement learning algorithms take advantage of an experience replay buffer to repeatedly train on samples the agent has observed in the past. Not all samples carry the same amount of significance and simply assigning equal importance to each of the samples is a naïve strategy. In this paper, we propose a method to prioritize samples based on how much we can learn from a sample. We define the learn-ability of a sample as the steady decrease of the training loss associated with this sample over time. We develop an algorithm to prioritize samples with high learn-ability, while assigning lower priority to those that are hard-to-learn, typically caused by noise or stochasticity. We empirically show that across multiple domains our method is more robust than random sampling and also better than just prioritizing with respect to the training loss, i.e. the temporal difference loss, which is used in prioritized experience replay.

**Abstract(Chinese)**: 大多数强化学习算法利用经验重演缓冲区，反复对代理程序过去观察到的样本进行训练。并非所有样本都具有相同的重要性，简单地赋予每个样本相等的重要性是一种天真的策略。在本文中，我们提出一种基于样本的学习价值来优先考虑样本的方法。我们将样本的学习价值定义为随着时间的推移与该样本相关的训练损失的稳定下降。我们开发了一种算法，以优先考虑具有较高学习能力的样本，同时给那些难以学习的样本赋予较低的优先级，通常是由噪声或随机性引起的。我们凭经验证明，在多个领域中，我们的方法比随机抽样更加稳健，也优于只优先考虑与训练损失相关的方法，即优先经验重演中使用的时间差损失。

**URL**: https://nips.cc/virtual/2023/poster/70869

---

## Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design
**Author**: Matthew T Jackson · Minqi Jiang · Jack Parker-Holder · Risto Vuorio · Chris Lu · Greg Farquhar · Shimon Whiteson · Jakob Foerster

**Abstract**: The past decade has seen vast progress in deep reinforcement learning (RL) on the back of algorithms manually designed by human researchers. Recently, it has been shown that it is possible to meta-learn update rules, with the hope of discovering algorithms that can perform well on a wide range of RL tasks. Despite impressive initial results from algorithms such as Learned Policy Gradient (LPG), there remains a generalization gap when these algorithms are applied to unseen environments. In this work, we examine how characteristics of the meta-training distribution impact the generalization performance of these algorithms. Motivated by this analysis and building on ideas from Unsupervised Environment Design (UED), we propose a novel approach for automatically generating curricula to maximize the regret of a meta-learned optimizer, in addition to a novel approximation of regret, which we name algorithmic regret (AR). The result is our method, General RL Optimizers Obtained Via Environment Design (GROOVE). In a series of experiments, we show that GROOVE achieves superior generalization to LPG, and evaluate AR against baseline metrics from UED, identifying it as a critical component of environment design in this setting. We believe this approach is a step towards the discovery of truly general RL algorithms, capable of solving a wide range of real-world environments.

**Abstract(Chinese)**: 过去十年来，深度强化学习取得了巨大进展，这要归功于人类研究人员手动设计的算法。最近，有人表明可以元学习更新规则，希望发现能够在各种强化学习任务中表现良好的算法。尽管像学得策略梯度（LPG）这样的算法初期取得了令人印象深刻的成果，但是当这些算法应用于未知环境时，仍然存在泛化差距。在这项工作中，我们研究了元训练分布的特征如何影响这些算法的泛化性能。受此分析的启发，借鉴了无监督环境设计（UED）的思想，我们提出了一种新方法，用于自动生成课程表，以最大化元学得优化器的遗憾，另外还提出了遗憾的新近似方法，我们称之为算法遗憾（AR）。其结果就是我们的方法，通过环境设计获得的通用强化学习优化器（GROOVE）。通过一系列实验，我们展示了GROOVE相较于LPG取得了更好的泛化性能，并且评估了AR与UED的基准指标相比，确定了它在此环境设计中的重要组成部分。我们相信这种方法是朝着真正通用的强化学习算法的发现迈出了一步，能够解决各种真实世界环境。

**URL**: https://nips.cc/virtual/2023/poster/70658

---

## H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation
**Author**: Yanjie Ze · Yanjie Ze · Yuyao Liu · Ruizhe Shi · Jiaxin Qin · Zhecheng Yuan · Jiashun Wang · Huazhe Xu

**Abstract**: Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human $\textbf{H}$and-$\textbf{In}$formed visual representation learning framework to solve difficult $\textbf{Dex}$terous manipulation tasks ($\textbf{H-InDex}$) with reinforcement learning. Our framework consists of three stages: $\textit{(i)}$ pre-training representations with 3D human hand pose estimation, $\textit{(ii)}$ offline adapting representations with self-supervised keypoint detection, and $\textit{(iii)}$ reinforcement learning with exponential moving average BatchNorm. The last two stages only modify $0.36$% parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study $\textbf{12}$ challenging dexterous manipulation tasks and find that $\textbf{H-InDex}$ largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code and videos are available at https://yanjieze.com/H-InDex .

**Abstract(Chinese)**: 摘要：

人类手部具有非凡的灵巧性，并长期作为机器人操作的灵感来源。在这项工作中，我们提出了一种人手信息驱动的视觉表示学习框架，用于通过强化学习解决困难的巧妙操作任务（H-InDex）。我们的框架包括三个阶段：（i）使用3D人手姿势估计进行预训练表示，（ii）使用自监督关键点检测对表示进行离线适应，以及（iii）使用指数移动平均BatchNorm进行强化学习。最后两个阶段仅总共修改了预先训练表示的0.36%的参数，确保从预训练中获得的知识得到充分保留。我们在实验中研究了12个具有挑战性的巧妙操作任务，并发现H-InDex在很大程度上超越了强基线方法和最近的用于运动控制的视觉基础模型。代码和视频可在https://yanjieze.com/H-InDex找到。

**URL**: https://nips.cc/virtual/2023/poster/70570

---

## Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning
**Author**: Haoran He · Chenjia Bai · Kang Xu · Zhuoran Yang · Weinan Zhang · Dong Wang · Bin Zhao · Xuelong Li

**Abstract**: Diffusion models have demonstrated highly-expressive generative capabilities in vision and NLP. Recent studies in reinforcement learning (RL) have shown that diffusion models are also powerful in modeling complex policies or trajectories in offline datasets. However, these works have been limited to single-task settings where a generalist agent capable of addressing multi-task predicaments is absent. In this paper, we aim to investigate the effectiveness of a single diffusion model in modeling large-scale multi-task offline data, which can be challenging due to diverse and multimodal data distribution. Specifically, we propose Multi-Task Diffusion Model (\textsc{MTDiff}), a diffusion-based method that incorporates Transformer backbones and prompt learning for generative planning and data synthesis in multi-task offline settings. \textsc{MTDiff} leverages vast amounts of knowledge available in multi-task data and performs implicit knowledge sharing among tasks. For generative planning, we find \textsc{MTDiff} outperforms state-of-the-art algorithms across 50 tasks on Meta-World and 8 maps on Maze2D. For data synthesis, \textsc{MTDiff} generates high-quality data for testing tasks given a single demonstration as a prompt, which enhances the low-quality datasets for even unseen tasks.

**Abstract(Chinese)**: 扩散模型在视觉和自然语言处理领域已经展示出了高度表现力的生成能力。最近在强化学习领域的研究表明，扩散模型在对离线数据集中的复杂策略或轨迹建模方面也十分强大。然而，这些工作都局限在单任务设置下，缺乏能够解决多任务困境的通用型代理。本文旨在研究单一扩散模型在建模大规模多任务离线数据方面的有效性，该过程可能具有挑战性，因为数据分布多样且多模态。具体而言，我们提出了基于Transformer骨干和提示学习的Multi-Task Diffusion Model（MTDiff）方法，用于在多任务离线设置中进行生成规划和数据合成。MTDiff利用多任务数据中的大量知识，并在任务之间进行隐式知识共享。在生成规划方面，我们发现MTDiff在Meta-World的50个任务和Maze2D的8个地图上优于现有算法。在数据合成方面，MTDiff可以针对单次演示作为提示生成高质量数据，从而为甚至未知任务增强了低质量数据集。

**URL**: https://nips.cc/virtual/2023/poster/70916

---

## Doubly Robust Augmented Transfer for Meta-Reinforcement Learning
**Author**: Yuankun Jiang · Nuowen Kan · Chenglin Li · Wenrui Dai · Junni Zou · Hongkai Xiong

**Abstract**: Meta-reinforcement learning (Meta-RL), though enabling a fast adaptation to learn new skills by exploiting the common structure shared among different tasks, suffers performance degradation in the sparse-reward setting. Current hindsight-based sample transfer approaches can alleviate this issue by transferring relabeled trajectories from other tasks to a new task so as to provide informative experience for the target reward function, but are unfortunately constrained with the unrealistic assumption that tasks differ only in reward functions. In this paper, we propose a doubly robust augmented transfer (DRaT) approach, aiming at addressing the more general sparse reward meta-RL scenario with both dynamics mismatches and varying reward functions across tasks. Specifically, we design a doubly robust augmented estimator for efficient value-function evaluation, which tackles dynamics mismatches with the optimal importance weight of transition distributions achieved by minimizing the theoretically derived upper bound of mean squared error (MSE) between the estimated values of transferred samples and their true values in the target task. Due to its intractability, we then propose an interval-based approximation to this optimal importance weight, which is guaranteed to cover the optimum with a constrained and sample-independent upper bound on the MSE approximation error. Based on our theoretical findings, we finally develop a DRaT algorithm for transferring informative samples across tasks during the training of meta-RL. We implement DRaT on an off-policy meta-RL baseline, and empirically show that it significantly outperforms other hindsight-based approaches on various sparse-reward MuJoCo locomotion tasks with varying dynamics and reward functions.

**Abstract(Chinese)**: 元元强化学习（元RL）虽然通过利用在不同任务中共享的通用结构快速适应学习新技能，但在稀疏奖励设置中存在性能下降。当前基于事后转移的样本转移方法可以通过从其他任务转移重新标记的轨迹到新任务，从而为目标奖励函数提供信息性经验，但遗憾的是受限于任务仅在奖励函数上不同的不切实际的假设。在本文中，我们提出了一种双重稳健增强转移（DRaT）方法，旨在解决更一般的稀疏奖励元RL场景，其中动态不匹配和任务间的奖励函数变化。具体而言，我们设计了一种双重稳健增强估计器，用于高效价值函数评估，通过最小化理论推导的均方误差（MSE）的上界，处理转移样本的估计值与目标任务中真实值之间的差异。由于其不可行性，我们提出了对这种最佳重要性权重的基于区间的近似，它保证了在MSE近似误差上有一定的约束和独立于样本的上界。基于我们的理论发现，最后我们开发了一个DRaT算法，用于在元RL训练期间在不同任务之间转移信息性样本。我们在基于政策的元RL基线上实施了DRaT，并经验证明它在具有不同动态和奖励函数的各种稀疏奖励MuJoCo运动任务中明显优于其他基于事后的方法。

**URL**: https://nips.cc/virtual/2023/poster/70281

---

## Team-PSRO for Learning Approximate TMECor in Large Team Games via Cooperative Reinforcement Learning
**Author**: Stephen McAleer · Gabriele Farina · Gaoyue Zhou · Mingzhi Wang · Yaodong Yang · Tuomas Sandholm

**Abstract**: Recent algorithms have achieved superhuman performance at a number of two-player zero-sum games such as poker and go. However, many real-world situations are multi-player games. Zero-sum two-team games, such as bridge and football, involve two teams where each member of the team shares the same reward with every other member of that team, and each team has the negative of the reward of the other team. A popular solution concept in this setting, called TMECor, assumes that teams can jointly correlate their strategies before play, but are not able to communicate during play. This setting is harder than two-player zero-sum games because each player on a team has different information and must use their public actions to signal to other members of the team. Prior works either have game-theoretic guarantees but only work in very small games, or are able to scale to large games but do not have game-theoretic guarantees. In this paper we introduce two algorithms: Team-PSRO, an extension of PSRO from two-player games to team games, and Team-PSRO Mix-and-Match which improves upon Team PSRO by better using population policies. In Team-PSRO, in every iteration both teams learn a joint best response to the opponent's meta-strategy via reinforcement learning. As the reinforcement learning joint best response approaches the optimal best response, Team-PSRO is guaranteed to converge to a TMECor. In experiments on Kuhn poker and Liar's Dice, we show that a tabular version of Team-PSRO converges to TMECor, and a version of Team PSRO using deep cooperative reinforcement learning beats self-play reinforcement learning in the large game of Google Research Football.

**Abstract(Chinese)**: 最近的算法在一些两人零和博弈（如扑克和围棋）中实现了超越人类的表现。然而，许多现实世界的情境是多人游戏。零和两队游戏（如桥牌和足球）涉及两个团队，其中每个团队成员与该团队的每个其他成员共享相同的奖励，并且每个团队获得另一个团队的相反奖励。在这种背景下，一个流行的解决方案概念称为TMECor，假设团队可以在比赛前共同相关其策略，但在比赛期间无法进行沟通。这种设置比两人零和游戏更难，因为团队中的每个玩家都有不同的信息，并且必须利用其公开行动向团队的其他成员发出信号。先前的研究要么具有博弈论保证，但仅适用于非常小的游戏，要么能够扩展到大型游戏，但不具有博弈论保证。在本文中，我们介绍了两种算法：Team-PSRO是从两人博弈扩展为团队游戏的PSRO的延伸，以及Team-PSRO Mix-and-Match，它通过更好地使用人口政策来改进Team-PSRO。在Team-PSRO中，每次迭代中，两个团队都学习对手的元策略的联合最佳响应，通过强化学习。随着强化学习联合最佳响应的逼近最优最佳响应，Team-PSRO保证收敛到TMECor。在Kuhn扑克和Liar's Dice的实验中，我们展示了Team-PSRO的标签版本收敛到了TMECor，并且Team PSRO使用深度合作强化学习打败了大型游戏Google Research Football中的自我对抗强化学习。

**URL**: https://nips.cc/virtual/2023/poster/70606

---

## Distributional Pareto-Optimal Multi-Objective Reinforcement Learning
**Author**: Xin-Qiang Cai · Pushi Zhang · Li Zhao · Jiang Bian · Masashi Sugiyama · Ashley Llorens

**Abstract**: Multi-objective reinforcement learning (MORL) has been proposed to learn control policies over multiple competing objectives with each possible preference over returns. However, current MORL algorithms fail to account for distributional preferences over the multi-variate returns, which are particularly important in real-world scenarios such as autonomous driving. To address this issue, we extend the concept of Pareto-optimality in MORL into distributional Pareto-optimality, which captures the optimality of return distributions, rather than the expectations. Our proposed method, called Distributional Pareto-Optimal Multi-Objective Reinforcement Learning~(DPMORL), is capable of learning distributional Pareto-optimal policies that balance multiple objectives while considering the return uncertainty. We evaluated our method on several benchmark problems and demonstrated its effectiveness in discovering distributional Pareto-optimal policies and satisfying diverse distributional preferences compared to existing MORL methods.

**Abstract(Chinese)**: 多目标强化学习（MORL）被提出来学习控制政策，以处理多个竞争性目标，每个目标对回报都有可能的偏好。然而，目前的MORL算法未能考虑多变量回报的分布偏好，这在现实场景中尤其重要，比如自动驾驶。为了解决这个问题，我们将MORL中帕累托优化的概念扩展为分布帕累托优化，这捕捉了回报分布的最优性，而不是期望值。我们提出的方法，称为Distributional Pareto-Optimal Multi-Objective Reinforcement Learning（DPMORL），能够学习分布帕累托最优政策，同时考虑回报的不确定性。我们在几个基准问题上评估了我们的方法，并展示了与现有MORL方法相比，其在发现分布帕累托最优政策和满足多样的分布偏好方面的有效性。

**URL**: https://nips.cc/virtual/2023/poster/70393

---

## Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation
**Author**: Wenhao Ding · Laixi Shi · Yuejie Chi · DING ZHAO

**Abstract**: Robustness has been extensively studied in reinforcement learning (RL) to handle various forms of uncertainty such as random perturbations, rare events, and malicious attacks. In this work, we consider one critical type of robustness against spurious correlation, where different portions of the state do not have correlations induced by unobserved confounders. These spurious correlations are ubiquitous in real-world tasks, for instance, a self-driving car usually observes heavy traffic in the daytime and light traffic at night due to unobservable human activity. A model that learns such useless or even harmful correlation could catastrophically fail when the confounder in the test case deviates from the training one. Although motivated, enabling robustness against spurious correlation poses significant challenges since the uncertainty set, shaped by the unobserved confounder and causal structure, is difficult to characterize and identify. Existing robust algorithms that assume simple and unstructured uncertainty sets are therefore inadequate to address this challenge. To solve this issue, we propose Robust State-Confounded Markov Decision Processes (RSC-MDPs) and theoretically demonstrate its superiority in avoiding learning spurious correlations compared with other robust RL counterparts. We also design an empirical algorithm to learn the robust optimal policy for RSC-MDPs, which outperforms all baselines in eight realistic self-driving and manipulation tasks.

**Abstract(Chinese)**: 在强化学习（RL）中，鲁棒性已被广泛研究，以处理各种形式的不确定性，例如随机扰动、罕见事件和恶意攻击。在这项工作中，我们考虑一种针对虚假相关性的关键鲁棒性类型，即状态的不同部分不具有由未观察到的混淆因素引起的相关性。这些虚假相关性在现实世界的任务中是普遍存在的，例如，自动驾驶汽车通常会在白天观察到交通拥堵，而在夜晚会观察到交通疏通，这是由于无法观察到的人类活动。学习这种无用甚至有害相关性的模型在测试案例中混淆因素偏离训练案例时可能会灾难性地失败。尽管存在动机，使模型对虚假相关性具有鲁棒性会带来重大挑战，因为由未观察到的混淆因素和因果结构塑造的不确定性集很难进行表征和识别。因此，假设简单和无结构的不确定性集的现有鲁棒算法无法解决这一挑战。为了解决这个问题，我们提出了Robust State-Confounded Markov Decision Processes（RSC-MDPs），并在理论上证明了它在避免学习虚假相关性方面的优越性，与其他鲁棒强化学习对手相比。我们还设计了一个实证算法来学习RSC-MDPs的鲁棒最优策略，在八个真实的自动驾驶和操纵任务中胜过所有基线。

**URL**: https://nips.cc/virtual/2023/poster/71143

---

## One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning
**Author**: Marc Rigter · Marc Rigter · Bruno Lacerda · Nick Hawes

**Abstract**: Offline reinforcement learning (RL) is suitable for safety-critical domains where online exploration is not feasible. In such domains, decision-making should take into consideration the risk of catastrophic outcomes. In other words, decision-making should be risk-averse. An additional challenge of offline RL is avoiding distributional shift, i.e. ensuring that  state-action pairs visited by the policy remain near those in the dataset. Previous offline RL algorithms that consider risk combine offline RL techniques (to avoid distributional shift), with risk-sensitive RL algorithms (to achieve risk-aversion). In this work, we propose risk-aversion as a mechanism to jointly address both of these issues. We propose a model-based approach, and use an ensemble of models to estimate epistemic uncertainty, in addition to aleatoric uncertainty. We train a policy that is risk-averse, and avoids high uncertainty actions. Risk-aversion to epistemic uncertainty prevents distributional shift, as areas not covered by the dataset have high epistemic uncertainty. Risk-aversion to aleatoric uncertainty discourages actions that are risky due to environment stochasticity. Thus, by considering epistemic uncertainty via a model ensemble and introducing risk-aversion, our algorithm (1R2R) avoids distributional shift in addition to achieving risk-aversion to aleatoric risk. Our experiments show that 1R2R achieves strong performance on deterministic benchmarks, and outperforms existing approaches for risk-sensitive objectives in stochastic domains.

**Abstract(Chinese)**: 离线强化学习（RL）适用于在线探索不可行的安全关键领域。在这些领域中，决策应考虑灾难性结果的风险。换句话说，决策应该具有风险规避性。离线RL的另一个挑战是避免分布偏移，即确保策略访问的状态-动作对保持接近数据集中的状态-动作对。以前考虑风险的离线RL算法将离线RL技术（以避免分布偏移）与风险敏感RL算法（以实现风险规避）相结合。在这项工作中，我们提出风险规避作为共同解决这些问题的机制。我们提出一种基于模型的方法，并使用模型集合来估计认知不确定性，以及随机不确定性。我们训练一个具有风险规避性的策略，并避免高不确定性动作。对认知不确定性的风险规避防止分布偏移，因为数据集未覆盖的区域具有较高的认知不确定性。对随机不确定性的风险规避会阻止由于环境的随机性而具有风险的动作。因此，通过考虑模型集合中的认知不确定性并引入风险规避，我们的算法（1R2R）避免了分布偏移，同时实现了对随机风险的风险规避。我们的实验表明1R2R在确定性基准测试中表现出色，并且在随机领域中超越了现有的风险敏感目标方法。

**URL**: https://nips.cc/virtual/2023/poster/70351

---

## VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning
**Author**: Jiayi Guan · Guang Chen · Jiaming Ji · Long Yang · ao zhou · Zhijun Li · changjun jiang

**Abstract**: Offline safe reinforcement learning (RL) algorithms promise to learn policies that satisfy safety constraints directly in offline datasets without interacting with the environment. This arrangement is particularly important in scenarios with high sampling costs and potential dangers, such as autonomous driving and robotics. However, the influence of safety constraints and out-of-distribution (OOD) actions have made it challenging for previous methods to achieve high reward returns while ensuring safety. In this work, we propose a Variational Optimization with Conservative Eestimation algorithm (VOCE) to solve the problem of optimizing safety policies in the offline dataset. Concretely, we reframe the problem of offline safe RL using probabilistic inference, which introduces variational distributions to make the optimization of policies more flexible. Subsequently, we utilize pessimistic estimation methods to estimate the Q-value of cost and reward, which mitigates the extrapolation errors induced by OOD actions. Finally, extensive experiments demonstrate that the VOCE algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety.

**Abstract(Chinese)**: 离线安全强化学习（RL）算法承诺直接在离线数据集中学习满足安全约束的策略，而无需与环境进行交互。这种安排在高采样成本和潜在危险的情况下尤为重要，比如自动驾驶和机器人技术。然而，安全约束和超出分布（OOD）行为的影响使得先前的方法在确保安全的同时很难实现高回报。在这项工作中，我们提出了一种具有保守估计的变分优化算法（VOCE），以解决优化离线数据集中安全策略的问题。具体地说，我们重新构建了使用概率推断的离线安全RL问题，引入变分分布以使策略的优化更加灵活。随后，我们利用悲观估计方法来估计成本和回报的Q值，从而减轻OOD行为引起的外推错误。最后，大量实验证明VOCE算法在多个实验任务中取得了竞争性能，特别是在安全性方面优于最先进的算法。

**URL**: https://nips.cc/virtual/2023/poster/70278

---

## Supported Value Regularization for Offline Reinforcement Learning
**Author**: Yixiu Mao · Hongchang Zhang · Chen Chen · Yi Xu · Xiangyang Ji

**Abstract**: Offline reinforcement learning suffers from the extrapolation error and value overestimation caused by out-of-distribution (OOD) actions. To mitigate this issue, value regularization approaches aim to penalize the learned value functions to assign lower values to OOD actions. However, existing value regularization methods lack a proper distinction between the regularization effects on in-distribution (ID) and OOD actions, and fail to guarantee optimal convergence results of the policy. To this end, we propose Supported Value Regularization (SVR), which penalizes the Q-values for all OOD actions while maintaining standard Bellman updates for ID ones. Specifically, we utilize the bias of importance sampling to compute the summation of Q-values over the entire OOD region, which serves as the penalty for policy evaluation. This design automatically separates the regularization for ID and OOD actions without manually distinguishing between them. In tabular MDP, we show that the policy evaluation operator of SVR is a contraction, whose fixed point outputs unbiased Q-values for ID actions and underestimated Q-values for OOD actions. Furthermore, the policy iteration with SVR guarantees strict policy improvement until convergence to the optimal support-constrained policy in the dataset. Empirically, we validate the theoretical properties of SVR in a tabular maze environment and demonstrate its state-of-the-art performance on a range of continuous control tasks in the D4RL benchmark.

**Abstract(Chinese)**: 离线强化学习受到由于超出分布（OOD）行为引起的外推错误和价值过高估计的困扰。为了缓解这一问题，价值正则化方法旨在惩罚学习价值函数，从而为OOD行为分配更低的价值。然而，现有的价值正则化方法在区分对分布（ID）行为和OOD行为的正则化效果方面缺乏明确，并且无法保证策略的最优收敛结果。为此，我们提出了支持价值正则化（SVR），该方法惩罚所有OOD行为的Q值，同时为ID行为保持标准的Bellman更新。具体而言，我们利用重要性抽样的偏差来计算整个OOD区域上的Q值总和，这作为策略评估的惩罚。这种设计自动地将ID和OOD行为的正则化分开，而无需手动区分它们。在表格MDP中，我们展示了SVR的策略评估算子是一个收缩运算，其不变点输出了ID行为的无偏Q值和OOD行为的低估Q值。此外，使用SVR进行策略迭代可保证严格的政策改进，直至收敛到数据集中的最优支持约束策略。从经验上看，我们在表格迷宫环境中验证了SVR的理论特性，并在D4RL基准测试中展示了其在一系列连续控制任务中的最新性能。

**URL**: https://nips.cc/virtual/2023/poster/70875

---

## Reining Generalization in Offline Reinforcement Learning via Representation Distinction
**Author**: Yi Ma · Hongyao Tang · Dong Li · Zhaopeng Meng

**Abstract**: Offline Reinforcement Learning (RL) aims to address the challenge of distribution shift between the dataset and the learned policy, where the value of out-of-distribution (OOD) data may be erroneously estimated due to overgeneralization. It has been observed that a considerable portion of the benefits derived from the conservative terms designed by existing offline RL approaches originates from their impact on the learned representation. This observation prompts us to scrutinize the learning dynamics of offline RL, formalize the process of generalization, and delve into the prevalent overgeneralization issue in offline RL. We then investigate the potential to rein the generalization from the representation perspective to enhance offline RL. Finally, we present  Representation Distinction (RD), an innovative plug-in method for improving offline RL algorithm performance by explicitly differentiating between the representations of in-sample and OOD state-action pairs generated by the learning policy. Considering scenarios in which the learning policy mirrors the behavioral policy and similar samples may be erroneously distinguished, we suggest a dynamic adjustment mechanism for RD based on an OOD data generator to prevent data representation collapse and further enhance policy performance. We demonstrate the efficacy of our approach by applying RD to specially-designed backbone algorithms and widely-used offline RL algorithms. The proposed RD method significantly improves their performance across various continuous control tasks on D4RL datasets, surpassing several state-of-the-art offline RL algorithms.

**Abstract(Chinese)**: 离线强化学习（RL）旨在解决数据集和学习策略之间的分布转移挑战，其中由于过度泛化，可能会错误估计超出分布（OOD）数据的价值。观察到现有离线RL方法设计的保守项中的相当一部分好处来自它们对学习表示的影响，这一观察促使我们审视离线RL的学习动态，形式化泛化过程，并深入研究离线RL中普遍存在的过度泛化问题。然后，我们研究从表示角度约束泛化以增强离线RL的潜力。最后，我们提出了“表示区分”（RD），这是一种创新的插件方法，通过明确区分学习策略生成的样本中的样本内和OOD状态-动作对的表示，来改进离线RL算法的性能。考虑到学习策略镜像行为策略且类似样本可能被错误区分的情况，我们建议基于OOD数据生成器的RD动态调整机制，以防止数据表示崩溃并进一步提高策略性能。通过将RD应用到特别设计的骨干算法和广泛使用的离线RL算法，我们展示了我们方法的有效性。所提出的RD方法显着改善了它们在D4RL数据集上各种连续控制任务上的性能，超越了几种最先进的离线RL算法。

**URL**: https://nips.cc/virtual/2023/poster/70549

---

## Revisiting the Minimalist Approach to Offline Reinforcement Learning
**Author**: Denis Tarasov · Vladislav Kurenkov · Alexander Nikulin · Sergey Kolesnikov

**Abstract**: Recent years have witnessed significant advancements in offline reinforcement learning (RL), resulting in the development of numerous algorithms with varying degrees of complexity. While these algorithms have led to noteworthy improvements, many incorporate seemingly minor design choices that impact their effectiveness beyond core algorithmic advances. However, the effect of these design choices on established baselines remains understudied. In this work, we aim to bridge this gap by conducting a retrospective analysis of recent works in offline RL and propose ReBRAC, a minimalistic algorithm that integrates such design elements built on top of the TD3+BC method. We evaluate ReBRAC on 51 datasets with both proprioceptive and visual state spaces using D4RL and V-D4RL benchmarks, demonstrating its state-of-the-art performance among ensemble-free methods in both offline and offline-to-online settings. To further illustrate the efficacy of these design choices, we perform a large-scale ablation study and hyperparameter sensitivity analysis on the scale of thousands of experiments.

**Abstract(Chinese)**: 近年来，离线强化学习（RL）取得了重大进展，导致了许多算法的发展，复杂程度不同。虽然这些算法带来了显著的改进，但许多都融入了看似微小的设计选择，影响了它们的有效性超越核心算法的进步。然而，这些设计选择对已建立的基线的影响仍未得到研究。在这项工作中，我们旨在通过对最近的离线RL研究进行回顾性分析，并提出了ReBRAC，这是一种集成了这些设计元素的极简算法，是建立在TD3+BC方法之上的。我们使用D4RL和V-D4RL基准测试在51个数据集上评估了ReBRAC，展示了它在离线和离线到在线设置中在整套方法中的最先进性能。为了进一步阐明这些设计选择的效果，我们进行了大规模的消融研究和超参数敏感性分析，涉及数千个实验。

**URL**: https://nips.cc/virtual/2023/poster/70088

---

## Flexible Attention-Based Multi-Policy Fusion for Efficient Deep Reinforcement Learning
**Author**: Zih-Yun Chiu · Yi-Lin Tuan · William Yang Wang · Michael Yip

**Abstract**: Reinforcement learning (RL) agents have long sought to approach the efficiency of human learning. Humans are great observers who can learn by aggregating external knowledge from various sources, including observations from others' policies of attempting a task. Prior studies in RL have incorporated external knowledge policies to help agents improve sample efficiency. However, it remains non-trivial to perform arbitrary combinations and replacements of those policies, an essential feature for generalization and transferability. In this work, we present Knowledge-Grounded RL (KGRL), an RL paradigm fusing multiple knowledge policies and aiming for human-like efficiency and flexibility. We propose a new actor architecture for KGRL, Knowledge-Inclusive Attention Network (KIAN), which allows free knowledge rearrangement due to embedding-based attentive action prediction. KIAN also addresses entropy imbalance, a problem arising in maximum entropy KGRL that hinders an agent from efficiently exploring the environment, through a new design of policy distributions. The experimental results demonstrate that KIAN outperforms alternative methods incorporating external knowledge policies and achieves efficient and flexible learning. Our implementation is available at https://github.com/Pascalson/KGRL.git .

**Abstract(Chinese)**: 强化学习（RL）代理长期以来一直致力于接近人类学习的效率。人类是优秀的观察者，能够通过从各种来源聚合外部知识来学习，包括来自他人尝试任务的观察。在强化学习的先前研究中，已经纳入了外部知识策略以帮助代理程序提高样本效率。然而，执行这些策略的任意组合和替换仍然并不是微不足道的，这是泛化和可迁移性的重要特征。在这项工作中，我们提出了知识驱动的强化学习（KGRL），这是一种融合多种知识政策并旨在实现类人效率和灵活性的RL范式。我们提出了KGRL的新型参与者架构KIAN（Knowledge-Inclusive Attention Network），它允许由基于嵌入的专注动作预测进行自由知识重排。KIAN还解决了最大熵KGRL中出现的熵不平衡问题，该问题妨碍了代理程序有效地探索环境，通过一种新的策略分布设计。实验证明，KIAN优于纳入外部知识政策的替代方法，并实现了高效和灵活的学习。我们的实现可在 https://github.com/Pascalson/KGRL.git 上找到。

**URL**: https://nips.cc/virtual/2023/poster/70577

---

## SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning
**Author**: Dohyeok Lee · Seungyub Han · Taehyun Cho · Jungwoo Lee

**Abstract**: Alleviating overestimation bias is a critical challenge for deep reinforcement learning to achieve successful performance on more complex tasks or offline datasets containing out-of-distribution data. In order to overcome overestimation bias, ensemble methods for Q-learning have been investigated to exploit the diversity of multiple Q-functions. Since network initialization has been the predominant approach to promote diversity in Q-functions, heuristically designed diversity injection methods have been studied in the literature. However, previous studies have not attempted to approach guaranteed independence over an ensemble from a theoretical perspective. By introducing a novel regularization loss for Q-ensemble independence based on random matrix theory, we propose spiked Wishart Q-ensemble independence regularization (SPQR) for reinforcement learning. Specifically, we modify the intractable hypothesis testing criterion for the Q-ensemble independence into a tractable KL divergence between the spectral distribution of the Q-ensemble and the target Wigner's semicircle distribution. We implement SPQR in several online and offline ensemble Q-learning algorithms. In the experiments, SPQR outperforms the baseline algorithms in both online and offline RL benchmarks.

**Abstract(Chinese)**: 缓解高估偏差是深度强化学习在更复杂任务或包含分布外数据的离线数据集上取得成功表现的关键挑战。为了克服高估偏差，研究人员已经调查了Q-learning的集成方法，以利用多个Q函数的多样性。由于网络初始化一直是促进Q函数多样性的主要方法，因此在文献中已经研究了启发式设计的多样性注入方法。然而，先前的研究并未尝试从理论角度保证集成模型的独立性。通过引入基于随机矩阵理论的新型正则化损失以实现Q集成的独立性，我们提出了用于强化学习的尖峰威沙特Q集成独立正则化（SPQR）。具体来说，我们将Q集成的独立性的难题假设检测标准修改为Q集成的频谱分布与目标维格纳半圆分布的KL散度。我们在几种在线和离线集成Q-learning算法中实施了SPQR。在实验中，SPQR在在线和离线RL基准测试中均优于基准算法。

**URL**: https://nips.cc/virtual/2023/poster/70384

---

## PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks
**Author**: Ian Char · Jeff Schneider

**Abstract**: Deep reinforcement learning (RL) has shown immense potential for learning to control systems through data alone. However, one challenge deep RL faces is that the full state of the system is often not observable. When this is the case, the policy needs to leverage the history of observations to infer the current state. At the same time, differences between the training and testing environments makes it critical for the policy not to overfit to the sequence of observations it sees at training time. As such, there is an important balancing act between having the history encoder be flexible enough to extract relevant information, yet be robust to changes in the environment. To strike this balance, we look to the PID controller for inspiration. We assert the PID controller's success shows that only summing and differencing are needed to accumulate information over time for many control tasks. Following this principle, we propose two architectures for encoding history: one that directly uses PID features and another that extends these core ideas and can be used in arbitrary control tasks. When compared with prior approaches, our encoders produce policies that are often more robust and achieve better performance on a variety of tracking tasks. Going beyond tracking tasks, our policies achieve 1.7x better performance on average over previous state-of-the-art methods on a suite of locomotion control tasks.

**Abstract(Chinese)**: 深度强化学习（RL）已经展现了通过数据单独学习控制系统的巨大潜力。然而，深度RL面临的一个挑战是系统的完整状态通常是不可观测的。在这种情况下，策略需要利用历史观察来推断当前状态。与此同时，训练和测试环境之间的差异使得策略不会在训练时针对观察序列过拟合变得至关重要。因此，在历史编码器足够灵活以提取相关信息，同时又对环境变化具有鲁棒性之间需要找到一个重要的平衡。为了找到这个平衡，我们寻求PID控制器的灵感。我们断言PID控制器的成功表明，只需要对信息进行求和和差分处理即可累积时间，适用于许多控制任务。基于这一原则，我们提出了两种编码历史的架构：一种直接使用PID特征的架构，另一种扩展了这些核心思想并可用于任意控制任务的架构。与之前的方法相比，我们的编码器产生的策略在各种跟踪任务上通常更加鲁棒，并且实现了更好的性能。在超越跟踪任务的同时，我们的策略在一系列运动控制任务上平均实现了比先前最先进方法高出1.7倍的性能。

**URL**: https://nips.cc/virtual/2023/poster/70420

---

## Iterative Reachability Estimation for Safe Reinforcement Learning
**Author**: Milan Ganai · Zheng Gong · Chenning Yu · Sylvia Herbert · Sicun Gao

**Abstract**: Ensuring safety is important for the practical deployment of reinforcement learning (RL). Various challenges must be addressed, such as handling stochasticity in the environments, providing rigorous guarantees of persistent state-wise safety satisfaction, and avoiding overly conservative behaviors that sacrifice performance. We propose a new framework, Reachability Estimation for Safe Policy Optimization (RESPO), for safety-constrained RL in general stochastic settings. In the feasible set where there exist violation-free policies, we optimize for rewards while maintaining persistent safety. Outside this feasible set, our optimization produces the safest behavior by guaranteeing entrance into the feasible set whenever possible with the least cumulative discounted violations. We introduce a class of algorithms using our novel reachability estimation function to optimize in our proposed framework and in similar frameworks such as those concurrently handling multiple hard and soft constraints. We theoretically establish that our algorithms almost surely converge to locally optimal policies of our safe optimization framework. We evaluate the proposed methods on a diverse suite of safe RL environments from Safety Gym, PyBullet, and MuJoCo, and show the benefits in improving both reward performance and safety compared with state-of-the-art baselines.

**Abstract(Chinese)**: 确保安全对于强化学习（RL）的实际部署非常重要。必须解决各种挑战，比如处理环境中的随机性，提供对持久状态安全满意的严格保证，以及避免牺牲性能的过度保守行为。我们提出了一个新的框架，即针对一般随机设置下的安全约束RL的可达性估计优化（RESPO）框架。在可行集中存在无违规策略的情况下，我们会在保持持久安全的同时优化奖励。在这个可行集之外，我们的优化通过保证在可能的情况下以最小的累积折扣违规进入可行集来产生最安全的行为。我们介绍了一类使用我们的新颖可达性估计函数在我们提出的框架中进行优化的算法，以及在同时处理多个硬约束和软约束的类似框架中使用。我们在理论上建立了我们的算法几乎一定会收敛到我们的安全优化框架的局部最优策略。我们在安全学习环境套件（Safety Gym）、PyBullet和MuJoCo中评估了所提出的方法，并展示了与最先进基线相比在提高奖励性能和安全性方面的好处。

**URL**: https://nips.cc/virtual/2023/poster/70926

---

## Reinforcement Learning with Simple Sequence Priors
**Author**: Tankred Saanum · Noémi Éltető · Peter Dayan · Marcel Binz · Eric Schulz

**Abstract**: In reinforcement learning (RL), simplicity is typically quantified on an action-by-action basis -- but this timescale ignores temporal regularities, like repetitions, often present in sequential strategies. We therefore propose an RL algorithm that learns to solve tasks with sequences of actions that are compressible. We explore two possible sources of simple action sequences: Sequences that can be learned by autoregressive models, and sequences that are compressible with off-the-shelf data compression algorithms. Distilling these preferences into sequence priors, we derive a novel information-theoretic objective that incentivizes agents to learn policies that maximize rewards while conforming to these priors. We show that the resulting RL algorithm leads to faster learning, and attains higher returns than state-of-the-art model-free approaches in a series of continuous control tasks from the DeepMind Control Suite. These priors also produce a powerful information-regularized agent that is robust to noisy observations and can perform open-loop control.

**Abstract(Chinese)**: 在强化学习（RL）中，简单性通常是根据动作来量化的，但这个时间尺度忽略了时间上的规律性，比如重复，在序列策略中通常存在。因此，我们提出了一种RL算法，该算法学习解决具有可压缩动作序列的任务。我们探索了两种可能的简单动作序列来源：可以由自回归模型学习的序列，以及可以通过现成数据压缩算法进行压缩的序列。将这些偏好蒸馏成序列先验，我们推导出一种新颖的信息理论目标，鼓励智能体学习最大化奖励的策略，同时符合这些先验。我们展示，由此产生的RL算法导致更快的学习，并获得了比DeepMind控制套件中的最先进的无模型方法更高的回报，涉及连续控制任务系列。这些先验还产生了一个强大的信息正则化智能体，能够抵御嘈杂的观测并进行开环控制。

**URL**: https://nips.cc/virtual/2023/poster/70331

---

## Deep Reinforcement Learning with Plasticity Injection
**Author**: Evgenii Nikishin · Junhyuk Oh · Georg Ostrovski · Clare Lyle · Razvan Pascanu · Will Dabney · Andre Barreto

**Abstract**: A growing body of evidence suggests that neural networks employed in deep reinforcement learning (RL) gradually lose their plasticity, the ability to learn from new data; however, the analysis and mitigation of this phenomenon is hampered by the complex relationship between plasticity, exploration, and performance in RL. This paper introduces plasticity injection, a minimalistic intervention that increases the network plasticity without changing the number of trainable parameters or biasing the predictions. The applications of this intervention are two-fold: first, as a diagnostic tool — if injection increases the performance, we may conclude that an agent's network was losing its plasticity. This tool allows us to identify a subset of Atari environments where the lack of plasticity causes performance plateaus, motivating future studies on understanding and combating plasticity loss. Second, plasticity injection can be used to improve the computational efficiency of RL training if the agent has to re-learn from scratch due to exhausted plasticity or by growing the agent's network dynamically without compromising performance. The results on Atari show that plasticity injection attains stronger performance compared to alternative methods while being computationally efficient.

**Abstract(Chinese)**: 越来越多的证据表明，深度强化学习（RL）中使用的神经网络逐渐失去了其可塑性，也就是从新数据中学习的能力；然而，对这一现象的分析和缓解受到了可塑性、探索和RL中表现之间复杂关系的阻碍。本文介绍了可塑性注入，这是一种最简化的干预手段，它可以增加网络的可塑性而不改变可训练参数的数量，也不使预测产生偏差。这种干预的应用有两个方面：首先，作为一种诊断工具——如果注入增加了性能，我们可以得出结论，即一个代理的网络正在失去其可塑性。这种工具使我们能够识别Atari环境的一个子集，在这些环境中，缺乏可塑性导致性能出现平台现象，从而激发了对理解和对抗可塑性丧失的未来研究的动力。其次，如果代理由于失去可塑性而需要从头开始重新学习，或者动态地增加代理的网络大小而不影响性能，可塑性注入可以用于提高RL训练的计算效率。Atari的结果表明，相比替代方法，可塑性注入在性能上取得了更好的表现，同时计算效率更高。

**URL**: https://nips.cc/virtual/2023/poster/70670

---

## Active Vision Reinforcement Learning under Limited Visual Observability
**Author**: Jinghuan Shang · Michael S Ryoo

**Abstract**: In this work, we investigate Active Vision Reinforcement Learning (ActiveVision-RL), where an embodied agent simultaneously learns action policy for the task while also controlling its visual observations in partially observable environments. We denote the former as motor policy and the latter as sensory policy. For example, humans solve real world tasks by hand manipulation (motor policy) together with eye movements (sensory policy). ActiveVision-RL poses challenges on coordinating two policies given their mutual influence. We propose SUGARL, Sensorimotor Understanding Guided Active Reinforcement Learning, a framework that models motor and sensory policies separately, but jointly learns them using with an intrinsic sensorimotor reward. This learnable reward is assigned by sensorimotor reward module, incentivizes the sensory policy to select observations that are optimal to infer its own motor action, inspired by the sensorimotor stage of humans. Through a series of experiments, we show the effectiveness of our method across a range of observability conditions and its adaptability to existed RL algorithms. The sensory policies learned through our method are observed to exhibit effective active vision strategies.

**Abstract(Chinese)**: 在这项工作中，我们研究了主动视觉强化学习（ActiveVision-RL），其中一个具有实体的代理同时学习任务的行动策略，同时在部分可观测环境中控制其视觉观察。我们将前者表示为运动策略，后者表示为感知策略。例如，人类通过手部操作（运动策略）和眼部运动（感知策略）来解决现实世界的任务。ActiveVision-RL在协调这两个策略方面存在挑战，因为它们相互影响。我们提出了SUGARL，感觉运动理解引导的主动强化学习，这是一个模型，它分别建模了运动和感知策略，但同时使用内在的感觉运动奖励来学习它们。这种可学习的奖励是由感觉运动奖励模块分配的，鼓励感知策略选择对于推断自身运动操作最优的观察，受到人类感觉运动阶段的启发。通过一系列实验，我们展示了我们的方法在各种可观测条件下的有效性以及对现有强化学习算法的适应能力。通过我们的方法学习得到的感知策略被观察到表现出有效的主动视觉策略。

**URL**: https://nips.cc/virtual/2023/poster/70709

---

## Creating Multi-Level Skill Hierarchies in Reinforcement Learning
**Author**: Joshua B. Evans · Özgür Şimşek

**Abstract**: What is a useful skill hierarchy for an autonomous agent? We propose an answer based on a graphical representation of how the interaction between an agent and its environment may unfold. Our approach uses modularity maximisation as a central organising principle to expose the structure of the interaction graph at multiple levels of abstraction. The result is a collection of skills that operate at varying time scales, organised into a hierarchy, where skills that operate over longer time scales are composed of skills that operate over shorter time scales. The entire skill hierarchy is generated automatically, with no human input, including the skills themselves (their behaviour, when they can be called, and when they terminate) as well as the dependency structure between them. In a wide range of environments, this approach generates skill hierarchies that are intuitively appealing and that considerably improve the learning performance of the agent.

**Abstract(Chinese)**: 自主代理的有用技能层次结构是什么？我们提出了一个答案，基于代理和其环境之间交互的图形表示。我们的方法使用模块化最大化作为中心组织原则，以展示多个抽象层次上的交互图结构。结果是一系列在不同时间尺度下运行的技能，按层次组织在一起，较长时间尺度下运行的技能由较短时间尺度下运行的技能组成。整个技能层次结构是自动生成的，无需人为输入，包括技能本身（它们的行为、何时可以调用以及何时终止）以及它们之间的依赖结构。在广泛的环境中，这种方法生成的技能层次结构既直观又显著改善了代理的学习性能。

**URL**: https://nips.cc/virtual/2023/poster/70828

---

## Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?
**Author**: Jialu Gao · Kaizhe Hu · Guowei Xu · Huazhe Xu

**Abstract**: Pre-trained text-to-image generative models can produce diverse, semantically rich, and realistic images from natural language descriptions. Compared with language, images usually convey information with more details and less ambiguity. In this study, we propose Learning from the Void (LfVoid), a method that leverages the power of pre-trained text-to-image models and advanced image editing techniques to guide robot learning. Given natural language instructions, LfVoid can edit the original observations to obtain goal images, such as "wiping" a stain off a table. Subsequently, LfVoid trains an ensembled goal discriminator on the generated image to provide reward signals for a reinforcement learning agent, guiding it to achieve the goal. The ability of LfVoid to learn with zero in-domain training on expert demonstrations or true goal observations (the void) is attributed to the utilization of knowledge from web-scale generative models. We evaluate LfVoid across three simulated tasks and validate its feasibility in the corresponding real-world scenarios. In addition, we offer insights into the key considerations for the effective integration of visual generative models into robot learning workflows. We posit that our work represents an initial step towards the broader application of pre-trained visual generative models in the robotics field. Our project page: https://lfvoid-rl.github.io/.

**Abstract(Chinese)**: 预训练的文本到图像生成模型可以从自然语言描述中生成多样性、语义丰富且逼真的图像。与语言相比，图像通常以更多细节和更少歧义地传达信息。在本研究中，我们提出了一种名为Learning from the Void (LfVoid)的方法，该方法利用预训练的文本到图像模型和先进的图像编辑技术来指导机器人学习。给定自然语言指令，LfVoid可以编辑原始观察以获得目标图像，比如“擦掉”桌子上的污渍。随后，LfVoid对生成的图像训练了一个集成目标鉴别器，为强化学习代理提供奖励信号，指导其实现目标。LfVoid能够在专家演示或真实目标观察（虚空）的零领域训练中学习的能力，归因于利用了来自规模化网络生成模型的知识。我们在三个模拟任务中评估了LfVoid，并验证了其在相应的真实场景中的可行性。此外，我们提供了有关将视觉生成模型有效集成到机器人学习工作流程中的关键考虑因素的见解。我们认为我们的工作代表了预训练视觉生成模型在机器人领域更广泛应用的初步步骤。我们的项目页面：https://lfvoid-rl.github.io/。

**URL**: https://nips.cc/virtual/2023/poster/70655

---

## Multi-Agent Meta-Reinforcement Learning: Sharper Convergence Rates with Task Similarity
**Author**: Weichao Mao · Haoran Qiu · Chen Wang · Hubertus Franke · Zbigniew Kalbarczyk · Ravishankar Iyer · Tamer Basar

**Abstract**: Multi-agent reinforcement learning (MARL) has primarily focused on solving a single task in isolation, while in practice the environment is often evolving, leaving many related tasks to be solved. In this paper, we investigate the benefits of meta-learning in solving multiple MARL tasks collectively. We establish the first line of theoretical results for meta-learning in a wide range of fundamental MARL settings, including learning Nash equilibria in two-player zero-sum Markov games and Markov potential games, as well as learning coarse correlated equilibria in general-sum Markov games. Under natural notions of task similarity, we show that meta-learning achieves provable sharper convergence to various game-theoretical solution concepts than learning each task separately. As an important intermediate step, we develop multiple MARL algorithms with initialization-dependent convergence guarantees. Such algorithms integrate optimistic policy mirror descents with stage-based value updates, and their refined convergence guarantees (nearly) recover the best known results even when a good initialization is unknown. To our best knowledge, such results are also new and might be of independent interest. We further provide numerical simulations to corroborate our theoretical findings.

**Abstract(Chinese)**: 多智能体强化学习（MARL）主要侧重于在孤立环境中解决单一任务，而实际上环境经常在演变，导致许多相关任务需要解决。本文研究了元学习在集体解决多个MARL任务中的益处。我们在一系列基础MARL设置下建立了元学习的首批理论结果，其中包括学习两人零和马尔可夫博弈和马尔可夫潜在博弈的纳什均衡，以及在一般和马尔可夫博弈中学习粗糙相关均衡。在自然的任务相似性概念下，我们表明元学习达到了对各种博弈理论解概念的收敛证明比单独学习每个任务更加明显。作为一个重要的中间步骤，我们开发了多个MARL算法，具有依赖初始化的收敛保证。这些算法将乐观策略镜像下降与基于阶段的价值更新相结合，它们的改进的收敛保证（几乎）即使在不知道良好初始化时，也能恢复最佳已知结果。据我们所知，这样的结果也是新的，可能具有独立的兴趣。我们还提供了数值模拟来证实我们的理论发现。

**URL**: https://nips.cc/virtual/2023/poster/73066

---

## BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning
**Author**: Xuan Chen · Wenbo Guo · Wenbo Guo · Guanhong Tao · Xiangyu Zhang · Dawn Song

**Abstract**: Backdoor attacks pose a severe threat to the supply chain management of deep reinforcement learning (DRL) policies. Despite initial defenses proposed in recent studies, these methods have very limited generalizability and scalability. To address this issue, we propose BIRD, a technique to detect and remove backdoors from a pretrained DRL policy in a clean environment without requiring any knowledge about the attack specifications and accessing its training process. By analyzing the unique properties and behaviors of backdoor attacks, we formulate trigger restoration as an optimization problem and design a novel metric to detect backdoored policies. We also design a finetuning method to remove the backdoor, while maintaining the agent's performance in the clean environment. We evaluate BIRD against three backdoor attacks in ten different single-agent or multi-agent environments. Our results verify the effectiveness, efficiency, and generalizability of BIRD, as well as its robustness to different attack variations and adaptions.

**Abstract(Chinese)**: 后门攻击对深度强化学习（DRL）政策的供应链管理构成严重威胁。尽管最近的研究提出了最初的防御措施，但这些方法的泛化能力和可扩展性非常有限。为了解决这个问题，我们提出了BIRD，这是一种技术，可以在干净的环境中检测和移除预先训练的DRL政策中的后门，而无需了解攻击规格并访问其训练过程。通过分析后门攻击的独特属性和行为，我们将触发器恢复作为一个优化问题，并设计了一种新的度量来检测安装了后门的政策。我们还设计了一种微调方法来清除后门，同时保持智能体在干净环境中的性能。我们在十个不同的单智能体或多智能体环境中对BIRD进行了三次后门攻击的评估。我们的结果验证了BIRD的有效性、效率和泛化能力，以及其对不同攻击变体和适应性的鲁棒性。

**URL**: https://nips.cc/virtual/2023/poster/70618

---

## Regret-Optimal Model-Free Reinforcement Learning for Discounted MDPs with Short Burn-In Time
**Author**: Xiang Ji · Gen Li

**Abstract**: A crucial problem in reinforcement learning is learning the optimal policy. We study this in tabular infinite-horizon discounted Markov decision processes under the online setting. The existing algorithms either fail to achieve regret optimality or have to incur a high memory and computational cost. In addition, existing optimal algorithms all require a long burn-in time in order to achieve optimal sample efficiency, i.e., their optimality is not guaranteed unless sample size surpasses a high threshold. We address both open problems by introducing a model-free algorithm that employs variance reduction and a novel technique that switches the execution policy in a slow-yet-adaptive manner. This is the first regret-optimal model-free algorithm in the discounted setting, with the additional benefit of a low burn-in time.

**Abstract(Chinese)**: 在强化学习中一个关键问题是学习最优策略。我们研究了在线设置下的表格无限时间段折扣马尔科夫决策过程。现有算法要么无法达到遗憾最优性，要么需要付出很高的内存和计算成本。此外，现有的最优算法都需要很长的燃烧期时间才能实现最优的样本效率，即除非样本大小超过一个很高的阈值，否则其最优性无法得到保证。我们通过引入一种无模型算法来解决这两个开放性问题，该算法采用方差减少和一种新颖的技术，在慢而自适应的方式下切换执行策略。这是第一个在折扣设置下达到遗憾最优的无模型算法，还能额外获得低燃烧期时间的好处。

**URL**: https://nips.cc/virtual/2023/poster/70515

---

## Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds
**Author**: Jiayi Huang · Han Zhong · Liwei Wang · Lin Yang

**Abstract**: While numerous works have focused on devising efficient algorithms for reinforcement learning (RL) with uniformly bounded rewards, it remains an open question whether sample or time-efficient algorithms for RL with large state-action space exist when the rewards are \emph{heavy-tailed}, i.e., with only finite $(1+\epsilon)$-th moments for some $\epsilon\in(0,1]$. In this work, we address the challenge of such rewards in RL with linear function approximation. We first design an algorithm, \textsc{Heavy-OFUL}, for heavy-tailed linear bandits, achieving an \emph{instance-dependent} $T$-round regret of $\tilde{O}\big(d T^{\frac{1-\epsilon}{2(1+\epsilon)}} \sqrt{\sum_{t=1}^T \nu_t^2} + d T^{\frac{1-\epsilon}{2(1+\epsilon)}}\big)$, the \emph{first} of this kind. Here, $d$ is the feature dimension, and $\nu_t^{1+\epsilon}$ is the $(1+\epsilon)$-th central moment of the reward at the $t$-th round. We further show the above bound is minimax optimal when applied to the worst-case instances in stochastic and deterministic linear bandits. We then extend this algorithm to the RL settings with linear function approximation. Our algorithm, termed as \textsc{Heavy-LSVI-UCB}, achieves the \emph{first} computationally efficient \emph{instance-dependent} $K$-episode regret of $\tilde{O}(d \sqrt{H \mathcal{U}^*} K^\frac{1}{1+\epsilon} + d \sqrt{H \mathcal{V}^* K})$. Here, $H$ is length of the episode, and $\mathcal{U}^*, \mathcal{V}^*$ are instance-dependent quantities scaling with the central moment of reward and value functions, respectively. We also provide a matching minimax lower bound $\Omega(d H K^{\frac{1}{1+\epsilon}} + d \sqrt{H^3 K})$ to demonstrate the optimality of our algorithm in the worst case. Our result is achieved via a novel robust self-normalized concentration inequality that may be of independent interest in handling heavy-tailed noise in general online regression problems.

**Abstract(Chinese)**: 尽管许多研究着眼于设计强化学习中的高效算法（RL），以处理有限界奖励，但当奖励呈\emph{重尾}分布时，即对于某些$\epsilon\in(0,1]$，仅有有限的$(1+\epsilon)$-th矩存在时，关于RL具有大状态-动作空间的样本或时间效率算法是否存在仍然是一个开放问题。在本文中，我们解决了使用线性函数逼近RL中此类奖励的挑战。我们首先设计了一种算法\textsc{Heavy-OFUL}，用于处理重尾线性赌臂问题，实现了\emph{实例相关}的$T$轮遗憾，为$\tilde{O}\big(d T^{\frac{1-\epsilon}{2(1+\epsilon)}} \sqrt{\sum_{t=1}^T \nu_t^2} + d T^{\frac{1-\epsilon}{2(1+\epsilon)}}\big)$，这是\emph{首次}实现。这里，$d$是特征维度，$\nu_t^{1+\epsilon}$是第$t$轮奖励的$(1+\epsilon)$-th中心矩。我们进一步证明了当应用于随机和确定性线性赌臂的最坏情况实例时，上述界是最小最大最优的。然后，将此算法扩展到了具有线性函数逼近的RL设置。我们的算法，命名为\textsc{Heavy-LSVI-UCB}，实现了\emph{第一次}计算上高效的\emph{实例相关}的$K$局遗憾，为$\tilde{O}(d \sqrt{H \mathcal{U}^*} K^\frac{1}{1+\epsilon} + d \sqrt{H \mathcal{V}^* K})$。这里，$H$是每集的长度，$\mathcal{U}^*, \mathcal{V}^*$是与奖励和值函数的中心矩相关的实例相关数量。我们还提供了一个匹配最小最大下界$\Omega(d H K^{\frac{1}{1+\epsilon}} + d \sqrt{H^3 K})$，以展示我们算法在最坏情况下的最优性。我们的结果通过一种新颖的鲁棒自标准浓缩不等式实现，这可能是处理一般在线回归问题中的重尾噪声时独立感兴趣的问题。

**URL**: https://nips.cc/virtual/2023/poster/70887

---

## Provably Efficient Offline Goal-Conditioned Reinforcement Learning with General Function Approximation and Single-Policy Concentrability
**Author**: Hanlin Zhu · Amy Zhang

**Abstract**: Goal-conditioned reinforcement learning (GCRL) refers to learning general-purpose skills that aim to reach diverse goals. In particular, offline GCRL only requires purely pre-collected datasets to perform training tasks without additional interactions with the environment. Although offline GCRL has become increasingly prevalent and many previous works have demonstrated its empirical success, the theoretical understanding of efficient offline GCRL algorithms is not well established, especially when the state space is huge and the offline dataset only covers the policy we aim to learn. In this paper, we provide a rigorous theoretical analysis of an existing empirically successful offline GCRL algorithm. We prove that under slight modification, this algorithm enjoys an $\tilde{O}(\text{poly}(1/\epsilon))$ sample complexity (where $\epsilon$ is the desired suboptimality of the learned policy) with general function approximation thanks to the property of (semi-)strong convexity of the objective functions. We only require nearly minimal assumptions on the dataset (single-policy concentrability) and the function class (realizability). Moreover, this algorithm consists of two uninterleaved optimization steps, which we refer to as $V$-learning and policy learning, and is computationally stable since it does not involve minimax optimization. We also empirically validate our theory by showing that the modified algorithm outperforms the previous algorithm in various real-world environments.To the best of our knowledge, this is the first algorithm that is both provably efficient with general function approximation and single-policy concentrability, and empirically successful without requiring solving minimax optimization problems.

**Abstract(Chinese)**: 目标导向强化学习（GCRL）是指学习通用技能，旨在实现多样化目标。特别是离线GCRL只需要纯粹预先收集的数据集来执行训练任务，而无需与环境进行额外的交互。尽管离线GCRL变得日益普遍，许多先前的研究已经证明了其经验成功，但是尤其在状态空间巨大且离线数据集仅覆盖我们希望学习的策略时，对于高效的离线GCRL算法的理论理解还不够成熟。在本文中，我们对一种现有的在经验上成功的离线GCRL算法进行了严格的理论分析。我们证明，通过轻微修改，该算法在一般函数逼近的情况下享有近似$\tilde{O}(\text{poly}(1/\epsilon))$的样本复杂度（其中$\epsilon$是学习策略的期望次优性），这要归功于（半）强凸性目标函数的性质。我们对数据集（单策略可集中性）和函数类（可实现性）仅需要几乎最小的假设。此外，该算法包括两个不相间隔的优化步骤，我们称之为$V$-学习和策略学习，并且在计算上是稳定的，因为它不涉及极小-极大优化。我们还通过经验验证了我们的理论，证明修改后的算法在各种现实环境中优于先前的算法。据我们所知，这是第一个既在一般函数逼近又在单策略可集中性上具有可证效率、并且在不需要解决极小-极大优化问题的情况下在经验上成功的算法。

**URL**: https://nips.cc/virtual/2023/poster/70359

---

## Tempo Adaptation in Non-stationary Reinforcement Learning
**Author**: Hyunin Lee · Yuhao Ding · Jongmin Lee · Ming Jin · Javad Lavaei · Somayeh Sojoudi

**Abstract**: We first raise and tackle a ``time synchronization'' issue between the agent and the environment in non-stationary reinforcement learning (RL), a crucial factor hindering its real-world applications. In reality, environmental changes occur over wall-clock time ($t$) rather than episode progress ($k$), where wall-clock time signifies the actual elapsed time within the fixed duration $t \in [0, T]$. In existing works, at episode $k$, the agent rolls a trajectory and trains a policy before transitioning to episode $k+1$. In the context of the time-desynchronized environment, however, the agent at time $t_{k}$ allocates $\Delta t$ for trajectory generation and training, subsequently moves to the next episode at $t_{k+1}=t_{k}+\Delta t$. Despite a fixed total number of episodes ($K$), the agent accumulates different trajectories influenced by the choice of interaction times ($t_1,t_2,...,t_K$), significantly impacting the suboptimality gap of the policy. We propose a Proactively Synchronizing Tempo ($\texttt{ProST}$) framework that computes a suboptimal sequence {$t_1,t_2,...,t_K$} (= { $t_{1:K}$}) by minimizing an upper bound on its performance measure, i.e., the dynamic regret. Our main contribution is that we show that a suboptimal {$t_{1:K}$} trades-off between the policy training time (agent tempo) and how fast the environment changes (environment tempo). Theoretically, this work develops a suboptimal {$t_{1:K}$} as a function of the degree of the environment's non-stationarity while also achieving a sublinear dynamic regret. Our experimental evaluation on various high-dimensional non-stationary environments shows that the $\texttt{ProST}$ framework achieves a higher online return at suboptimal {$t_{1:K}$} than the existing methods.

**Abstract(Chinese)**: 我们首先在非静态强化学习（RL）中提出并解决了代理和环境之间的“时间同步”问题，这是阻碍其在现实世界应用中的关键因素。实际上，环境变化发生在挂钟时间（$t$）而不是在情节进展（$k$）中，其中挂钟时间表示在固定持续时间 $t \in [0, T]$ 内实际经过的时间。在现有的研究中，代理在情节 $k$ 处生成一条轨迹并训练策略，然后过渡到情节 $k+1$。然而，在时钟不同步的环境情况下，代理在时间 $t_{k}$ 分配 $\Delta t$ 来生成轨迹和训练，随后在 $t_{k+1}=t_{k}+\Delta t$ 处转移到下一情节。尽管情节的总数是固定的（$K$），代理积累了由相互作用时间的选择 ($t_1,t_2,...,t_K$) 影响的不同轨迹，显著影响了策略的次优性差距。我们提出了一个主动同步节奏（$\texttt{ProST}$）框架，通过最小化其性能度量的上界，即动态遗憾，来计算一个次优序列 {$t_1,t_2,...,t_K$} （= { $t_{1:K}$}）。我们的主要贡献是，我们展示了次优 {$t_{1:K}$} 在策略训练时间（代理节奏）和环境变化速度（环境节奏）之间进行权衡。从理论上讲，该工作将一个次优 {$t_{1:K}$} 开发为环境非静态性的程度的函数，同时实现一个次线性动态遗憾。我们在各种高维非静态环境上的实验评估表明，$\texttt{ProST}$ 框架在次优 {$t_{1:K}$} 上实现了更高的在线回报，而现有方法没有。

**URL**: https://nips.cc/virtual/2023/poster/72708

---

## Policy Finetuning in Reinforcement Learning via Design of Experiments using Offline Data
**Author**: Ruiqi Zhang · Andrea Zanette

**Abstract**: In some applications of reinforcement learning, a dataset of pre-collected experience is already availablebut it is also possible to acquire some additional online data to help improve the quality of the policy.However, it may be preferable to gather additional data with a single, non-reactive exploration policyand avoid the engineering costs associated with switching policies. In this paper we propose an algorithm with provable guarantees that can leverage an offline dataset to design a single non-reactive policy for exploration. We theoretically analyze the algorithm and measure the quality of the final policy as a function of the local coverage of the original dataset and the amount of additional data collected.

**Abstract(Chinese)**: 在一些强化学习的应用中，已经有一些预先收集的经验数据集，但也有可能获取一些额外的线上数据，以帮助提高策略的质量。然而，最好的方式可能是使用单一的、非反应性的探索策略来收集额外的数据，并避免与切换策略相关的工程成本。在本文中，我们提出了一种具有可证明保证的算法，可以利用离线数据集来设计一个单一的非反应性探索策略。我们在理论上分析了该算法，并根据原始数据集的局部覆盖程度和额外收集的数据量来衡量最终策略的质量。

**URL**: https://nips.cc/virtual/2023/poster/70892

---

## Learning to Influence Human Behavior with Offline Reinforcement Learning
**Author**: Joey Hong · Sergey Levine · Anca Dragan

**Abstract**: When interacting with people, AI agents do not just influence the state of the world -- they also influence the actions people take in response to the agent, and even their underlying intentions and strategies. Accounting for and leveraging this influence has mostly been studied in settings where it is sufficient to assume that human behavior is near-optimal: competitive games, or general-sum settings like autonomous driving alongside human drivers. Instead, we focus on influence in settings where there is a need to capture human suboptimality. For instance, imagine a collaborative task in which, due either to cognitive biases or lack of information, people do not perform very well -- how could an agent influence them towards more optimal behavior? Assuming near-optimal human behavior will not work here, and so the agent needs to learn from real human data. But experimenting online with humans is potentially unsafe, and creating a high-fidelity simulator of the environment is often impractical. Hence, we  focus on learning from an offline dataset of human-human interactions. Our observation is that offline reinforcement learning (RL) can learn to effectively influence suboptimal humans by extending and combining elements of observed human-human behavior. We demonstrate that offline RL can solve two challenges with effective influence. First, we show that by learning from a dataset of suboptimal human-human interaction on a variety of tasks -- none of which contains examples of successful influence -- an agent can learn influence strategies to steer humans towards better performance even on new tasks. Second, we show that by also modeling and conditioning on human behavior, offline RL can learn to affect not just the human's actions but also their underlying strategy, and adapt to changes in their strategy.

**Abstract(Chinese)**: 当与人们互动时，AI代理不仅影响世界的状态，还影响人们对代理采取的行动，甚至影响他们的基本意图和策略。考虑和利用这种影响大多在研究中得到了实践，而这种研究往往发生在足够假设人类行为接近最优的情境中：如竞技游戏，或者自动驾驶与人类司机并存的一般和局设置。相反，我们关注的是在需要捕捉人类次优性的情境中的影响。例如，想象一种合作任务，在这种任务中，由于认知偏见或缺乏信息，人们表现不佳——代理如何影响他们朝着更优化的行为发展？在这里假设人类行为接近最优将行不通，因此代理需要从真实的人类数据中学习。但是在网上与人类进行实验可能是不安全的，而且创建一个高度逼真的环境模拟器通常是不切实际的。因此，我们着眼于从人际互动的离线数据集中学习。我们观察到，离线强化学习（RL）通过延伸和结合观察到的人-人行为元素，可以有效地影响次优的人类。我们证明了离线RL可以解决有效影响的两个挑战。首先，我们展示了通过从各种任务的次优人-人互动数据集中学习——其中都没有成功影响的例子——代理可以学习影响策略，甚至在新任务中指引人类获得更好的表现。其次，我们表明，通过建模和对人类行为加以约束，离线RL不仅可以学习影响人类的行动，还可以影响他们的基础策略，并适应策略的变化。

**URL**: https://nips.cc/virtual/2023/poster/70757

---

## TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning
**Author**: Shuo Sun · Molei Qin · Wentao Zhang · Haochong Xia · Chuqiao Zong · Jie Ying · Yonggang Xie · Lingxuan Zhao · Xinrun Wang · Bo An

**Abstract**: The financial markets, which involve over \$90 trillion market capitals, attract the attention of innumerable profit-seeking investors globally. Recent explosion of reinforcement learning in financial trading (RLFT) research has shown stellar performance on many quantitative trading tasks. However, it is still challenging to deploy reinforcement learning (RL) methods into real-world financial markets due to the highly composite nature of this domain, which entails design choices and interactions between components that collect financial data, conduct feature engineering, build market environments, make investment decisions, evaluate model behaviors and offers user interfaces. Despite the availability of abundant financial data and advanced RL techniques, a remarkable gap still exists between the potential and realized utilization of RL in financial trading. In particular, orchestrating an RLFT project lifecycle poses challenges in engineering (i.e. hard to build), benchmarking (i.e. hard to compare) and usability (i.e. hard to optimize, maintain and use). To overcome these challenges, we introduce TradeMaster, a holistic open-source RLFT platform that serves as a i) software toolkit, ii) empirical benchmark, and iii) user interface. Our ultimate goal is to provide infrastructures for transparent and reproducible RLFT research and facilitate their real-world deployment with industry impact. TradeMaster will be updated continuously and welcomes contributions from both RL and finance communities.

**Abstract(Chinese)**: 金融市场涉及超过90万亿美元的市值，吸引着全球无数追求利润的投资者的关注。最近，强化学习在金融交易（RLFT）研究领域的爆发展现出了在许多量化交易任务上的出色表现。然而，由于金融市场本质的高度复合性，其中涉及收集金融数据、开展特征工程、构建市场环境、做出投资决策、评估模型行为以及提供用户界面等组件之间的设计选择和交互，因此将强化学习（RL）方法应用于现实世界的金融市场仍然具有挑战性。尽管有丰富的金融数据和先进的强化学习技术，但在金融交易中潜在利用和实际利用强化学习之间仍然存在显著差距。特别是，在组织RLFT项目生命周期方面，工程（即难以构建）、基准测试（即难以比较）和可用性（即难以优化、维护和使用）方面存在挑战。为了克服这些挑战，我们引入了TradeMaster，一个全面的开源RLFT平台，它充当i）软件工具包、ii）经验基准和iii）用户界面。我们的最终目标是为透明和可重现的RLFT研究提供基础设施，并促进它们在具有行业影响的现实世界中的部署。TradeMaster将不断更新，并欢迎来自RL和金融社区的贡献。

**URL**: https://nips.cc/virtual/2023/poster/73483

---

## Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning
**Author**: Fuqi Jia · Yuhang Dong · Minghao Liu · Pei Huang · Feifei Ma · Jian Zhang

**Abstract**: Cylindrical Algebraic Decomposition (CAD) is one of the pillar algorithms of symbolic computation, and its worst-case complexity is double exponential to the number of variables. Researchers found that variable order dramatically affects efficiency and proposed various heuristics. The existing learning-based methods are all supervised learning methods that cannot cope with diverse polynomial sets.This paper proposes two Reinforcement Learning (RL) approaches combined with Graph Neural Networks (GNN) for Suggesting Variable Order (SVO). One is GRL-SVO(UP), a branching heuristic integrated with CAD. The other is GRL-SVO(NUP), a fast heuristic providing a total order directly. We generate a random dataset and collect a real-world dataset from SMT-LIB. The experiments show that our approaches outperform state-of-the-art learning-based heuristics and are competitive with the best expert-based heuristics. Interestingly, our models show a strong generalization ability, working well on various datasets even if they are only trained on a 3-var random dataset. The source code and data are available at https://github.com/dongyuhang22/GRL-SVO.

**Abstract(Chinese)**: 圆柱代数分解（CAD）是符号计算中的支柱算法之一，其最坏情况复杂度是相对于变量数量的双指数级。研究人员发现变量顺序会极大地影响效率，并提出了各种启发式方法。现有的基于学习的方法都是监督学习方法，无法应对多样的多项式集合。本文提出了两种结合图神经网络（GNN）的强化学习（RL）方法，用于建议变量顺序（SVO）。一种是GRL-SVO(UP)，它是一个与CAD集成的分支启发式方法。另一种是GRL-SVO(NUP)，它是一个快速启发式方法，可直接提供总顺序。我们生成了一个随机数据集，并从SMT-LIB中收集了一个真实世界的数据集。实验证明，我们的方法在性能上胜过了最先进的基于学习的启发式方法，并与最好的基于专家的启发式方法竞争。有趣的是，我们的模型表现出很强的泛化能力，即使它们只是在一个包含3个变量的随机数据集上进行训练，也能在各种数据集上表现良好。源代码和数据可在https://github.com/dongyuhang22/GRL-SVO上获得。

**URL**: https://nips.cc/virtual/2023/poster/70104

---

## Reflexion: language agents with verbal reinforcement learning
**Author**: Noah Shinn · Federico Cassano · Ashwin Gopinath · Karthik Narasimhan · Shunyu Yao

**Abstract**: Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose \emph{Reflexion}, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91\% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance. We release all code, demos, and datasets at \url{https://github.com/noahshinn024/reflexion}.

**Abstract(Chinese)**: 大型语言模型（LLMs）越来越被用来与外部环境（例如游戏、编译器、API）交互，作为目标驱动的代理。然而，对于这些语言代理来说，通过传统的强化学习方法来快速高效地进行试错学习仍然具有挑战，因为这需要大量的训练样本和昂贵的模型微调。我们提出\emph{Reflexion}，这是一个新颖的框架，用于加强语言代理，不是通过更新权重，而是通过语言反馈。具体地，Reflexion代理在任务反馈信号上进行语言反思，然后在一个叙事式内存缓冲区中保留自己的反思文本，以在后续试验中引导更好的决策。Reflexion足够灵活，可以整合各种类型（标量值或自由形式语言）和来源（外部或内部模拟）的反馈信号，并且在不同任务（顺序决策制定、编码、语言推理）中相对于基线代理取得显著改进。例如，Reflexion在HumanEval编码基准测试中达到91\%的一次通过准确率，超过了之前最先进的GPT-4达到的80\%。我们还使用不同的反馈信号、反馈整合方法和代理类型进行剔除和分析研究，并提供了它们对性能的影响的见解。我们在\url{https://github.com/noahshinn024/reflexion}发布了所有的代码、演示和数据集。

**URL**: https://nips.cc/virtual/2023/poster/70114

---

## Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark
**Author**: Jiaming Ji · Borong Zhang · Jiayi Zhou · Xuehai Pan · Weidong Huang · Ruiyang Sun · Yiran Geng · Yifan Zhong · Josef Dai · Yaodong Yang

**Abstract**: Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called Safety-Gymnasium, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.

**Abstract(Chinese)**: 人工智能（AI）系统具有推动社会进步的重要潜力。然而，由于存在重大安全隐患，它们的部署常常面临障碍。安全强化学习（SafeRL）成为一种解决方案，可以在优化策略的同时遵守多个约束条件，从而解决在安全关键场景中集成强化学习的挑战。在本文中，我们提出了一个名为Safety-Gymnasium的环境套件，其中包括单个和多智能体情况下的安全关键任务，接受矢量和仅视觉输入。此外，我们提供了一个名为Safe Policy Optimization（SafePO）的算法库，其中包括16个最先进的SafeRL算法。这一全面的算法库可以作为研究社区的验证工具。通过引入这一基准，我们旨在促进对安全性能的评估和比较，从而推动强化学习在更安全、更可靠和更负责的真实世界应用中的发展。该项目的网站可在https://sites.google.com/view/safety-gymnasium上访问。

**URL**: https://nips.cc/virtual/2023/poster/73567

---

## MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning
**Author**: Zeyuan Ma · Hongshu Guo · Jiacheng Chen · Zhenrui Li · Guojun Peng · Yue-Jiao Gong · Yining Ma · Zhiguang Cao

**Abstract**: Recently, Meta-Black-Box Optimization with Reinforcement Learning (MetaBBO-RL) has showcased the power of leveraging RL at the meta-level to mitigate manual fine-tuning of low-level black-box optimizers. However, this field is hindered by the lack of a unified benchmark. To fill this gap, we introduce MetaBox, the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. Our MetaBox is open-source and accessible at: https://github.com/GMC-DRL/MetaBox.

**Abstract(Chinese)**: 最近，元-黑盒优化与强化学习（MetaBBO-RL）展示了利用元级别的强化学习来减轻低级黑盒优化器的手动微调的能力。然而，这一领域受到统一基准测试的缺乏阻碍。为了填补这一空白，我们推出MetaBox，这是专门为开发和评估MetaBBO-RL方法而量身定制的第一个基准平台。MetaBox提供了一个灵活的算法模板，允许用户在平台内轻松实现其独特设计。此外，它还提供了广泛的问题实例范围，从合成到现实场景，以及包括传统黑盒优化器和最近的MetaBBO-RL方法在内的19种基准方法的广泛库。此外，MetaBox引入了三个标准化性能指标，使方法的评估更加全面。为了展示MetaBox在促进严格评估和深入分析方面的实用性，我们对现有的MetaBBO-RL方法进行了广泛的基准研究。我们的MetaBox是开源的，可在以下网址获取：https://github.com/GMC-DRL/MetaBox。

**URL**: https://nips.cc/virtual/2023/poster/73497

---

## Semantic HELM: A Human-Readable Memory for Reinforcement Learning
**Author**: Fabian Paischer · Thomas Adler · Markus Hofmarcher · Sepp Hochreiter

**Abstract**: Reinforcement learning agents deployed in the real world often have to cope with partially observable environments. Therefore, most agents employ memory mechanisms to approximate the state of the environment. Recently, there have been impressive success stories in mastering partially observable environments, mostly in the realm of computer games like Dota 2, StarCraft II, or MineCraft. However, existing methods lack interpretability in the sense that it is not comprehensible for humans what the agent stores in its memory.In this regard, we propose a novel memory mechanism that represents past events in human language.Our method uses CLIP to associate visual inputs with language tokens. Then we feed these tokens to a pretrained language model that serves the agent as memory and provides it with a coherent and human-readable representation of the past.We train our memory mechanism on a set of partially observable environments and find that it excels on tasks that require a memory component, while mostly attaining performance on-par with strong baselines on tasks that do not. On a challenging continuous recognition task, where memorizing the past is crucial, our memory mechanism converges two orders of magnitude faster than prior methods.Since our memory mechanism is human-readable, we can peek at an agent's memory and check whether crucial pieces of information have been stored.This significantly enhances troubleshooting and paves the way toward more interpretable agents.

**Abstract(Chinese)**: 现实世界中部署的强化学习代理通常必须应对部分可观察环境。因此，大多数代理使用内存机制来近似环境的状态。最近，在掌握部分可观察环境方面取得了令人印象深刻的成功故事，主要是在计算机游戏领域，如Dota 2、星际争霸 II或我的世界。然而，现有方法在可解释性方面存在不足，即人类无法理解代理存储在内存中的内容。在这方面，我们提出了一种新颖的内存机制，用人类语言表示过去事件。我们的方法使用CLIP将视觉输入与语言标记相关联。然后，我们将这些标记馈送到预训练的语言模型，作为代理的内存，并为其提供过去的连贯且可读的表示。我们在一组部分可观察的环境中训练我们的内存机制，并发现它在需要内存组件的任务上表现出色，而在不需要的任务上，其表现大多与强基线相当。在具有挑战性的连续识别任务中，过去的记忆至关重要，我们的内存机制的收敛速度比先前的方法快两个数量级。由于我们的内存机制是可读的，我们可以查看代理的记忆，并检查关键信息是否已存储。这显著提高了故障排除能力，并为更具解释性的代理铺平了道路。

**URL**: https://nips.cc/virtual/2023/poster/73882

---

## Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks
**Author**: Maxime Chevalier-Boisvert · Bolun Dai · Mark Towers · Rodrigo Perez-Vicente · Lucas Willems · Salem Lahlou · Suman Pal · Pablo Samuel Castro · J Terry

**Abstract**: We present the Minigrid and Miniworld libraries which provide a suite of goal-oriented 2D and 3D environments. The libraries were explicitly created with a minimalistic design paradigm to allow users to rapidly develop new environments for a wide range of research-specific needs. As a result, both have received widescale adoption by the RL community, facilitating research in a wide range of areas. In this paper, we outline the design philosophy, environment details, and their world generation API.  We also showcase the additional capabilities brought by the unified API between Minigrid and Miniworld through case studies on transfer learning (for both RL agents and humans) between the different observation spaces. The source code of Minigrid and Miniworld can be found at https://github.com/Farama-Foundation/Minigrid and https://github.com/Farama-Foundation/Miniworld along with their documentation at https://minigrid.farama.org/ and https://miniworld.farama.org/.

**Abstract(Chinese)**: 我们提供Minigrid和Miniworld库，这些库提供了一套面向目标的2D和3D环境。这些库是根据极简设计范式明确创建的，以便让用户快速为各种特定研究需求开发新的环境。因此，两者已被RL社区广泛采用，促进了广泛领域的研究。在本文中，我们概述了设计哲学、环境细节和它们的世界生成API。我们还通过案例研究展示了Minigrid和Miniworld统一API带来的额外功能，包括在不同观察空间之间进行的传输学习（针对RL代理和人类）。Minigrid和Miniworld的源代码可在https://github.com/Farama-Foundation/Minigrid和https://github.com/Farama-Foundation/Miniworld找到，同时它们的文档可在https://minigrid.farama.org/和https://miniworld.farama.org/找到。

**URL**: https://nips.cc/virtual/2023/poster/73602

---

## Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks
**Author**: Yun Qu · Boyuan Wang · Jianzhun Shao · Yuhang Jiang · Chen Chen · Zhenbin Ye · Liu Linc · Yang Feng · Lin Lai · Hongyang Qin · Minwen Deng · Juchao Zhuo · Deheng Ye · Qiang Fu · YANG GUANG · Wei Yang · Lanxiao Huang · Xiangyang Ji

**Abstract**: The advancement of Offline Reinforcement Learning (RL) and Offline Multi-Agent Reinforcement Learning (MARL) critically depends on the availability of high-quality, pre-collected offline datasets that represent real-world complexities and practical applications. However, existing datasets often fall short in their simplicity and lack of realism. To address this gap, we propose Hokoff, a comprehensive set of pre-collected datasets that covers both offline RL and offline MARL, accompanied by a robust framework, to facilitate further research. This data is derived from Honor of Kings, a recognized Multiplayer Online Battle Arena (MOBA) game known for its intricate nature, closely resembling real-life situations. Utilizing this framework, we benchmark a variety of offline RL and offline MARL algorithms. We also introduce a novel baseline algorithm tailored for the inherent hierarchical action space of the game. We reveal the incompetency of current offline RL approaches in handling task complexity, generalization and multi-task learning.

**Abstract(Chinese)**: 离线强化学习（RL）和离线多智体强化学习（MARL）的推进关键取决于高质量、预先收集的离线数据集的可用性，这些数据集代表现实世界的复杂性和实际应用。然而，现有数据集通常在简单性和现实性方面存在不足。为了解决这一问题，我们提出Hokoff，这是一套全面的预先收集数据集，涵盖了离线RL和离线MARL，配有一个强大的框架，以促进进一步的研究。这些数据源自荣耀王者，这是一种被认可的多人在线战斗竞技（MOBA）游戏，以其错综复杂的本质而闻名，与现实生活中的情况十分相似。利用这一框架，我们对各种离线RL和离线MARL算法进行了基准测试。我们还介绍了一种针对游戏固有的层级行动空间的新型基准算法。我们揭示了当前离线RL方法在处理任务复杂性、泛化和多任务学习方面的不足。

**URL**: https://nips.cc/virtual/2023/poster/73492

---

## Survival Instinct in Offline Reinforcement Learning
**Author**: Anqi Li · Dipendra Misra · Andrey Kolobov · Ching-An Cheng

**Abstract**: We present a novel observation about the behavior of offline reinforcement learning (RL) algorithms: on many benchmark datasets, offline RL can produce well-performing and safe policies even when trained with "wrong" reward labels, such as those that are zero everywhere or are negatives of the true rewards. This phenomenon cannot be easily explained by offline RL's return maximization objective. Moreover, it gives offline RL a degree of robustness that is uncharacteristic of its online RL counterparts, which are known to be sensitive to reward design. We demonstrate that this surprising robustness property is attributable to an interplay between the notion of pessimism in offline RL algorithms and certain implicit biases in common data collection practices. As we prove in this work, pessimism endows the agent with a survival instinct, i.e., an incentive to stay within the data support in the long term, while the limited and biased data coverage further constrains the set of survival policies. Formally, given a reward class -- which may not even contain the true reward -- we identify conditions on the training data distribution that enable offline RL to learn a near-optimal and safe policy from any reward within the class. We argue that the survival instinct should be taken into account when interpreting results from existing offline RL benchmarks and when creating future ones. Our empirical and theoretical results suggest a new paradigm for offline RL, whereby an agent is "nudged" to learn a desirable behavior with imperfect reward but purposely biased data coverage. Please visit our website https://survival-instinct.github.io for accompanied code and videos.

**Abstract(Chinese)**: 我们提出了一个关于离线强化学习（RL）算法行为的新观察：在许多基准数据集上，离线RL即使在使用“错误”的奖励标签（例如全零或真实奖励的负值）训练时，也能产生表现良好且安全的策略。这种现象不能仅通过离线RL的回报最大化目标来轻易解释。此外，这使得离线RL具有一定的鲁棒性，这与在线RL不同，后者被认为对奖励设计敏感。我们证明了这一令人惊讶的鲁棒性属性是离线RL算法中悲观主义和常见数据收集偏见之间相互作用的结果。正如我们在本文中证明的那样，悲观主义赋予了Agent生存本能，即长期内有待于支持数据中，而有限且带有偏见的数据覆盖进一步限制了生存策略的集合。在形式上，鉴于可能不包含真实奖励的奖励类别，我们确定了训练数据分布上的条件，使得离线RL能够从类别内的任何奖励学习接近最佳和安全的策略。我们认为在解释现有离线RL基准结果并创建未来基准时，应该考虑生存本能。我们的实证和理论结果表明了离线RL的一个新范式，Agent 被“引导”以学习具有不完美奖励但故意偏见的数据覆盖的可取行为。请访问我们的网站https://survival-instinct.github.io 获取附带代码和视频。

**URL**: https://nips.cc/virtual/2023/poster/70254

---

## Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models
**Author**: Siyan Zhao · Aditya Grover

**Abstract**: Reinforcement learning presents an attractive paradigm to reason about several distinct aspects of sequential decision making, such as specifying complex goals, planning future observations and actions, and critiquing their utilities. However, the combined integration of these capabilities poses competing algorithmic challenges in retaining maximal expressivity while allowing for flexibility in modeling choices for efficient learning and inference. We present Decision Stacks, a generative framework that decomposes goal-conditioned policy agents into 3 generative modules. These modules simulate the temporal evolution of observations, rewards, and actions via independent generative models that can be learned in parallel via teacher forcing. Our framework guarantees both expressivity and flexibility in designing individual modules to account for key factors such as architectural bias, optimization objective and dynamics, transferrability across domains, and inference speed. Our empirical results demonstrate the effectiveness of Decision Stacks for offline policy optimization for several MDP and POMDP environments, outperforming existing methods and enabling flexible generative decision making.

**Abstract(Chinese)**: 强化学习提供了一种有吸引力的范例，用于推理关于顺序决策制定的几个不同方面，例如指定复杂目标、规划未来观察和行动，并批判它们的效用。然而，这些功能的组合集成在保留最大表现力的同时允许对有效学习和推理建模选择的灵活性方面提出了竞争算法挑战。我们提出了决策堆栈（Decision Stacks），这是一个将有目标条件策略代理分解为3个生成模块的框架。这些模块通过独立的生成模型模拟观察、奖励和行动的时间演变，可以通过教师强制并行学习。我们的框架保证了在设计单独模块以解释关键因素时既具有表现力又具有灵活性，这些因素包括体系结构偏差、优化目标和动力学、跨领域可转移性以及推理速度。我们的实证结果证明了决策堆栈对于离线策略优化在几种MDP和POMDP环境中的有效性，胜过现有方法并实现了灵活的生成决策制定。

**URL**: https://nips.cc/virtual/2023/poster/69911

---

## Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning
**Author**: Sotetsu Koyamada · Shinri Okano · Soichiro Nishimori · Yu Murata · Keigo Habara · Haruka Kita · Shin Ishii

**Abstract**: We propose Pgx, a suite of board game reinforcement learning (RL) environments written in JAX and optimized for GPU/TPU accelerators. By leveraging JAX's auto-vectorization and parallelization over accelerators, Pgx can efficiently scale to thousands of simultaneous simulations over accelerators. In our experiments on a DGX-A100 workstation, we discovered that Pgx can simulate RL environments 10-100x faster than existing implementations available in Python. Pgx includes RL environments commonly used as benchmarks in RL research, such as backgammon, chess, shogi, and Go. Additionally, Pgx offers miniature game sets and baseline models to facilitate rapid research cycles. We demonstrate the efficient training of the Gumbel AlphaZero algorithm with Pgx environments. Overall, Pgx provides high-performance environment simulators for researchers to accelerate their RL experiments. Pgx is available at https://github.com/sotetsuk/pgx.

**Abstract(Chinese)**: 我们提出了Pgx，这是一套使用JAX编写的并针对GPU/TPU加速器进行了优化的棋盘游戏强化学习（RL）环境。通过利用JAX的自动矢量化和加速器上的并行化，Pgx 可以有效地扩展到数千个加速器上的同时模拟。在我们在DGX-A100工作站上的实验中，我们发现Pgx可以比现有的Python实现快10-100倍地模拟RL环境。Pgx 包括在RL研究中常用作基准测试的RL环境，例如双陆棋、国际象棋、将棋和围棋。此外，Pgx还提供了迷你游戏集和基准模型，以促进快速研究周期。我们展示了在Pgx环境中高效训练Gumbel AlphaZero算法。总的来说，Pgx为研究人员提供了高性能的环境模拟器，以加速他们的RL实验。Pgx 可以在 https://github.com/sotetsuk/pgx 上获得。

**URL**: https://nips.cc/virtual/2023/poster/73576

---

## A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning
**Author**: Florian Felten · Lucas N. Alegre · Ann Nowe · Ana Bazzan · El Ghazali Talbi · Grégoire Danoy · Bruno C. da Silva

**Abstract**: Multi-objective reinforcement learning algorithms (MORL) extend standard reinforcement learning (RL) to scenarios where agents must optimize multiple---potentially conflicting---objectives, each represented by a distinct reward function. To facilitate and accelerate research and benchmarking in multi-objective RL problems, we introduce a comprehensive collection of software libraries that includes: (i) MO-Gymnasium, an easy-to-use and flexible API enabling the rapid construction of novel MORL environments. It also includes more than 20 environments under this API. This allows researchers to effortlessly evaluate any algorithms on any existing domains; (ii) MORL-Baselines, a collection of reliable and efficient implementations of state-of-the-art MORL algorithms, designed to provide a solid foundation for advancing research. Notably, all algorithms are inherently compatible with MO-Gymnasium; and(iii) a thorough and robust set of benchmark results and comparisons of MORL-Baselines algorithms, tested across various challenging MO-Gymnasium environments. These benchmarks were constructed to serve as guidelines for the research community, underscoring the properties, advantages, and limitations of each particular state-of-the-art method.

**Abstract(Chinese)**: 多目标强化学习算法（MORL）扩展了标准的强化学习（RL），用于代理必须优化多个——可能相互冲突的——目标的情景，每个目标由不同的奖励函数表示。为了便于和加速多目标RL问题的研究和基准测试，我们引入了一个全面的软件库集合，其中包括：（i）MO-Gymnasium，一个易于使用和灵活的API，可以快速构建新的MORL环境。它还包括超过20个在此API下的环境。这允许研究人员轻松评估任何算法在任何现有领域上的表现；（ii）MORL-Baselines，可靠高效的最新MORL算法实现的集合，旨在为推动研究提供坚实的基础。值得注意的是，所有算法与MO-Gymnasium本质上是兼容的；和（iii）一套严谨而强大的基准测试结果和MORL-Baselines算法的比较，在各种具有挑战性的MO-Gymnasium环境中进行了测试。这些基准测试旨在为研究社区提供指南，突出每种最新方法的特性、优势和局限性。

**URL**: https://nips.cc/virtual/2023/poster/73489

---

## Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning
**Author**: Hongyu Zang · Xin Li · Leiji Zhang · Yang Liu · Baigui Sun · Riashat Islam · Riashat Islam · Remi Tachet des Combes · Romain Laroche

**Abstract**: While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks,  their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to understand why bisimulation methods succeed in online settings, but falter in offline tasks. Our analysis reveals that missing transitions in the dataset are particularly harmful to the bisimulation principle, leading to ineffective estimation. We also shed light on the critical role of reward scaling in bounding the scale of bisimulation measurements and of the value error they induce. Based on these findings, we propose to apply the expectile operator for representation learning to our offline RL setting, which helps to prevent overfitting to incomplete data. Meanwhile, by introducing an appropriate reward scaling strategy, we avoid the risk of feature collapse in representation space. We implement these recommendations on two state-of-the-art bisimulation-based algorithms, MICo and SimSR, and demonstrate performance gains on two benchmark suites: D4RL and Visual D4RL. Codes are provided at \url{https://github.com/zanghyu/Offline_Bisimulation}.

**Abstract(Chinese)**: 基于双模拟的方法在学习强化学习（RL）任务的稳健状态表示方面表现出了潜力，但在离线RL任务中它们的有效性并不如人意。在某些情况下，它们的性能甚至明显地低于替代方法。我们的目标是了解为什么双模拟方法在在线环境中取得成功，而在离线任务中却失败。我们的分析显示，数据集中缺失的转换对双模拟原则特别有害，导致估计失效。我们还阐明了奖励缩放在限制双模拟测量尺度和诱发值误差方面的关键作用。基于这些发现，我们建议在我们的离线RL设置中应用期望操作符来进行表示学习，以帮助防止对不完整数据的过度拟合。同时，通过引入适当的奖励缩放策略，我们避免了表示空间中特征的崩溃风险。我们将这些建议应用于两种最先进的基于双模拟的算法MICo和SimSR，并在两个基准测试套件D4RL和Visual D4RL上展示了性能的提升。代码提供在\url{https://github.com/zanghyu/Offline_Bisimulation}。

**URL**: https://nips.cc/virtual/2023/poster/70269

---

## Self-Supervised Reinforcement Learning that Transfers using Random Features
**Author**: Boyuan Chen · Chuning Zhu · Pulkit Agrawal · Kaiqing Zhang · Abhishek Gupta

**Abstract**: Model-free reinforcement learning algorithms have exhibited great potential in solving single-task sequential decision-making problems with high-dimensional observations and long horizons, but are known to be hard to generalize across tasks. Model-based RL, on the other hand, learns task-agnostic models of the world that naturally enables transfer across different reward functions, but struggles to scale to complex environments due to the compounding error. To get the best of both worlds, we propose a self-supervised reinforcement learning method that enables the transfer of behaviors across tasks with different rewards, while circumventing the challenges of model-based RL. In particular, we show self-supervised pre-training of model-free reinforcement learning with a number of random features as rewards allows implicit modeling of long-horizon environment dynamics. Then, planning techniques like model-predictive control using these implicit models enable fast adaptation to problems with new reward functions. Our method is self-supervised in that it can be trained on offline datasets without reward labels, but can then be quickly deployed on new tasks. We validate that our proposed method enables transfer across tasks on a variety of manipulation and locomotion domains in simulation, opening the door to generalist decision-making agents.

**Abstract(Chinese)**: 无模型强化学习算法在解决具有高维观测和长期视野的单一任务顺序决策问题方面表现出巨大潜力，但已知在跨任务上很难进行泛化。另一方面，基于模型的强化学习学习世界的任务不可知模型，自然地实现了在不同奖励函数之间的转移，但由于复利误差的存在而难以扩展到复杂环境。为了兼顾两者的优势，我们提出了一种自监督强化学习方法，该方法能够实现在具有不同奖励的任务之间传递行为，并规避了基于模型的强化学习的挑战。具体来说，我们展示了无监督预训练的无模型强化学习与一些随机特征作为奖励，可以隐式建模长期视野环境动态。然后，利用这些隐式模型的规划技术，如模型预测控制，能够快速适应具有新奖励函数的问题。我们的方法是自监督的，因为它可以在没有奖励标签的离线数据集上进行训练，但随后可以快速部署到新任务上。我们验证了我们提出的方法在模拟中在各种操纵和运动领域之间实现了任务转移，从而为通用决策代理打开了大门。

**URL**: https://nips.cc/virtual/2023/poster/70151

---

## Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback
**Author**: Jangwon Kim · Hangyeol Kim · Jiwook Kang · Jongchan Baek · Soohee Han

**Abstract**: We present a novel actor-critic algorithm for an environment with delayed feedback, which addresses the state-space explosion problem of conventional approaches. Conventional approaches use an augmented state constructed from the last observed state and actions executed since visiting the last observed state. Using the augmented state space, the correct Markov decision process for delayed environments can be constructed; however, this causes the state space to explode as the number of delayed timesteps increases, leading to slow convergence. Our proposed algorithm, called Belief-Projection-Based Q-learning (BPQL), addresses the state-space explosion problem by evaluating the values of the critic for which the input state size is equal to the original state-space size rather than that of the augmented one. We compare BPQL to traditional approaches in continuous control tasks and demonstrate that it significantly outperforms other algorithms in terms of asymptotic performance and sample efficiency. We also show that BPQL solves long-delayed environments, which conventional approaches are unable to do.

**Abstract(Chinese)**: 我们提出了一种新的演员-评论家算法，用于延迟反馈环境，该算法解决了常规方法中状态空间爆炸问题。常规方法使用一个增强状态，该状态由最后观察到的状态和自上次观察到的状态以来执行的动作构建而成。使用增强状态空间，可以构建延迟环境的正确马尔可夫决策过程；然而，随着延迟时间步数的增加，状态空间会爆炸，达到缓慢收敛。我们提出的算法名为基于信念投影的Q学习（BPQL），通过评估评论家的值来解决状态空间爆炸问题，其中输入状态大小等于原始状态空间大小，而不是增强状态空间的大小。我们将BPQL与传统方法在连续控制任务中进行比较，并证明在渐近性能和样本效率方面，它明显优于其他算法。我们还表明BPQL解决了长延迟环境，而常规方法无法做到。

**URL**: https://nips.cc/virtual/2023/poster/70249

---

## RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization
**Author**: Zhecheng Yuan · Sizhe Yang · Pu Hua · Can Chang · Kaizhe Hu · Huazhe Xu

**Abstract**: Visual Reinforcement Learning (Visual RL), coupled with high-dimensional observations, has consistently confronted the long-standing challenge of out-of-distribution generalization. Despite the focus on algorithms aimed at resolving visual generalization problems, we argue that the devil is in the existing benchmarks as they are restricted to isolated tasks and generalization categories, undermining a comprehensive evaluation of agents' visual generalization capabilities. To bridge this gap, we introduce RL-ViGen: a novel Reinforcement Learning Benchmark for Visual Generalization, which contains diverse tasks and a wide spectrum of generalization types, thereby facilitating the derivation of more reliable conclusions. Furthermore, RL-ViGen incorporates the latest generalization visual RL algorithms into a unified framework, under which the experiment results indicate that no single existing algorithm has prevailed universally across tasks. Our aspiration is that Rl-ViGen will serve as a catalyst in this area, and lay a foundation for the future creation of universal visual generalization RL agents suitable for real-world scenarios.  Access to our code and implemented algorithms is provided at https://gemcollector.github.io/RL-ViGen/.

**Abstract(Chinese)**: 视觉强化学习（Visual RL）结合高维观察一直面临着跨分布泛化的长期挑战。尽管关注于旨在解决视觉泛化问题的算法，但我们认为问题出在现有的基准测试上，因为它们仅限于孤立任务和泛化类别，从而削弱了对代理的视觉泛化能力的全面评估。为了弥补这一差距，我们引入了RL-ViGen：一种新颖的用于视觉泛化的强化学习基准，其中包含各种任务和广泛的泛化类型，从而有助于得出更可靠的结论。此外，RL-ViGen将最新的泛化视觉RL算法融入统一框架，实验结果表明，没有任何一种现有算法能够在所有任务中普遍胜出。我们希望RL-ViGen将成为该领域的催化剂，并为未来创建适用于真实场景的通用视觉泛化RL代理奠定基础。我们提供了我们的代码和实现算法的访问链接：https://gemcollector.github.io/RL-ViGen/。

**URL**: https://nips.cc/virtual/2023/poster/73592

---

## Small batch deep reinforcement learning
**Author**: Johan Obando Ceron · Marc Bellemare · Pablo Samuel Castro

**Abstract**: In value-based deep reinforcement learning with replay memories, the batch size parameter specifies how many transitions to sample for each gradient update. Although critical to the learning process, this value is typically not adjusted when proposing new algorithms. In this work we present a broad empirical study that suggests reducing the batch size can result in a number of significant performance gains; this is surprising, as the general tendency when training neural networks is towards larger batch sizes for improved performance. We complement our experimental findings with a set of empirical analyses towards better understanding this phenomenon.

**Abstract(Chinese)**: 在基于价值观的深度强化学习中，使用回放记忆，批量大小参数指定每个梯度更新要采样多少个转换。尽管对学习过程至关重要，但在提出新算法时通常不调整此值。在这项工作中，我们提出了一项广泛的实证研究，表明减小批量大小可能带来许多显著的性能提升；这令人惊讶，因为在训练神经网络时一般倾向于使用更大的批量大小以提高性能。我们将实验发现与一系列实证分析相结合，以更好地理解这一现象。

**URL**: https://nips.cc/virtual/2023/poster/70060

---

## Pitfall of Optimism: Distributional Reinforcement Learning by Randomizing Risk Criterion
**Author**: Taehyun Cho · Seungyub Han · Heesoo Lee · Kyungjae Lee · Jungwoo Lee

**Abstract**: Distributional reinforcement learning algorithms have attempted to utilize estimated uncertainty for exploration, such as optimism in the face of uncertainty. However, using the estimated variance for optimistic exploration may cause biased data collection and hinder convergence or performance. In this paper, we present a novel distributional reinforcement learning that selects actions by randomizing risk criterion without losing the risk-neutral objective. We provide a perturbed distributional Bellman optimality operator by distorting the risk measure. Also,we prove the convergence and optimality of the proposed method with the weaker contraction property. Our theoretical results support that the proposed method does not fall into biased exploration and is guaranteed to converge to an optimal return. Finally, we empirically show that our method outperforms other existing distribution-based algorithms in various environments including Atari 55 games.

**Abstract(Chinese)**: 分布式强化学习算法试图利用估计的不确定性来进行探索，比如在不确定性面前保持乐观态度。然而，利用估计的方差进行乐观探索可能导致偏倚的数据收集并妨碍收敛或性能。本文提出了一种新颖的分布式强化学习方法，通过随机化风险标准来选择行动，而不失风险中性目标。我们通过扭曲风险度量来提供一个扰动的分布式贝尔曼最优算子。而且，我们证明了所提方法的收敛性和最优性，具有较弱的收缩特性。我们的理论结果支持所提方法不会陷入偏倚探索，并保证收敛到最优回报。最后，我们在包括Atari 55游戏在内的各种环境中经验性地表明，我们的方法在性能上优于其他现有的基于分布的算法。

**URL**: https://nips.cc/virtual/2023/poster/70117

---

## CQM: Curriculum Reinforcement Learning with a Quantized World Model
**Author**: Seungjae Lee · Daesol Cho · Jonghae Park · H. Jin Kim

**Abstract**: Recent curriculum Reinforcement Learning (RL) has shown notable progress in solving complex tasks by proposing sequences of surrogate tasks. However, the previous approaches often face challenges when they generate curriculum goals in a high-dimensional space. Thus, they usually rely on manually specified goal spaces. To alleviate this limitation and improve the scalability of the curriculum, we propose a novel curriculum method that automatically defines the semantic goal space which contains vital information for the curriculum process, and suggests curriculum goals over it. To define the semantic goal space, our method discretizes continuous observations via vector quantized-variational autoencoders (VQ-VAE) and restores the temporal relations between the discretized observations by a graph. Concurrently, ours suggests uncertainty and temporal distance-aware curriculum goals that converges to the final goals over the automatically composed goal space. We demonstrate that the proposed method allows efficient explorations in an uninformed environment with raw goal examples only. Also, ours outperforms the state-of-the-art curriculum RL methods on data efficiency and performance, in various goal-reaching tasks even with ego-centric visual inputs.

**Abstract(Chinese)**: 最近的课程强化学习（RL）在提出替代任务序列的解决方案方面取得了显著进展。然而，先前的方法在生成高维空间中的课程目标时经常面临挑战。因此，它们通常依赖于手动指定的目标空间。为了减轻这一限制并提高课程的可扩展性，我们提出了一种新颖的课程方法，自动定义语义目标空间，其中包含课程过程的重要信息，并在其上建议课程目标。为了定义语义目标空间，我们的方法通过矢量量化变分自动编码器（VQ-VAE）离散化连续观察结果，并通过图形恢复离散观察结果之间的时间关系。与此同时，我们提出了一个不确定性和时间距离感知的课程目标，使其在自动组成的目标空间中收敛到最终目标。我们证明了所提出的方法允许在只有原始目标示例的未知环境中进行高效的探索。此外，我们在各种目标达成任务中，即使只有以自我为中心的视觉输入，也比现有技术的课程强化学习方法在数据效率和性能方面表现优越。

**URL**: https://nips.cc/virtual/2023/poster/70196

---

## Gigastep - One Billion Steps per Second Multi-agent Reinforcement Learning
**Author**: Mathias Lechner · lianhao yin · Tim Seyde · Tsun-Hsuan Johnson Wang · Wei Xiao · Ramin Hasani · Joshua Rountree · Daniela Rus

**Abstract**: Multi-agent reinforcement learning (MARL) research is faced with a trade-off: it either uses complex environments requiring large compute resources, which makes it inaccessible to researchers with limited resources, or relies on simpler dynamics for faster execution, which makes the transferability of the results to more realistic tasks challenging. Motivated by these challenges, we present Gigastep, a fully vectorizable, MARL environment implemented in JAX, capable of executing up to one billion environment steps per second on consumer-grade hardware. Its design allows for comprehensive MARL experimentation, including a complex, high-dimensional space defined by 3D dynamics, stochasticity, and partial observations. Gigastep supports both collaborative and adversarial tasks, continuous and discrete action spaces, and provides RGB image and feature vector observations, allowing the evaluation of a wide range of MARL algorithms. We validate Gigastep's usability through an extensive set of experiments, underscoring its role in widening participation and promoting inclusivity in the MARL research community.

**Abstract(Chinese)**: 多智能体强化学习（MARL）研究面临一个折衷：要么使用需要大量计算资源的复杂环境，这使得资源有限的研究人员难以接触，要么依赖更简单的动态以加快执行速度，这使得结果的可转移性到更现实的任务变得具有挑战性。受到这些挑战的启发，我们提出了Gigastep，这是一个在JAX中实现的全矢量化MARL环境，能够在消费级硬件上每秒执行高达十亿个环境步骤。它的设计允许进行全面的MARL实验，包括由3D动态、随机性和部分观测定义的复杂高维空间。Gigastep支持合作和对抗性任务，连续和离散动作空间，并提供RGB图像和特征向量观测，可评估各种MARL算法。我们通过一系列广泛的实验验证了Gigastep的可用性，凸显了其在扩大参与和促进MARL研究社区包容性方面的作用。

**URL**: https://nips.cc/virtual/2023/poster/73577

---

## OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment
**Author**: Yiheng Zhu · Yang Zhan · Xuankun Huang · Yuwei Chen · yujie Chen · Jiangwen Wei · Wei Feng · Yinzhi Zhou · Haoyuan Hu · Jieping Ye

**Abstract**: The dramatic growth of global e-commerce has led to a surge in demand for efficient and cost-effective order fulfillment which can increase customers' service levels and sellers' competitiveness. However, managing order fulfillment is challenging due to a series of interdependent online sequential decision-making problems. To clear this hurdle, rather than solving the problems separately as attempted in some recent researches, this paper proposes a method based on multi-agent reinforcement learning to integratively solve the series of interconnected problems, encompassing order handling, packing and pickup, storage, order consolidation, and last-mile delivery. In particular, we model the integrated problem as a Markov game, wherein a team of agents learns a joint policy via interacting with a simulated environment. Since no simulated environment supporting the complete order fulfillment problem exists, we devise Order Fulfillment COoperative mUlti-agent Reinforcement learning Scalable Environment (OFCOURSE) in the OpenAI Gym style, which allows reproduction and re-utilization to build customized applications. By constructing the fulfillment system in OFCOURSE, we optimize a joint policy that solves the integrated problem, facilitating sequential order-wise operations across all fulfillment units and minimizing the total cost of fulfilling all orders within the promised time. With OFCOURSE, we also demonstrate that the joint policy learned by multi-agent reinforcement learning outperforms the combination of locally optimal policies. The source code of OFCOURSE is available at: https://github.com/GitYiheng/ofcourse.

**Abstract(Chinese)**: 全球电子商务的戏剧性增长导致对高效和具有成本效益的订单履行需求急剧增加，这可以提高客户服务水平和卖家的竞争力。然而，由于一系列相互依存的在线顺序决策问题，订单履行的管理具有挑战性。为了克服这一障碍，本文提出了一种基于多智能体强化学习的方法，以整合地解决一系列相互连接的问题，包括订单处理、打包和取件、存储、订单合并和末端交付。具体来说，我们将集成问题建模为马尔可夫博弈，在这个博弈中，一组智能体通过与模拟环境的交互学习联合策略。由于目前没有支持完整订单履行问题的模拟环境，我们设计了OFCOURSE（Order Fulfillment Cooperative Multi-agent Reinforcement Learning Scalable Environment），它采用OpenAI Gym的风格，允许复制和重复利用以构建定制应用程序。通过在OFCOURSE中构建履行系统，我们优化了一个解决整合问题的联合策略，促进了跨所有履行单元的顺序操作，并最大程度地降低了在承诺时间内完成所有订单的总成本。通过OFCOURSE，我们还证明了由多智能体强化学习学习的联合策略优于局部最优策略的组合。OFCOURSE的源代码可在以下位置找到：https://github.com/GitYiheng/ofcourse

**URL**: https://nips.cc/virtual/2023/poster/73723

---

## On Sample-Efficient Offline Reinforcement Learning: Data Diversity, Posterior Sampling and Beyond
**Author**: Thanh Nguyen-Tang · Raman Arora

**Abstract**: We seek to understand what facilitates sample-efficient learning from historical datasets for sequential decision-making, a problem that is popularly known as offline reinforcement learning (RL). Further, we are interested in algorithms that enjoy sample efficiency while leveraging (value) function approximation. In this paper, we address these fundamental questions by (i) proposing a notion of data diversity that subsumes the previous notions of coverage measures in offline RL and (ii) using this notion to \emph{unify} three distinct classes of offline RL algorithms based on version spaces (VS), regularized optimization (RO), and posterior sampling (PS). We establish that VS-based, RO-based, and PS-based algorithms, under standard assumptions, achieve \emph{comparable} sample efficiency, which recovers the state-of-the-art sub-optimality bounds for finite and linear model classes with the standard assumptions. This result is surprising, given that the prior work suggested an unfavorable sample complexity of the RO-based algorithm compared to the VS-based algorithm, whereas posterior sampling is rarely considered in offline RL due to its explorative nature. Notably, our proposed model-free PS-based algorithm for offline RL is \emph{novel}, with sub-optimality bounds that are \emph{frequentist} (i.e., worst-case) in nature.

**Abstract(Chinese)**: 我们寻求理解什么促进了从历史数据集中对于顺序决策制定的样本高效学习，这个问题广泛被称为离线强化学习（RL）。此外，我们对享有样本效率且利用（值）函数逼近的算法感兴趣。在本文中，我们通过（i）提出了一个数据多样性的概念，它包含了离线RL中以前的覆盖度量的概念，以及（ii）利用这个概念来\emph{统一}基于版本空间（VS）、正则化优化（RO）和后验抽样（PS）的三类离线RL算法。我们发现，在标准假设下，基于VS、基于RO和基于PS的算法实现了\emph{可比较}的样本效率，这一发现恢复了具有标准假设下有限和线性模型类别的状态下优化次优性边界的现有技术。这一结果令人惊讶，因为先前的研究表明，在标准假设下，基于RO的算法的样本复杂性不如基于VS的算法，而后验抽样在离线RL中很少考虑，因为其具有探索性质。值得注意的是，我们提出的面向无模型的离线RL的PS算法是\emph{新颖}的，其次优性边界是\emph{频率主义}（即，最坏情况）性质的。

**URL**: https://nips.cc/virtual/2023/poster/70257

---

## On the Importance of Exploration for Generalization in Reinforcement Learning
**Author**: Yiding Jiang · J. Zico Kolter · Roberta Raileanu

**Abstract**: Existing approaches for improving generalization in deep reinforcement learning (RL) have mostly focused on representation learning, neglecting RL-specific aspects such as exploration. We hypothesize that the agent's exploration strategy plays a key role in its ability to generalize to new environments.Through a series of experiments in a tabular contextual MDP, we show that exploration is helpful not only for efficiently finding the optimal policy for the training environments but also for acquiring knowledge that helps decision making in unseen environments. Based on these observations, we propose EDE: Exploration via Distributional Ensemble, a method that encourages the exploration of states with high epistemic uncertainty through an ensemble of Q-value distributions. The proposed algorithm is the first value-based approach to achieve strong performance on both Procgen and Crafter, two benchmarks for generalization in RL with high-dimensional observations. The open-sourced implementation can be found at https://github.com/facebookresearch/ede.

**Abstract(Chinese)**: 深度强化学习（RL）中改善泛化的现有方法大多集中在表示学习上，忽视了RL特有的探索等方面。我们假设代理的探索策略在其泛化到新环境的能力中起着关键作用。通过在表格化情境MDP中进行一系列实验，我们展示了探索不仅有助于有效地找到训练环境的最优策略，而且还有助于获取知识，帮助在未知环境中做出决策。基于这些观察结果，我们提出了EDE: 通过分布集成进行探索，这是一种方法，它通过Q值分布的集成鼓励对具有高认知不确定性的状态进行探索。所提出的算法是第一个基于价值的方法，可以在高维观察下实现Procgen和Crafter两个RL泛化基准的强大性能。开源实现可在https://github.com/facebookresearch/ede 上找到。

**URL**: https://nips.cc/virtual/2023/poster/69969

---

## SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning
**Author**: Benjamin Ellis · Jonathan Cook · Skander Moalla · Mikayel Samvelyan · Mingfei Sun · Anuj Mahajan · Jakob Foerster · Shimon Whiteson

**Abstract**: The availability of challenging benchmarks has played a key role in the recent progress of machine learning. In cooperative multi-agent reinforcement learning, the StarCraft Multi-Agent Challenge (SMAC) has become a popular testbed for centralised training with decentralised execution. However, after years of sustained improvement on SMAC, algorithms now achieve near-perfect performance. In this work, we conduct new analysis demonstrating that SMAC lacks the stochasticity and partial observability to require complex closed-loop policies. In particular, we show that an open-loop policy conditioned only on the timestep can achieve non-trivial win rates for many SMAC scenarios. To address this limitation, we introduce SMACv2, a new version of the benchmark where scenarios are procedurally generated and require agents to generalise to previously unseen settings (from the same distribution) during evaluation. We also introduce the extended partial observability challenge (EPO), which augments SMACv2 to ensure meaningful partial observability. We show that these changes ensure the benchmarkrequires the use of closed-loop policies. We evaluate state-of-the-art algorithms on SMACv2 and show that it presents significant challenges not present in the original benchmark.  Our analysis illustrates that SMACv2 addresses the discovered deficiencies of SMAC and can help benchmark the next generation of MARL methods. Videos of training are available on our website.

**Abstract(Chinese)**: 挑战性基准测试的可用性在机器学习的最新进展中发挥了关键作用。 在合作多智体强化学习中，星际争霸多智体挑战（SMAC）已成为中央化训练和分散执行的热门试验台。 但是，在多年对SMAC的持续改进之后，算法现在实现了接近完美的性能。 在这项工作中，我们进行了新的分析，表明SMAC缺乏需要复杂闭环策略的随机性和部分可观测性。 特别是，我们表明仅在时间步上有条件的开环策略可以在许多SMAC场景中实现非平凡的胜率。 为了解决这个局限性，我们引入了SMACv2，这是基准测试的新版本，在这个版本中，场景是以程序方式生成的，在评估过程中需要智能体适应以前未见过的设置（具有相同分布）。 我们还介绍了扩展的部分可观测性挑战（EPO），它增加了SMACv2以确保有意义的部分可观测性。 我们表明，这些改变确保了基准测试需要使用闭环策略。 我们评估了SMACv2上的最新算法，并表明它提出了与原始基准测试中不存在的重大挑战。 我们的分析说明了SMACv2已解决了SMAC的发现的缺陷，并可以帮助评估下一代多智体强化学习方法。 我们的网站上提供了训练视频。

**URL**: https://nips.cc/virtual/2023/poster/73695

---

## Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms
**Author**: Akifumi Wachi · Wataru Hashimoto · Xun Shen · Kazumune Hashimoto

**Abstract**: Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.

**Abstract(Chinese)**: 安全探索对于在许多现实场景中实际应用强化学习（RL）至关重要。在本文中，我们将广义安全探索（GSE）问题作为常见安全探索问题的统一表述。然后，我们提出了GSE问题的解决方案，即安全探索的元算法MASE，它将无约束的RL算法与不确定性量化器结合起来，在当前情节中保证安全并在实际安全违规前适当地惩罚不安全的探索，以防止它们在未来情节中发生。MASE的优势在于，我们可以在高概率下优化政策，以保证不违反任何安全约束的前提。具体而言，我们提出了两种MASE的变体，其不确定性量化器采用不同的构造方式：一种基于广义线性模型，具有安全性和接近最优性的理论保证，另一种结合了高斯过程和深度RL算法，以确保安全并最大化奖励。最后，我们证明了我们提出的算法在网格世界和Safety Gym基准测试上比最先进的算法表现更好，且在训练过程中没有违反任何安全约束。

**URL**: https://nips.cc/virtual/2023/poster/71024

---

## Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints
**Author**: Dohyeong Kim · Kyungjae Lee · Songhwai Oh

**Abstract**: In safety-critical robotic tasks, potential failures must be reduced, and multiple constraints must be met, such as avoiding collisions, limiting energy consumption, and maintaining balance.Thus, applying safe reinforcement learning (RL) in such robotic tasks requires to handle multiple constraints and use risk-averse constraints rather than risk-neutral constraints.To this end, we propose a trust region-based safe RL algorithm for multiple constraints called a safe distributional actor-critic (SDAC).Our main contributions are as follows: 1) introducing a gradient integration method to manage infeasibility issues in multi-constrained problems, ensuring theoretical convergence, and 2) developing a TD($\lambda$) target distribution to estimate risk-averse constraints with low biases. We evaluate SDAC through extensive experiments involving multi- and single-constrained robotic tasks.While maintaining high scores, SDAC shows 1.93 times fewer steps to satisfy all constraints in multi-constrained tasks and 1.78 times fewer constraint violations in single-constrained tasks compared to safe RL baselines.Code is available at: https://github.com/rllab-snu/Safe-Distributional-Actor-Critic.

**Abstract(Chinese)**: 在安全关键的机器人任务中，必须减少潜在故障，并满足多个约束，例如避免碰撞、限制能量消耗和保持平衡。因此，在这些机器人任务中应用安全的强化学习（RL）需要处理多个约束，并使用风险回避约束而不是风险中性约束。为此，我们提出了一种基于信任域的安全RL算法，用于多个约束，称为安全分布式演员-评论家（SDAC）。我们的主要贡献如下：1）引入了一种梯度整合方法来处理多约束问题中的不可行性问题，确保理论收敛，2）开发了一个TD（$\lambda$）目标分布来估计低偏差的风险回避约束。我们通过涉及多个和单个约束的机器人任务的广泛实验来评估SDAC。在保持高得分的同时，与安全RL基线相比，SDAC在多约束任务中满足所有约束的步数少了1.93倍，在单一约束任务中违反约束的次数减少了1.78倍。代码可在以下位置获得：https://github.com/rllab-snu/Safe-Distributional-Actor-Critic。

**URL**: https://nips.cc/virtual/2023/poster/70373

---

## Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation
**Author**: Ruida Zhou · Tao Liu · Min Cheng · Dileep Kalathil · P. R. Kumar · Chao Tian

**Abstract**: We study robust reinforcement learning (RL) with the goal of determining a well-performing policy that is robust against model mismatch between the training simulator and the testing environment. Previous policy-based robust RL algorithms mainly focus on the tabular setting under uncertainty sets that facilitate robust policy evaluation, but are no longer tractable when the number of states scales up. To this end, we propose two novel uncertainty set formulations, one based on double sampling and the other on an integral probability metric. Both make large-scale robust RL tractable even when one only has access to a simulator. We propose a robust natural actor-critic (RNAC) approach that incorporates the new uncertainty sets and employs function approximation. We provide finite-time convergence guarantees for the proposed RNAC algorithm to the optimal robust policy within the function approximation error. Finally, we demonstrate the robust performance of the policy learned by our proposed RNAC approach in multiple  MuJoCo environments and a real-world TurtleBot navigation task.

**Abstract(Chinese)**: 我们研究强化学习（RL）的鲁棒性，旨在确定一个性能良好的策略，该策略对训练模拟器和测试环境之间的模型不匹配具有鲁棒性。先前基于策略的鲁棒RL算法主要侧重于在不确定性集合下开展表格式设置，以促进鲁棒策略评估，但当状态数量扩大时已不再可行。为此，我们提出了两种新的不确定性集合形式，一种基于双重采样，另一种基于积分概率度量。当一个人仅能访问模拟器时，这两种方法都使大规模鲁棒RL成为可能。我们提出了一种强大的自然actor-critic（RNAC）方法，该方法结合了新的不确定性集合，并采用函数逼近。对于提出的RNAC算法，我们提供了有限时间内收敛到最佳鲁棒策略的保证，考虑到函数逼近误差。最后，我们展示了由我们提出的RNAC方法学习的策略在多个MuJoCo环境和现实世界的TurtleBot导航任务中的鲁棒性能。

**URL**: https://nips.cc/virtual/2023/poster/70037

---

## Replicable Reinforcement Learning
**Author**: Eric Eaton · Marcel Hussing · Michael Kearns · Jessica Sorrell

**Abstract**: The replicability crisis in the social, behavioral, and data sciences has led to the formulation of algorithm frameworks for replicability --- i.e., a requirement that an algorithm produce identical outputs (with high probability) when run on two different samples from the same underlying distribution. While still in its infancy, provably replicable algorithms have been developed for many fundamental tasks in machine learning and statistics, including statistical query learning, the heavy hitters problem, and distribution testing. In this work we initiate the study of replicable reinforcement learning, providing a provably replicable algorithm for parallel value iteration, and a provably replicable version of R-Max in the episodic setting. These are the first formal replicability results for control problems, which present different challenges for replication than batch learning settings.

**Abstract(Chinese)**: 社会、行为和数据科学领域中的可复制性危机导致了针对可复制性的算法框架的制定 --- 即，要求算法在从相同潜在分布中抽取的两个不同样本上运行时产生相同输出（很高的概率）。虽然这一领域仍处于起步阶段，但已经为许多机器学习和统计学中的基本任务开发了可以证明可复制性的算法，包括统计查询学习、重点问题和分布测试。在这项工作中，我们开始研究可复制性强化学习，提供了并行价值迭代的可证复制算法，并在情景设置中提供了R-Max的可证复制版本。这是控制问题的首个正式可复制性结果，这些问题对复制提出了与批量学习设置不同的挑战。

**URL**: https://nips.cc/virtual/2023/poster/70222

---

