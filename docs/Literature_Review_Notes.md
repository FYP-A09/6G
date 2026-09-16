Tab 1
6G Network Slicing — Literature Review Notes


Paper 1: AI-Driven Digital Twins: Optimizing 5G/6G Network Slicing with NTNs
	Ali & Arslan (2025) — IEEE Wireless Communications Letters
	Problem Statement
	Static Digital Twin updates can't keep pace with the rapid topology changes caused by UAV-mounted flying base stations during disaster recovery or urban blockage events, leading to latency spikes for NTN-integrated eMBB slices.
	Methodology / Algorithm
	A real-time DT synchronization scheme continuously tracks the spatial distance between mobile users and the flying base station, feeding a continuous-action DDPG agent that allocates bandwidth. The state space folds in path loss, Rayleigh fading and satellite interference.
	Dataset / Simulation Tool
	Custom Python-based 5G simulator with 50 UEs over 20 MHz bandwidth.
	Results
	~25% latency reduction vs static allocation (avg. 33 ms), jitter down to 2.1 ms, ~92% resource utilization.
	Research Gap
	Assumes clean, noise-free telemetry and has no predictive topology mapping — it reacts to raw signals rather than filtering anomalies through learned representations.
	How it relates to our project
	Confirms DDPG works well inside a Digital Twin loop, but its lack of representation learning is exactly the gap our SSL module is meant to close, and its purely reactive design motivates adding TGNN-based prediction.
	

Paper 2: A Flexible and Scalable Multi-Agent Learning Framework for Dynamic RAN Slicing in 6G Native-AI Networks
	Li, Wang, et al. (2026) — IEEE Transactions on Mobile Computing
	Problem Statement
	Changing slice configurations forces costly model retraining, and frequent radio-resource reconfiguration creates heavy signalling and computational overhead.
	Methodology / Algorithm
	Proposes a Statistic-Field Deep Truncated Monte Carlo (SF-DTMC) algorithm operating in a RAN Partially Observable Stochastic Game (RAN-POSG), paired with an Adaptive Retention Framework (ARF) that penalizes unnecessary state transitions to balance responsiveness against reconfiguration stability.
	Dataset / Simulation Tool
	NS-3-based high-fidelity RAN slicing simulator.
	Results
	Large drop in reconfiguration time and SLA violations compared with LLM-based agents and standard RL baselines.
	Research Gap
	Scalability is limited once you move past tightly localized cell clusters, and control decisions still rely purely on local/reactive state observations.
	How it relates to our project
	Backs our decentralized multi-agent design, but our framework goes further by adding SSL and TGNN to enrich state representation and add predictive foresight, which this paper doesn't attempt.
	







Paper 3: A Framework for AI-Native Semantic-Based Dynamic Slicing for 6G Networks
	Chowdhury et al. (2025) — TechRxiv
	Problem Statement
	Traditional architectures treat data content and network resource orchestration as separate concerns — insufficient for 6G's semantic-aware ambitions.
	Methodology / Algorithm
	Introduces a semantic slicing framework built around a horizontal "Reasoning Plane" that sits between the data and control planes, using symbolic AI, GFlowNets, and signaling games to build shared semantic languages.
	Dataset / Simulation Tool
	Python / TensorFlow.
	Results
	Faster semantic task execution and improved convergence as the system settles on a shared semantic vocabulary.
	Research Gap
	Highly theoretical — it doesn't quantify the computational latency the Reasoning Plane would add on real edge hardware.
	How it relates to our project
	Supplies the conceptual blueprint for the Reasoning Plane and data slicing, which we position as future work once the core SSL+TGNN+MARL pipeline is validated.
	

Paper 4: GAN-Enhanced Deep Reinforcement Learning for Semantic-Aware Resource Allocation in 6G Network Slicing
	Daniel Benniah John (2026) — arXiv (cs.NI)
	Problem Statement
	Discrete action quantization plus "semantic blindness" in classical resource allocation wastes an estimated 35% of bandwidth on redundant data.
	Methodology / Algorithm
	GAN-DDPG framework: conditional GANs generate synthetic semantic traffic while a continuous-action DDPG agent optimizes allocation against a semantic-aware reward instead of raw bit-rate maximization.
	Dataset / Simulation Tool
	Python / custom 6G slicing simulator.
	Results
	+22% URLLC spectral efficiency, +20% eMBB spectral efficiency, -31% packet loss, -18% latency versus baseline.
	Research Gap
	Semantic state definitions are domain-specific and don't generalize; semantic extraction itself adds computational delay at the edge.
	How it relates to our project
	Gives concrete empirical proof that reward-level semantic awareness beats classical MARL objectives — direct evidence for why we flag semantic communication as a strong future-work extension.
	

Paper 5: Multi-Agent Reinforcement Learning for Slicing Resource Allocation in Vehicular Networks
	(vehicular networks MARL study) (2023) — IEEE Transactions on Intelligent Transportation Systems
	Problem Statement
	High-speed vehicular mobility forces frequent RAN slicing changes, generating heavy reconfiguration overhead.
	Methodology / Algorithm
	Formulates the problem as a POMDP and solves it with Multi-Agent DDPG (MADDPG) to minimize long-term system cost.
	Dataset / Simulation Tool
	Python with a Mininet emulator and Ryu SDN controller.
	Results
	Lower system cost overall, with an average QoS satisfaction rate of 96.5%.
	Research Gap
	Validation stays confined to simulation (true of ~86% of surveyed studies), and coordination delays of 120–180 ms exceed URLLC latency budgets.
	How it relates to our project
	Validates multi-agent handling of mobility, but has no Digital Twin safety net for policy exploration — the gap our NDT sandbox is designed to fill.
	



Paper 1: Native Network Digital Twin Architecture for 6G: From Design to Practice (2025)


1. Problem Statement
* Existing Network Digital Twins (NDTs) lack standardized architecture, real-time synchronization, and efficient data collection for large-scale 6G networks.
* Traditional KPI-based monitoring cannot support service validation, prediction, or safe testing of network changes.
* The paper proposes a native Network Digital Twin architecture for real-time monitoring, simulation, verification, and optimization.


2. Methodology / Algorithm
* Introduces a three-layer architecture: Network Element Layer, NDT Layer, and Application Layer.
* Defines the complete NDT lifecycle: Preparation, Creation, Runtime, and Feedback.
* Proposes native high-speed data collection using kernel pass, sidecar agents, Kafka/RDMA/QUIC.
* Uses a gray-box modelling approach combining protocol knowledge with learned behaviour.


3. Dataset / Simulation Tool
* No public dataset is used.
* Validated using a real operator-inspired signalling storm scenario during disaster recovery.
* Digital Twin is built using real topology, signalling statistics, historical traffic patterns, and network behaviour models.


4. Results
* Achieved ≈3% MAPE for registration arrival rate prediction.
* Achieved ≈25% MAPE for successful registration count.
* Successfully replicated signalling storm behaviour and validated safe traffic control strategies.
* Demonstrated the effectiveness of the proposed architecture and gray-box modelling.


5. Research Gap
* Evaluation is limited to a signalling storm use case.
* No Graph Neural Networks (GNN), Temporal GNN (TGNN), Self-Supervised Learning (SSL), or Reinforcement Learning (RL) are incorporated.
* Focuses on simulation and verification rather than autonomous optimization.


6. How it Relates to Our Project
* Provides the architectural foundation for our Digital Twin framework.
* Supports real-time synchronization, modular Digital Twin lifecycle, and gray-box modelling.
* Our work extends this architecture using SSL, TGNN, and MARL for intelligent network slicing.
________________




Paper 2: Attention to Virtualization: Making Network Digital Twins Aware of Network Slicing (2025)


1. Problem Statement
* Existing Digital Twins ignore virtualization effects such as resource contention between virtual network functions.
* Conventional GNNs suffer from over-smoothing and poor long-term dependency modelling.
* The paper proposes a virtualization-aware Digital Twin using Graph Attention Networks for improved KPI prediction.


2. Methodology / Algorithm
* Measures virtualization overhead using Docker-based UPF deployments.
* Represents physical and virtual resources as a graph.
* Combines GraphSAGE + Graph Attention Network (GAT) to learn virtualization-aware node embeddings.
* Performs "what-if" analysis before slice deployment using the Digital Twin.


3. Dataset / Simulation Tool
* Experimental measurements collected from Docker-based UPFs.
* Digital Twin evaluated using an extended OMNeT++ simulator.
* Simulation includes 30 CNFs, 5 physical nodes, and GraphSAGE + GAT trained using a 70/30 split.


4. Results
* Virtualization significantly increases delay and packet loss.
* GraphSAGE + GAT outperforms conventional GNNs in KPI prediction.
* Achieves lower MAE and MSE than RouteNet-Fermi.
* Improves delay prediction across eMBB, URLLC, and mMTC slices.


5. Research Gap
* Uses a static graph and cannot model temporal traffic evolution.
* Depends on supervised learning with labelled data.
* No Reinforcement Learning for autonomous optimization.


6. How it Relates to Our Project
* Introduces virtualization-aware Digital Twins for network slicing.
* Provides the graph abstraction used in our framework
________________




Paper 3: A Graph Neural Network-Based Digital Twin for Network Slicing Management (2022)


1. Problem Statement
* Existing analytical models and conventional deep learning methods cannot effectively model graph-structured communication networks.
* Shared physical resources create complex interactions between network slices, making E2E latency prediction difficult.
* The paper proposes a GraphSAGE-based Digital Twin for accurate End-to-End latency prediction.


2. Methodology / Algorithm
* Represents physical infrastructure and network slices as graphs.
* Uses GraphSAGE with neighbour sampling and mean aggregation to generate node embeddings.
* Aggregates node embeddings into slice embeddings and predicts End-to-End latency.
* Trains using the Log-Cosh loss and Adam optimizer.


3. Dataset / Simulation Tool
* Uses three network topologies: NSFNET (14 nodes), GEANT2 (24 nodes), and Synthetic (50 nodes).
* Trains on 70% of graph nodes and evaluates on unseen nodes.
* Uses traffic matrices, VNF resource utilization, and E2E latency as input features.


4. Results
* Achieved less than 5% prediction error for End-to-End latency.
* Successfully generalized to unseen network topologies.
* Demonstrated transfer learning for jitter prediction.
* Supported SLA monitoring, link failure recovery, and deployment optimization.


5. Research Gap
* Models the network as a static graph, ignoring temporal changes.
* Requires labelled datasets due to supervised learning.
* Does not perform Reinforcement Learning-based optimization.
* Does not explicitly model virtualization effects.


6. How it Relates to Our Project
* Provides the graph-based modelling foundation for our Digital Twin.
* Our work extends GraphSAGE using Temporal Graph Neural Networks (TGNNs).
 
Paper 4: Revolutionizing Wireless Networks with Self-Supervised Learning: A Pathway to Intelligent Communications (2024)
1. Problem Statement
* Supervised learning requires large labelled datasets, which are difficult to obtain in dynamic wireless communication networks.
* Massive amounts of unlabelled traffic, channel, and mobility data remain underutilized.
* The paper investigates Self-Supervised Learning (SSL) as an alternative learning paradigm for AI-native 6G networks.
2. Methodology / Algorithm
* Reviews three SSL paradigms: Generative, Contrastive, and Predictive learning.
* Describes the SSL workflow consisting of pretext task generation, representation learning, and downstream task fine-tuning.
* Surveys SSL applications in traffic prediction, channel estimation, anomaly detection, semantic communication, and network optimization.
3. Dataset / Simulation Tool
* Survey paper; no single dataset is used.
* Reviews multiple wireless communication datasets and existing SSL-based communication systems.
* Covers applications in wireless networking, semantic communications, and edge intelligence.
4. Results
* Demonstrates that SSL effectively learns meaningful representations from unlabelled wireless data.
* Reduces dependence on labelled datasets while improving model generalization.
* Successfully supports various communication tasks, including channel estimation, anomaly detection, and intelligent network management.
5. Research Gap
* Designing efficient pretext tasks for wireless communication remains challenging.
* Existing SSL methods have high computational complexity and limited robustness in dynamic environments.
* Integration with Digital Twins, TGNNs, and Reinforcement Learning remains largely unexplored.


6. How it Relates to Our Project
* Provides the motivation for using Self-Supervised Learning in AI-native 6G networks.
* Supports learning from unlabelled Digital Twin data before TGNN-based modelling.
* Our framework integrates SSL with TGNN and MARL for intelligent dynamic network slicing.
Paper 5: Network-to-Network: Self-Supervised Network Representation Learning via Position Prediction (2025)
1. Problem Statement
* Existing Graph Neural Networks (GNNs) rely heavily on labelled data, making training expensive and limiting scalability.
* Most network representation learning methods capture structural information but fail to preserve node positional information and rich node content simultaneously.
* The paper proposes a Self-Supervised Network Representation Learning framework (Net2Net) that learns meaningful node embeddings without requiring large labelled datasets. 
2. Methodology / Algorithm
* Introduces Net2Net, a self-supervised encoder-decoder framework based on Node Position Prediction (PosPredict).
* Constructs ego networks for each node and learns embeddings by combining node content and graph topology.
* Uses a hierarchical GNN encoder and cross-modal decoder to predict each node's position (PosID), followed by semi-supervised fine-tuning for downstream tasks. 
3. Dataset / Simulation Tool
* Evaluated on eight real-world graph datasets, including Cora, Citeseer, DBLP, PubMed, GitHub, Facebook, LastFM Asia, and Twitch.
* Compared against multiple graph embedding and GNN baselines such as DeepWalk, Node2Vec, GraphSAGE, GAT, DGI, and MVGRL.
* Performance was evaluated using node classification with Micro-F1 score. 
4. Results
* Net2Net consistently achieved higher node classification accuracy than existing supervised and unsupervised methods.
* The proposed Position Prediction task produced more discriminative node embeddings by preserving structural, positional, and content information.
* Demonstrated strong generalization with minimal labelled data across diverse graph datasets. 
5. Research Gap
* The framework learns static graph representations and does not capture temporal network evolution.
* Does not address Digital Twin modelling, dynamic network slicing, or reinforcement learning-based decision making.
* Primarily designed for node representation learning rather than autonomous network optimization. 
6. How it Relates to Our Project
* Provides the Self-Supervised Learning foundation for learning graph representations from unlabelled network data.
* The learned embeddings can serve as inputs to our Temporal Graph Neural Network (TGNN) for modelling dynamic network states.
* Our project extends this work by integrating Digital Twins, TGNN, and MARL for intelligent dynamic network slicing in AI-native 6G networks.
________________


Paper 6: Encrypted Network Traffic Classification in SDN using Self-Supervised Learning (2022)
1. Problem Statement
* Encrypted network traffic is difficult to classify using conventional supervised learning due to the need for large labelled datasets.
* Preparing labelled network traffic data is costly and prone to errors.
* The paper investigates Self-Supervised Learning (SSL) for accurate traffic classification in Software Defined Networks (SDNs). 
2. Methodology / Algorithm
* Builds an SDN testbed using Mininet, Open vSwitch, and the RYU Controller to generate and collect network traffic.
* Uses a two-stage SSL framework with pre-training on unlabelled traffic followed by fine-tuning on a small labelled dataset.
* Extracts protocol-independent traffic features from flow statistics for real-time encrypted traffic classification. 
3. Dataset / Simulation Tool
* Generated a custom SDN dataset containing five encrypted traffic classes: Counter Strike (Active/Idle), Quake III, DNS, and VoIP.
* Traffic was created using Mininet and the D-ITG traffic generator, while features were collected using the RYU SDN Controller.
* Both offline evaluation and real-time traffic classification experiments were conducted. 
4. Results
* SSL achieved 97.64% classification accuracy, outperforming the supervised model by approximately 2%.
* Real-time traffic classification achieved up to 99% accuracy with an average inference time of approximately 0.002 seconds.
* Demonstrated that SSL effectively reduces label dependency while maintaining high classification performance. 
5. Research Gap
* Focuses only on encrypted traffic classification in SDN environments.
* Does not perform network representation learning, Digital Twin modelling, or dynamic network slicing.
* The SSL framework is task-specific and not designed for graph-based network optimization. 
6. How it Relates to Our Project
* Demonstrates that Self-Supervised Learning can effectively learn from large amounts of unlabelled network data.
* Provides evidence that SSL reduces dependence on manually labelled datasets, supporting the motivation for our SSL module.
* Our framework extends SSL beyond traffic classification by combining it with Digital Twins, TGNN, and MARLfor autonomous network slicing in AI-native 6G networks.




















































KRISH S
Paper 1: Toward Enabling Network Slice Mobility to Support 6G System (2022)
1. Problem Statement:
* Future 6G networks will support diverse applications with different latency, bandwidth, and resource requirements.
* When users move, the VNFs providing their services may need to be migrated, scaled, created, or relocated to maintain the required QoS.
* Relocating VNFs closer to users can reduce end-to-end latency, but frequent relocation can cause service interruption.
* Therefore, the main problem is finding a balance between low end-to-end delay and minimum service relocation while maintaining the slice SLA
.
2. Methodology :
The authors propose three Network Service Management solutions:
* DELAY-NSM: Minimizes end-to-end delay by placing VNFs closer to users through VNF instantiation, migration and resource reconfiguration.
* R-NSM: Minimizes VNF/service relocation to reduce service interruption while satisfying delay, bandwidth and jitter requirements.
* FT-NSM: Uses the Kalai-Smorodinsky bargaining game to obtain a Pareto-fair trade-off between minimizing delay and minimizing service relocation.
The framework can dynamically perform VNF scaling up/down, VNF creation/deletion, VNF migration, and UE reassignment


3.  Dataset : 
* Uses a discrete-time network simulation, not a public dataset.
* Simulates UEs, eNB/gNBs, Edge Clouds, VNFs, and network slices.
* UE mobility is modeled using random walk with pymobility.
* Tests different numbers of UEs and Edge Clouds.
* Maximum end-to-end delay is set to 30 ms.
* Each simulation is repeated 80 times for reliable results.


4. Results:
* DELAY-NSM: Lowest latency of 16–17.5 ms.
* R-NSM: Lowest relocation time of 5–7.5 ms.
* FT-NSM: Good balance — around 18 ms latency and 8–10 ms relocation time.
* FT-NSM provides the best overall trade-off, but has higher computational cost.
5. Research Gap:
* Relies mainly on optimization and game theory.
* Does not use SSL for learning network representations.
* Does not use TGNN for future traffic prediction.
* Does not use MARL for intelligent cooperative resource allocation.
* Does not use a Digital Twin for network simulation and learning.


6. How it relates to our Project:
* Directly supports our dynamic 6G network slicing concept.
* Shows the importance of dynamically scaling, migrating and relocating VNFs.
* Demonstrates the need to balance latency, QoS and resource management.
* Our project extends this using SSL + TGNN + Digital Twin + MARL for predictive and intelligent network slicing.
________________


Paper 2: Slicing for AI: An Online Learning Framework for Network Slicing Supporting AI Services (2025) 
1.Problem Statement:
* 6G will support many AI-based services with different resource and QoS requirements.
* Static/offline slicing struggles with dynamic and time-varying network conditions.
* The goal is to dynamically allocate resources while improving AI model accuracy, latency, and cost




2.Methodology :
* Proposes an Online Learning for Slicing (OLS) framework.
* Uses the EXP3 online learning algorithm to adapt resource allocation over time.
* Introduces OLS-SA and OLS-RSA to reduce the decision space and improve learning speed.
* Jointly allocates computing + communication resources and tunes AI model parameters.
3.Dataset:
* Uses a mobile-health Deep Learning dataset with 245,921 samples.
* Dataset contains 13 physical-activity classes.
* Tests different training-data sizes and numbers of epochs.
* Results are averaged over 10 experiments
4.Results:
* Proposed methods converge toward optimal resource allocation.
* OLS-SA and OLS-RSA reduce the decision space and computational complexity.
* Performs better under dynamic network conditions and varying resource availability
5.Research Gap:
* Focuses mainly on slicing resources for AI services.
* Does not use TGNN for future network-traffic prediction.
* Does not use MARL for cooperative slicing decisions.
* Does not integrate a Digital Twin or SSL-based network representation learning.
6.How It Relates to Our Project:
* Strongly supports our dynamic 6G network slicing component.
* Shows how learning can adapt resource allocation to changing network conditions.
* Our project extends this with SSL + TGNN + Digital Twin + MARL for predictive and autonomous network slicing.


________________




Paper 3: Self-Supervised Spatiotemporal Graph Neural Networks With Self-Distillation for Traffic Prediction (2023) 
1.Problem Statement : 
* Existing spatiotemporal GNNs struggle to capture wider spatial-temporal dependencies.
* Deeper GNNs can suffer from overfitting and poor generalization.
* The paper aims to improve traffic prediction accuracy and robustness.
2. Methodology:
* Proposes SSGNN, combining Self-Supervised Learning + Spatiotemporal GNN + Self-Distillation.
* SSL reconstructs masked traffic signals to learn better features.
* GCN captures spatial relationships, while temporal convolutions capture time dependencies.
* Self-distillation improves model generalization and stability.
3.Dataset:
* Uses 6 real-world traffic datasets.
* METR-LA, PEMS-BAY – traffic speed.
* PEMS03, PEMS04, PEMS07, PEMS08 – traffic flow.
* Uses 1 hour of historical data to predict the next 1 hour.
4.Results:
* SSGNN achieved better or competitive performance against strong GNN baselines.
* Maximum improvements were 3.0% MAE, 5.2% RMSE and 3.8% MAPE.
* Shows that SSL and self-distillation improve traffic forecasting.


5.Research Gap:
* Predicts road traffic, rather than 6G network traffic/slice demand.
* Prediction is the final objective; it does not use predictions for dynamic resource allocation.
* No closed-loop mechanism to convert predicted traffic into network slicing decisions.
* Therefore, it does not address proactive 6G slice orchestration.
6.How It Relates to Our Project:
* Very relevant to our SSL + TGNN prediction stage.
* Its idea of learning spatial-temporal patterns from traffic data can conceptually be adapted to 6G network nodes and traffic.
* In our project, these learned patterns help predict future network/slice demand.
* We then go further: TGNN prediction → MARL decision-making → dynamic network slicing inside the Digital Twin.
________________


Paper 4: Deep Reinforcement Learning for End-to-End Network Slicing: Challenges and Solutions (2023) 
1. Problem Statement:
* End-to-end network slicing involves many configuration parameters across RAN, transport, core, edge, and cloud.
* Traditional model-based methods struggle with such complex and dynamic resource allocation.
* The paper investigates DRL for intelligent end-to-end slice orchestration.
2.  Methodology : 
* Models resource orchestration as a Markov Decision Process (MDP).
* DRL observes network state and traffic load and decides resource allocation for each slice.
* Explores Safety DRL, Distributed/Multi-Agent DRL, and Imitation Learning.
* Proposes a three-layer architecture: Network → Orchestration → Intelligence (DRL).
3. Dataset:
* Uses a real network slicing testbed, rather than a public dataset.
* Uses OpenAirInterface/FlexRAN, OpenDayLight, OpenFlow, Docker and USRP B210.
* DRL agents are implemented using PyTorch with 3-layer neural networks.






4. Results:
* Safety DRL reduces SLA violations from up to 15% to about 1%.
* Distributed DRL improves scalability using multiple agents.
* Imitation Learning gives DRL a better starting policy and helps accelerate online learning.
* The paper shows that DRL is promising for automating end-to-end resource orchestration, while practical deployment still faces scalability and convergence challenges.
5. Research Gap:
* DRL mainly reacts to observed network states; our project adds TGNN-based future traffic prediction before slicing decisions.
* The paper identifies difficulty in handling huge heterogeneous network data; our SSL stage conceptually addresses this by learning useful representations from unlabeled network data.
* DRL learning through real-network interaction can be slow and potentially unsafe; our Digital Twin provides a virtual environment for training/testing decisions.
6. How It Relates to Our Project : 
* Very closely related to our MARL + Dynamic Network Slicing stage.
* Its distributed DRL concept uses multiple agents that coordinate resource allocation, directly supporting our choice of MARL.
* Our project adds a predictive pipeline before these agents: SSL → TGNN → predicted network demand → MARL → dynamic slicing.
* Thus, this paper provides a strong foundation for the decision-making/resource-allocation part of our proposed 6G architecture.


________________








































Tab 2
Additional Papers






Paper 3: Toward Self-Optimizing 6G Networks Through Network Digital Twin Intelligence: A Real Network Traffic Evaluation (2026)


1. Problem Statement
* Existing Digital Twins mainly support monitoring and simulation rather than autonomous optimization.
* Manual network management cannot efficiently handle dynamic 6G traffic.
* The paper proposes an AI-native Digital Twin for prediction, anomaly detection, load balancing, and energy optimization.


2. Methodology / Algorithm
* Uses KPIs such as traffic, latency, load, energy efficiency, and reliability.
* Predicts traffic using a rolling average over previous observations.
* Employs Isolation Forest for anomaly detection and MLP for latency prediction.
* Performs adaptive load balancing and energy optimization in a closed-loop framework.


3. Dataset / Simulation Tool
* Uses the Telecom Italia Milan Grid Dataset.
* Contains approximately 10,000 cells and 1.85 million traffic records.
* Evaluates Internet traffic and Voice/SMS traffic separately.
* Models trained using a 60/20/20 train-validation-test split.


4. Results
* Achieved 96.7% Internet traffic prediction accuracy and 98.3% Voice/SMS prediction accuracy.
* Reduced network load by 25.5% and achieved 30.4% energy savings.
* Obtained high anomaly detection and prevention accuracy.
* Outperformed conventional rule-based optimization methods.


5. Research Gap
* Uses traditional ML models (Isolation Forest and MLP) instead of graph learning.
* Traffic prediction relies on a simple rolling average.
* Evaluation is limited to one dataset and a few KPIs.
6. How it Relates to Our Project
* Demonstrates AI-driven Digital Twin intelligence for autonomous network management.
* Our framework replaces conventional ML with SSL and TGNN for richer network representations.
* MARL enables collaborative and adaptive network slicing beyond rule-based optimization.
________________


Tab 3
Datasets:


Dataset / Tool
	Purpose
	Telecom Italia Milan
	Real cellular traffic forecasting
	NeversNet5G
	5G wireless state + mobility + QoS
	B5G Network Slicing Dataset
	eMBB/URLLC/mIoT slicing + QoS
	



1. Telecom Italia Milan Mobile Traffic Dataset
Telecom Italia dataset on Harvard Dataverse
Milan Grid dataset on Harvard Dataverse
Scientific Data paper describing the dataset
What it contains
This is one of the most important datasets for your project.
It contains anonymized mobile-network activity from Milan, aggregated into spatial grid cells and 10-minute intervals.
The telecom activity includes:
* SMS-in
* SMS-out
* Call-in
* Call-out
* Internet activity
* Grid ID
* Timestamp
* Country code
The original dataset covers November 1, 2013 to January 1, 2014, with the city represented spatially by a grid. Nature
The Milan grid is approximately 100 × 100 spatial cells, with cells around 235 m × 235 m. PLOS
Why it is extremely useful for you
This is almost tailor-made for the TGNN part.
Critical limitation
It does not contain actual 5G slice allocation, PRB allocation, latency constraints or MARL actions.
So don't pretend that Milan alone represents a 6G network.
Instead, use it as your real-world traffic-demand layer.


2. NeversNet5G
This is one of the most interesting datasets I found for your project because it is very recent.
NeversNet5G project page
NeversNet5G dataset on Hugging Face
Zenodo DOI for NeversNet5G
What it contains
NeversNet5G is a city-scale 5G NR vehicular network dataset generated using:
SUMO   +  Simu5G / OMNeT++
It contains approximately:
* 390 vehicle trajectories
* 19 real-world base-station locations
* 4 network-load scenarios
* 242+ million records
* SINR
* CQI
* throughput
* latency
* vehicle position
* speed
* road segment
* UE identity
* network measurements
The released data contains 242,304,261 rows across 716 per-UE CSV files. 6G-TWIN
Why this is extremely interesting for your Digital Twin
You essentially get:
Mobility
   +
5G network state
   +
Base stations
   +
SINR
   +
CQI
   +
Throughput
   +
Latency
That gives you a much richer state representation than Milan.
You can construct:
Graph:


gNB ───── gNB
 │                       │
 │                       │
 UE ─────── UE
and feed temporal network states into your TGNN.


Big limitation
It is generated through simulation, rather than being raw operational 5G measurements.










3. B5G Network Slicing Dataset (Farreras et al., 2024) 
Link: https://zenodo.org/records/10610616 (paper: https://doi.org/10.1016/j.dib.2024.110738)
* Generated through a packet-level simulator, it captures network slicing considering the three main 3GPP slice types: eMBB, URLLC, and mIoT, across a wide range of scenarios with varying topologies, slice instances, and traffic flows. Each sample pairs a network configuration (topology, traffic characteristics, routing) with performance metrics (delay, jitter, loss) per flow, and includes deliberately over- and under-provisioned scenarios. Zenodonih
* Why it fits: it's already labeled by your exact three slice types (eMBB/URLLC/mMTC), gives you real topologies to build your TGNN's graph structure on, and the over/under-provisioning cases are perfect negative examples for training your MARL reward function to avoid QoS violations.
What it contains
This dataset was specifically generated for 5G/B5G network slicing.
It contains:
* network topology
* traffic characteristics
* routing configurations
* eMBB
* URLLC
* mIoT
* delay
* jitter
* packet loss
* flow-level performance
The dataset was generated using packet-level network simulation and includes scenarios with different levels of over/under-provisioning. ScienceDirect
Why it fits your project
This is almost exactly the environment you need for:
MARL state
   ↓
Agent action
   ↓
Resource allocation
   ↓
Network
   ↓
Delay / jitter / loss
   ↓
Reward
You can formulate:
Agent state
Traffic demand
Available bandwidth
Current utilization
Slice type
Latency
Packet loss
Previous allocation
Action
Allocate bandwidth
Allocate resources
Change slice allocation
Reward
Something like:
Reward =
QoS satisfaction
+ resource utilization
- latency penalty
- packet-loss penalty
- SLA violation penalty
Major limitation
The dataset models the transport network, not the complete RAN.
The paper explicitly states that the scenarios exclude the RAN infrastructure. ScienceDirect
So this should be your slicing/control benchmark, not your entire 6G dataset.
________________


Additional Candidate Datasets (Round 2 — 16 Sep 2026)

4. 5G-NIDD (University College Dublin, NetSlab, 2022)
Link: Kaggle (search "5G-NIDD"), Figshare, and IETSIN/Fairdata mirror. Paper: "5G-NIDD: A Comprehensive Network Intrusion Detection Dataset."
What it contains
* Bidirectional flow records (CICFlowMeter-style, 100+ features) captured on a real, operational 5G testbed (not simulated) with commercial UEs and an OAI-based 5G core.
* Labeled benign traffic plus 8 attack classes (DoS variants, port scans, botnet, etc.), with per-flow duration, throughput, IPs/ports and timestamps.
Why it fits
* Genuine 5G-core traffic (not synthetic) — strong candidate for the SSL pretraining stage (masked/contrastive pretext tasks on real flow features).
* Anomaly/attack labels are also useful for an auxiliary "trustworthy slicing" angle (flagging traffic that shouldn't get URLLC-grade resources).
Limitation
* No slice-type labels (eMBB/URLLC/mMTC), no topology/graph structure, and no resource-allocation actions — needs pairing with a graph/slicing dataset.

5. DeepSlice / "5G Network Slicing" Dataset (Kaggle)
Link: Kaggle — "DeepSlice & Secure5G - 5G & LTE Wireless Dataset."
What it contains
* ~63,000 synthetic session-level records, each labeled with a Slice Type (eMBB / URLLC / mMTC).
* QoS/context features: latency, packet delay, packet-loss rate, throughput/speed, GBR flag, LTE/5G flag, and use-case tags (Smart City, Healthcare, Public Safety, Industry 4.0, IoT).
Why it fits
* Directly labeled by our exact three slice classes — a ready-made supervised benchmark for the classification/QoS-prediction head that sits downstream of SSL+TGNN.
Limitation
* Row-independent (no temporal sequencing, no topology/graph) and fully synthetic — cannot drive the TGNN or a closed-loop MARL reward on its own.

6. robertbotez/6g-network-slicing-dataset (Botez, Zinca & Dobrota, 2025)
Link: github.com/robertbotez/6g-network-slicing-dataset (CC-BY-4.0). Paper: "Redefining 6G Network Slicing: AI-Driven Solutions for Future Use Cases," Electronics 2025, 14, 368, https://doi.org/10.3390/electronics14020368
What it contains
* A synthetic CSV (v3) built specifically for 6G slice classification, extending beyond classic eMBB/URLLC/mMTC toward emerging 6G use cases (e.g. holographic communication, massive twinning).
Why it fits
* The newest (2025) dataset explicitly framed as "6G" rather than 5G — useful as a secondary benchmark to test whether our SSL+TGNN representations generalize past legacy 5G slice categories.
Limitation
* Same shape as DeepSlice: row-level, synthetic, no graph/temporal structure.

7. 5GAD-2022 (Idaho National Laboratory)
Link: github.com/IdahoLabResearch/5GAD, mirrored on Zenodo. Paper: "5GAD-2022: 5G Network Traffic Dataset for Machine Learning."
What it contains
* Traffic captured on a real (non-simulated) 5G testbed under normal and adversarial/attack conditions, intended for ML-based anomaly detection.
Why it fits
* A second source of genuinely real 5G traffic for SSL pretraining, complementing 5G-NIDD and diversifying the traffic distribution beyond a single testbed.
Limitation
* Security-oriented; no slice or topology labels.

8. 5G Campus Network QoS Dataset for Open-Source gNB (Zenodo)
What it contains
* QoS measurements (throughput, latency, RSRP/RSRQ/SINR) collected from a real, open-source 5G Standalone campus-network gNB deployment.
Why it fits
* Real RAN-side radio QoS telemetry — fills the gap left by Milan (core/backhaul-level) and NeversNet5G (simulated RAN) with genuinely measured radio-link quality.
Limitation
* Small scale (single campus deployment); no explicit multi-slice labels.

9. ITU AI/ML in 5G Challenge — QoS Prediction dataset
Link: github.com/ITU-AI-ML-in-5G-Challenge/Challenge_Archive
What it contains
* Standardized, published challenge dataset(s) (2021/2022 editions) mapping network-state features to a QoS-prediction task, with existing baseline results and leaderboards.
Why it fits
* Gives us an external, citable benchmark to show our SSL+TGNN representation beats a published baseline — useful evidence for the journal/survey novelty argument.
Limitation
* Task-specific (QoS classification), not a full slicing/control environment.

Simulation & Tooling Needed to Close the Remaining Gap
No single dataset above provides unlabelled telemetry + real topology + slice-tagged QoS + closed-loop control together in one place. Recommended combination by pipeline stage:
* SSL pretraining corpus (volume + realism): Telecom Italia Milan + 5G-NIDD + 5GAD-2022.
* TGNN topology & dynamics (graph-shaped, time-varying): NeversNet5G (gNB/UE graph with SINR, CQI, throughput, latency over time).
* MARL reward shaping / slice-type supervision: B5G Network Slicing Dataset (over/under-provisioned eMBB/URLLC/mIoT) + DeepSlice + robertbotez dataset for slice-type labels.
* Missing piece — a dataset with real topology AND time-varying per-slice resource-allocation actions/rewards together — does not appear to exist publicly. Plan to generate it ourselves using a system-level simulator:
  - Simu5G (OMNeT++) or ns-3 5G-LENA to emit synchronized topology + eMBB/URLLC/mMTC scheduling + per-slice QoS.
  - Wrap the simulator (or dataset replay) as a Dec-POMDP using PettingZoo + Ray RLlib so the MARL agents train against a standard multi-agent Gym-style API.
  - Use PyTorch Geometric Temporal as the reference TGNN library for the spatio-temporal encoder consuming SSL-pretrained node embeddings.
________________