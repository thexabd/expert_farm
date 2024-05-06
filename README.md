# Expert Initiated PPO

_Project done as part of my dissertation at the University of Nottingham Malaysia._

A modification of the PPO to allow it to learn from expert generated trajectories while gradually generating its own experiences to replicate the expert. The PPO operates on Farm0 from the Farm-gym library.

Full report will be posted in the future.

To run the baseline PPO, run the following command line code:
`python PPO.py`

To run the EI-PPO, run the following command line code:
`python EI_PPO.py --variaion (1 or 2) --mimicry_coef (any integer, default=beta)`

To evaluate the models after training, run the following command line code:
`python eval.py --model_loc (PPO.pt or EI_PPO.pt)`
