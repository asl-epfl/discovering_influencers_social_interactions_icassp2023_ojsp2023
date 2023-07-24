# discovering_influencers_social_interactions_icassp2023_ojsp2023

#### The data and the code for the experiments in the papers: 

V. Shumovskaia, M. Kayaalp, M. Cemri, and A. H. Sayed, “[Discovering influencers in opinion formation over social graphs](https://ieeexplore.ieee.org/document/10079214),” IEEE Open Journal of Signal Processing, pp. 1–20, 2023.

V. Shumovskaia, M. Kayaalp, and A. H. Sayed, “[Identifying opinion influencers over social networks](https://ieeexplore.ieee.org/document/10094722),” IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pp. 1–5, 2023.

#### To get the figures from the paper, run these scripts:
```
python plot_draft.py
```

#### To fully run experiments, we refer to runs_final.txt, e.g.:

```
python main.py --likelihood_regime 0 --times 70000 --adjacency_regime 3 --agents 20 --er_prob 0.2 --seed 21 --lr 0.1 --step_size 0.05 --likelihood_regime 3 --multistate --states 5 --comparison --window 50
```

To understand arguments for the parser we refer to ```parser.py``` or run python ```main.py --help```.

##### Find experiments on <span style="color:blue">Twitter</span> data in the following repo [asl-epfl/discovering_influencers_twitter_ojsp2023](https://github.com/asl-epfl/discovering_influencers_twitter_ojsp2023).
