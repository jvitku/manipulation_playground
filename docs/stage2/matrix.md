| skill / learner | metric | with force | without force | Welch p | MW p |
|---|---|---|---|---|---|
| TD3+BC insert, centred grasps | insert success | 74.8 % [67.4, 80.4] (n=10) | 56.2 % [51.6, 60.2] (n=10) | 0.00049 | 0.0027 |
| TD3+BC insert, randomised in-hand offsets | insert success | 62.0 % [56.0, 68.0] (n=6) | 52.7 % [47.7, 57.7] (n=6) | 0.06 | 0.065 |
| TD3+BC pick (held + lifted) | pick success | 99.0 % [98.0, 99.8] (n=10) | 99.6 % [99.0, 100.0] (n=10) | 0.27 | 0.32 |
| TD3+BC pick (centred grasp) | pick success | 48.7 % [42.3, 54.7] (n=6) | 43.0 % [36.7, 48.3] (n=6) | 0.26 | 0.29 |
| BC (MLP) insert, randomised offsets | insert success | 28.8 % [26.4, 30.8] (n=5) | 6.8 % [4.0, 10.0] (n=5) | 8e-06 | 0.012 |

| full sort (10 episodes each) | episodes | parts | pick | insert after pick |
|---|---|---|---|---|
| sequencer, unconstrained picks, force | 0/10 | 0/60 | 12/12 | 0/12 [0-24 %] |
| sequencer, unconstrained pick + offset-robust insert, no force | 1/10 | 14/60 | 34/43 | 11/31 [21-53 %] |
| sequencer, unconstrained pick + offset-robust insert, force | 0/10 | 7/60 | 20/20 | 7/19 [19-59 %] |
| sequencer, centred pick + offset-robust insert, force | 0/10 | 12/60 | 19/28 | 11/18 [39-80 %] |
| sequencer, centred pick + offset-robust insert, no force | 0/10 | 14/60 | 37/42 | 9/32 [16-45 %] |
