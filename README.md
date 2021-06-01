A Reinforcement Learning Environment For Job-Shop Scheduling
==============================

This folder contains the implementation of the paper "A Reinforcement Learning Environment For Job-Shop Scheduling".

It contains the deep reinforcement learning approach we have developed to solve the Job-Shop Scheduling problem.

The optimized environment is available as a separate [repository](https://github.com/prosysscience/JSSEnv).

![til](./ta01.gif)

If you've found our work useful for your research, you can cite the [paper](https://arxiv.org/abs/2104.03760) as follows:

```
@misc{tassel2021reinforcement,
      title={A Reinforcement Learning Environment For Job-Shop Scheduling}, 
      author={Pierre Tassel and Martin Gebser and Konstantin Schekotihin},
      year={2021},
      eprint={2104.03760},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Getting Started
------------

**This code has been tested on Ubuntu 18.04 and MacOs 10.15. 
Some users have reported difficulties running this program on Windows.**

This work uses Ray's RLLib, Tensorflow and Wandb.

Make sure you have `git`, `cmake`, `zlib1g`, and, on Linux, `zlib1g-dev` installed.

You also need to have a Weight and Bias account to log your metrics. 
Otherwise, just remove all occurrence of wandb and log the metrics in another way.


```shell
git clone https://github.com/prosysscience/JSS
cd JSS
pip install -r requirements.txt
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 

Project Organization
------------

    ├── README.md                 <- The top-level README for developers using this project.
    └── JSS
        ├── dispatching_rules/      <- Contains the code to run the disptaching rule FIFO and MWTR.
        ├── instances/              <- All Taillard's instances + 5 Demirkol instances.
        ├── randomLoop/             <- A random loop with action mask, usefull to debug environment and
        |                             to check if our agent learn.
        ├── CP.py                   <- OR-Tool's cp model for the JSS problem.
        ├── CustomCallbacks.py      <- A special RLLib's callback used to save the best solution found.
        ├── default_config.py       <- default config used for the disptaching rules.
        ├── env_wrapper.py          <- Envrionment wrapper to save the action's of the best solution found
        ├── main.py                 <- PPO approach, the main file to call to reproduce our approach.
        └── models.py               <- Tensorflow model who mask logits of illegal actions.

--------

## License

MIT License
