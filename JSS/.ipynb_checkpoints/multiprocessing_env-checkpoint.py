# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env
import gym
import numpy as np
import cloudpickle
import multiprocessing
from multiprocessing import Process, Pipe

from JSS.env_wrapper import BestActionsWrapper, MaxStepWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'reset_task':
                ob = env.reset_task()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_best_actions':
                remote.send((env.best_score, env.best_actions))
            elif cmd == 'get_best_timestep':
                remote.send(env.best_time_step)
            elif cmd == 'get_legal_actions':
                ob = env.get_legal_actions()
                remote.send(ob)
            else:
                raise NotImplementedError
        except EOFError:
            break


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None, start_method=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.nenvs = n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in multiprocessing.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_legal_actions(self):
        for remote in self.remotes:
            remote.send(('get_legal_actions', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def __len__(self):
        return self.nenvs


def make_seeded_env(i: int, env_name: str, seed: int, max_steps_per_episode: int, env_config: dict = {}):
    def _anon():
        if max_steps_per_episode is None:
            env = BestActionsWrapper(gym.make(env_name, env_config=env_config))
        else:
            env = BestActionsWrapper(MaxStepWrapper(gym.make(env_name, env_config=env_config), max_steps_per_episode))
        env.seed(seed + i)
        return env

    return _anon