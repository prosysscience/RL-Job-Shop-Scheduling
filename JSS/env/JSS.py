import bisect
import datetime
import random

import pandas as pd
import gym
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


class JSS(gym.Env):

    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:

        -The actions correspond to a job allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a job, the end of the job is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """
        if env_config is None:
            env_config = {'instance_path': '/home/local/IWAS/pierre/PycharmProjects/JSS/JSS/env/instances/ta80'}
        instance_path = env_config['instance_path']

        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.max_action_step = 0
        self.nb_legal_actions = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_time_step = float('inf')
        self.current_time_step = float('inf')
        self.next_time_step = list()
        self.legal_actions = None
        self.action_step = 0
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        instance_file = open(instance_path, 'r')
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(np.int, 2))
                # contains all the time to complete jobs
                self.jobs_length = np.zeros(self.jobs, dtype=np.int)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[job_nb][i // 2] = (machine, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[job_nb] += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()
        self.max_time_jobs = max(self.jobs_length)
        self.max_action_step = self.machines * self.jobs
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, 'We need at least 2 machines'
        assert self.max_action_step > 0
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        '''
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        '''
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs * 7,), dtype=np.float),
        })

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state.reshape(-1),
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.action_step = 0
        self.current_time_step = 0
        self.next_time_step = list()
        self.nb_legal_actions = self.jobs
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=np.bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=np.int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=np.int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=np.int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=np.int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=np.int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=np.int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=np.int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=np.int)
        for job in range(self.jobs):
            self.needed_machine_jobs[job] = self.instance_matrix[job][0][0]
        self.state = np.zeros((self.jobs, 7), dtype=np.float)
        return self._get_current_state_representation()

    def step(self, action: int):
        reward = 0.0
        if action == self.jobs:
            self.legal_actions[self.jobs] = False
            only_legal = np.where(self.legal_actions)[0][0]
            while self.nb_legal_actions == 1 and len(self.next_time_step) > 0:
                reward -= self._increase_time_step()
            scaled_reward = self._reward_scaler(reward)
            if self.nb_legal_actions > 1:
                self.legal_actions[only_legal] = False
                self.nb_legal_actions -= 1
            return self._get_current_state_representation(), scaled_reward, self._is_done(), {}
        self.action_step += 1
        current_time_step_job = self.todo_time_step_job[action]
        machine_needed = self.instance_matrix[action][current_time_step_job][0]
        time_needed = self.instance_matrix[action][current_time_step_job][1]
        reward += time_needed
        self.time_until_available_machine[machine_needed] = time_needed
        self.time_until_finish_current_op_jobs[action] = time_needed
        self.state[action][1] = time_needed / self.max_time_op
        bisect.insort_left(self.next_time_step, self.current_time_step + time_needed)
        self.solution[action][current_time_step_job] = self.current_time_step
        for job_id in range(self.jobs):
            if self.needed_machine_jobs[job_id] == machine_needed and self.legal_actions[job_id]:
                self.legal_actions[job_id] = False
                self.nb_legal_actions -= 1
        # if we can't allocate new job in the current timestep, we pass to the next one
        while self.nb_legal_actions == 0 and len(self.next_time_step) > 0:
            reward -= self._increase_time_step()
        if self.nb_legal_actions == 1 and len(self.next_time_step) > 0:
            self.legal_actions[self.jobs] = True
        elif self.legal_actions[self.jobs]:
            self.legal_actions[self.jobs] = False
        # we then need to scale the reward
        scaled_reward = self._reward_scaler(reward)
        return self._get_current_state_representation(), scaled_reward, self._is_done(), {}

    def _reward_scaler(self, reward):
        reward = reward / self.max_time_op
        return reward

    def _increase_time_step(self):
        '''
        The heart of the logic his here, we need to increase every counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        '''
        hole_planning = 0
        next_time_step = self.next_time_step.pop(0)
        difference = next_time_step - self.current_time_step
        self.current_time_step = next_time_step
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            performed_op_job = min(difference, was_left_time)
            self.time_until_finish_current_op_jobs[job] = max(0, self.time_until_finish_current_op_jobs[
                job] - difference)
            self.state[job][1] = self.time_until_finish_current_op_jobs[job] / self.max_time_op
            if was_left_time > 0:
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = self.total_perform_op_time_jobs[job] / self.max_time_jobs
                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.total_idle_time_jobs[job] += (difference - was_left_time)
                    self.state[job][6] = self.total_idle_time_jobs[job] / (self.max_time_jobs * self.jobs)
                    self.idle_time_jobs_last_op[job] = (difference - was_left_time)
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / (self.max_time_jobs * self.jobs)
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] += (1.0 / self.machines)
                    if self.todo_time_step_job[job] < self.machines:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][self.todo_time_step_job[job]][0]
                        self.state[job][4] = max(0, self.time_until_available_machine[
                                                 self.needed_machine_jobs[job]] - difference) / self.max_time_op
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a good candidate)
                        self.state[job][4] = 1.0
            else:
                self.total_idle_time_jobs[job] += (difference - was_left_time)
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / (self.max_time_jobs * self.jobs)
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(0, self.time_until_available_machine[
                machine] - difference)
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if self.needed_machine_jobs[job] == machine and not self.legal_actions[job]:
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
        return hole_planning

    def _is_done(self):
        if self.action_step >= self.max_action_step - 1:
            self.last_time_step = self.current_time_step
            return True
        return False

    def render(self, mode='human'):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = 'Job {}'.format(job)
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(self.instance_matrix[job][i][0])
                df.append(dict_op)
                i += 1
        if len(df) > 0:
            df = pd.DataFrame(df)
            colors = [
                tuple([random.random() for i in range(3)]) for _ in range(self.machines)
            ]
            fig = ff.create_gantt(df, index_col='Resource', colors=colors, show_colorbar=True,
                                  group_tasks=True)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        return fig