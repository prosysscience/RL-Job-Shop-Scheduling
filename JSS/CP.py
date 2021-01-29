from __future__ import print_function

import collections

# Import Python wrapper for or-tools CP-SAT solver.
import time

import wandb
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback


class WandbCallback(CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self):
        CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        wandb.log({'time': 0, 'solution_count': self.__solution_count})

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        self.__solution_count += 1
        wandb.log({'time': current_time - self.__start_time, 'make_span': obj, 'solution_count': self.__solution_count})

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


def MinimalJobshopSat(instance_config= {'instance_path': 'instances/ta51'}):
    wandb.init(config=instance_config)
    config = wandb.config
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = []
    machines_count = 0

    instance_file = open(config['instance_path'], 'r')
    line_str = instance_file.readline()
    line_cnt = 1
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, time = int(split_data[i]), int(split_data[i + 1])
                data.append((machine, time))
                i += 2
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    wandb_callback = WandbCallback()
    status = solver.SolveWithSolutionCallback(model, wandb_callback)


if __name__ == "__main__":
    MinimalJobshopSat()