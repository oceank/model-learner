import numpy as np
import os
import json

# the main for learning

from learner.mlearner import MLearner
from learner.model import genModelTermsfromString, Model, genModelfromCoeff
from learner.ready_db import ReadyDB
from learner.lib import *

# used for DQN learning 
from learner.power_system import powerSystem
from learner.DQN_agent import DQNAgent
from learner.DQN_learner import DQNLearner
 
model_path = os.path.expanduser("~/catkin_ws/src/cp1_base/power_models/")
learned_model_path = os.path.expanduser("~/cp1/")
config_list_file = os.path.expanduser('~/cp1/config_list.json')
config_list_file_true = os.path.expanduser('~/cp1/config_list_true.json')
ready_json = os.path.expanduser("~/ready")
learned_model_name = 'learned_model'
DQN_result_dir = os.path.expanduser("~/cp1/")

ndim = 20
test_size = 10000
mu, sigma = 0, 0.1
speed_list = [0.15, 0.3, 0.6]

# For DQN learning #
update_batch_size = 32 # number of samples used in memory replay in DQN learning
exploration_bonus = 0 # used by exploration function (EF). 0 means that EF is disabled.
opIDs={}
numOfOptions=20
for i in range(numOfOptions):
    opIDs["o"+str(i)]=i



class Learn:
    def __init__(self):
        self.ready = ReadyDB(ready_db=ready_json)
        self.budget = self.ready.get_budget()
        self.left_budget = self.budget
        self.model_name = self.ready.get_power_model()
        self.default_conf = np.reshape(np.zeros(ndim), (1, ndim))
        self.learned_model_filepath = os.path.join(learned_model_path, learned_model_name)
        self.true_model_filepath = os.path.join(model_path, self.model_name)
        self.config_list_file = config_list_file
        self.true_power_model = None
        self.learned_power_model = None
        self.learned_model = None
        self.learner = None
        self.DQN_learner = None

    def get_true_model(self):
        try:
            with open(self.true_model_filepath, 'r') as model_file:
                model_txt = model_file.read()

            power_model_terms = genModelTermsfromString(model_txt)
            self.true_power_model = Model(power_model_terms, ndim)
            print("The true model: {0}".format(self.true_power_model.__str__()))
            return self.true_power_model
        except Exception as e:
            raise Exception(e)

    def start_learning(self):

        # learn the model
        try:
            model_learning_budget = int(0.1*self.left_budget)
            self.left_budget -= model_learning_budget
            self.learner = MLearner(model_learning_budget, ndim, self.true_power_model)
            self.learned_model = self.learner.discover()
        except Exception as e:
            raise Exception(e)

    # Assumption: the learned power model is already dumpped
    def initialize_DQN_learner(self):
        learned_system = powerSystem(self.learned_model_filepath, opIDs)
        system = powerSystem(self.true_model_filepath, opIDs)
        
        input_layer_size = system.numOfOptions
        output_layer_size = system.numOfOptions

        agent = DQNAgent(input_layer_size, output_layer_size, exploration_bonus, update_batch_size,
                numOfOptionsToChangePerAction=1, numOfOptionsToChangeByGreedyPerAction=1)
        self.DQN_learner = DQNLearner(agent, system, DQN_result_dir, isDebug=False)

        # Initialize Replay Memory by using the learned system
        self.DQN_learner.initializeReplayMemory(learned_system)


    # Assumption: the self.DQN_learner has been intialized
    # Have DQN agent to learn and finally dump Pareto-optimial configurations
    def DQN_learning(self, budget):
        if self.left_budget < budget:
            raise Exception("[DQN Learning Error] The left budget ({0}) in learner is smaller the requested one {1}."
                    .format(self.left_budget, budget))

        yTestPower = self.DQN_learner.learning(totalIters = budget)

        # adding noise for the speed
        s = np.random.uniform(mu, sigma, budget)

        yTestSpeed = np.zeros(budget)
        for i in range(budget):
            yTestSpeed[i] = speed_list[i % len(speed_list)]

        yTestSpeed = yTestSpeed + s

        yDefaultPower = abs(self.learned_model.predict(self.default_conf))
        yDefaultSpeed = speed_list[2]

        idx_pareto, pareto_power, pareto_speed = self.learner.get_pareto_frontier(yTestPower, yTestSpeed, maxX=False, maxY=True)
        json_data = get_json(pareto_power, pareto_speed)

        # add the default configuration
        json_data['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower[0]/3600*1000,
            'power_load_w': yDefaultPower[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file, 'w') as outfile:
            json.dump(json_data, outfile)

        self.left_budget -= budget

    def dump_learned_model(self):
        """dumps model in ~/cp1/"""

        try:
            learned_power_model_terms = genModelfromCoeff(self.learned_model.named_steps['linear'].coef_, ndim)
            self.learned_power_model = Model(learned_power_model_terms, ndim)
        except Exception as e:
            raise Exception(e)

        print("The learned model: {0}".format(self.learned_power_model.__str__()))

        with open(self.learned_model_filepath, 'w') as model_file:
            model_file.write(self.learned_power_model.__str__())

        # configs = itertools.product(range(2), repeat=ndim)
        # xTest = np.zeros(shape=(2**ndim, ndim))
        # i = 0
        # for c in configs:
        #     xTest[i, :] = np.array(c)
        #     i += 1

        xTest = np.random.randint(2, size=(test_size, ndim))

        for i in range(test_size):
            if np.count_nonzero(xTest[i, :]) == 0:
                xTest = np.delete(xTest, i, 0)
                break

        # to avoid negative power load
        yTestPower = abs(self.learned_model.predict(xTest))
        yTestPower_true = self.true_power_model.evaluateModelFast(xTest)

        # adding noise for the speed
        s = np.random.uniform(mu, sigma, test_size)

        yTestSpeed = np.zeros(test_size)
        for i in range(test_size):
            yTestSpeed[i] = speed_list[i % len(speed_list)]

        yTestSpeed = yTestSpeed + s

        yDefaultPower = abs(self.learned_model.predict(self.default_conf))
        yDefaultPower_true = self.true_power_model.evaluateModelFast(self.default_conf)
        yDefaultSpeed = speed_list[2]

        idx_pareto, pareto_power, pareto_speed = self.learner.get_pareto_frontier(yTestPower, yTestSpeed, maxX=False, maxY=True)
        json_data = get_json(pareto_power, pareto_speed)

        json_data_true_model = get_json([yTestPower_true[i] for i in idx_pareto], [yTestSpeed[i] for i in idx_pareto])

        # add the default configuration
        json_data['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower[0]/3600*1000,
            'power_load_w': yDefaultPower[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file, 'w') as outfile:
            json.dump(json_data, outfile)

        json_data_true_model['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower_true[0]/3600*1000,
            'power_load_w': yDefaultPower_true[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file_true, 'w') as outfile:
            json.dump(json_data_true_model, outfile)


