import numpy as np
import os
import json

# the main for learning

from learner.mlearner import MLearner
from learner.model import genModelTermsfromString, Model, genModelfromCoeff
from learner.ready_db import ReadyDB
from learner.lib import *
from learner.constants import AdaptationLevel

from learner.tranlearner import TranLearner

model_path = os.path.expanduser("~/catkin_ws/src/cp1_base/power_models/")
learned_model_path = os.path.expanduser("~/cp1/")
config_list_file = os.path.expanduser('~/cp1/config_list.json')
config_list_file_true = os.path.expanduser('~/cp1/config_list_true.json')
ready_json = os.path.expanduser("~/ready")
learned_model_name = 'learned_model'
used_budget_report_path = os.path.expanduser("~/cp1/used_budget")

ndim = 20
test_size = 10000
mu, sigma = 0, 0.1
speed_list = [0.15, 0.3, 0.6]

offline_learning_budget_ratio = 0.2

opIDs={}
numOfOptions=20
for i in range(numOfOptions):
    opIDs["o"+str(i)]=i



class Learn:
    def __init__(self):
        self.ready = ReadyDB(ready_db=ready_json)
        self.budget = self.ready.get_budget()
        self.used_budget = 0
        self.model_name = self.ready.get_power_model()
        default_conf = np.concatenate((np.zeros(int(ndim/2)), np.ones(ndim-int(ndim/2))))
        self.default_conf = np.reshape(default_conf, (1, ndim))
        self.learned_model_filepath = os.path.join(learned_model_path, learned_model_name)
        self.true_model_filepath = os.path.join(model_path, self.model_name)
        self.config_list_file = config_list_file
        self.config_list_file_true = config_list_file_true
        self.true_power_model = None
        self.learned_power_model = None
        self.learned_model = None
        self.learner = None
       

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

    # For case c and case d (offline learning)
    def start_learning(self):

        # learn the model
        try:
            if self.ready.get_baseline() == AdaptationLevel.BASELINE_C:
                self.learner = MLearner(self.budget, ndim, self.true_power_model)
                self.learned_model = self.learner.discover()
                self.used_budget = self.budget
            elif self.ready.get_baseline() == AdaptationLevel.BASELINE_D:
                self.learner = TranLearner(self.budget, ndim, self.true_power_model)
                self.learned_model = self.learner.offline_learning()
                self.used_budget = self.learner.used_budget

            with open(used_budget_report_path, "w") as fp:
                fp.write(str(self.used_budget))
        except Exception as e:
            raise Exception(e)

    # For case d
    def start_online_learning(self):
        try:
            self.learned_model = self.learner.online_learning()
            self.used_budget = self.learner.used_budget
            with open(used_budget_report_path, "w") as fp:
                fp.write(str(self.used_budget))
        except Exception as e:
            raise Exception(e)

    # For case c
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


    def update_config_files(self):

        # configs = itertools.product(range(2), repeat=ndim)
        # xTest = np.zeros(shape=(2**ndim, ndim))
        # i = 0
        # for c in configs:
        #     xTest[i, :] = np.array(c)
        #     i += 1

        test_size = 10000

        xTest = np.random.randint(2, size=(test_size, ndim))

        for i in range(test_size):
            if np.count_nonzero(xTest[i, :]) == 0:
                xTest = np.delete(xTest, i, 0)
                break

        # to avoid negative power load
        if self.ready.get_baseline() == AdaptationLevel.BASELINE_C: 
            yTestPower = abs(self.learned_model.predict(xTest))
        if self.ready.get_baseline() == AdaptationLevel.BASELINE_D:
            predY, predYStd = self.learned_model.predict(xTest, with_noise=False)
            predY = np.ravel(predY)
            predYStd = np.ravel(predYStd)

            halfIntervals = 1.729*predYStd
            goodIndices = np.where(predY>halfIntervals)

            xTest       = xTest[goodIndices]
            test_size   = xTest.shape[0]
            
            predY       = predY[goodIndices]
            predYStd    = predYStd[goodIndices]
            yTestPower  = predY
 
        
        yTestPower_true = self.true_power_model.evaluateModelFast(xTest)

        # adding noise for the speed
        s = np.random.uniform(mu, sigma, test_size)

        yTestSpeed = np.zeros(test_size)
        for i in range(test_size):
            yTestSpeed[i] = speed_list[i % len(speed_list)]

        yTestSpeed = yTestSpeed + s

        if self.ready.get_baseline() == AdaptationLevel.BASELINE_C:
            yDefaultPower = abs(self.learned_model.predict(self.default_conf))
        elif self.ready.get_baseline() == AdaptationLevel.BASELINE_D:
            defaultPredY, defaultPredYVar = self.learned_model.predict(self.default_conf, with_noise=False)
            yDefaultPower = defaultPredY
            yDefaultPower = abs(np.ravel(yDefaultPower))

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
        with open(self.config_list_file, 'w') as outfile:
            json.dump(json_data, outfile)
            print("\n**Predicted**")
            print(json_data)

        json_data_true_model['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower_true[0]/3600*1000,
            'power_load_w': yDefaultPower_true[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file_true, 'w') as outfile:
            json.dump(json_data_true_model, outfile)
            print("\n**True**")
            print(json_data_true_model)



    def dump_true_default_config(self):
        '''
            Write the default config based on the true model for case A and case B.
            Write into config_list_true.json
        '''

        yDefaultPower_true = self.true_power_model.evaluateModelFast(self.default_conf)
        yDefaultSpeed = speed_list[2]
        json_data_true_model = {}
        json_data_true_model['configurations'] = []
        json_data_true_model['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower_true[0]/3600*1000,
            'power_load_w': yDefaultPower_true[0],
            'speed': yDefaultSpeed
        })
        with open(self.config_list_file_true, 'w') as outfile:
            json.dump(json_data_true_model, outfile)

    def has_budget(self):
        return self.budget > self.used_budget
