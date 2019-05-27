import random
import numpy as np
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import print_summary as modelSummary

import tensorflow as tf



class DQNAgent:
    def __init__(self, optionSize, actionSize, explorationBonus, updateBatchSize, useDoubleDQN=True, 
            numOfOptionsToChangePerAction=1, numOfOptionsToChangeByGreedyPerAction=1, actionType="e-greedy"):
        self.optionSize = optionSize
        self.actionSize = actionSize
        self.explorationBonus = explorationBonus
        self.updateBatchSize = updateBatchSize
        self.useDoubleDQN = useDoubleDQN
        self.exploredSAs = {} # a dictionary that tracks the exploration number of a state folloiwng an action. (state, action) : counts
        self.memory = {} # (state, action, nextState) : reward
        self.gamma = 0.95    # discount rate
        self.learningRateMax = 0.7
        self.learningRateMin = 0.1
        self.epsilonMax = 1.0
        self.epsilonMin = 0.1
        self.epsilon = 1.0  # used by epsilon-greedy exploration approach
        self.optimizer = Adam(lr=self.learningRateMax)
        self.alwaysRandomAction = True if actionType=="random" else False
        self.numOfOptionsToChangePerAction = numOfOptionsToChangePerAction
        self.numOfOptionsToChangeByGreedyPerAction = numOfOptionsToChangeByGreedyPerAction
        assert self.numOfOptionsToChangeByGreedyPerAction <= self.numOfOptionsToChangePerAction
        assert actionSize >= self.numOfOptionsToChangePerAction


        self.predictModel = self.buildModelCNN()
        self.targetModel = self.buildModelCNN()
        self.updateTargetModel()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """
    def huberLoss(self, yTrue, yPred, clipDelta=1.0):
        error = yTrue - yPred
        cond  = K.abs(error) <= clipDelta

        squaredLoss = 0.5 * K.square(error)
        quadraticLoss = 0.5 * K.square(clipDelta) + clipDelta * (K.abs(error) - clipDelta)

        return K.mean(tf.where(cond, squaredLoss, quadraticLoss))

    def buildModelCNN(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Reshape((self.optionSize, 1), input_shape=(1, self.optionSize)))
        model.add(Conv1D(32, 8, activation='relu', input_shape=(self.optionSize,1)))
        model.add(Conv1D(64, 4, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(AveragePooling1D(4))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.actionSize, activation='relu'))

        model.compile(loss=self.huberLoss, optimizer=self.optimizer)
        return model

    def get_Q_values(self, state):
        qValues=self.predictModel.predict(state)
        return qValues[0]

    def updateTargetModel(self):
        # copy weights from model to targetModel
        self.targetModel.set_weights(self.predictModel.get_weights())

    # If the current experience is already in the memory, return False.
    # Otherwise, add the experience into the memory, and return True.
    def remember(self, state, action, reward, nextState):
        policy = (tuple(state[0][0]), tuple(action), tuple(nextState[0][0]))
        if policy not in self.memory:
            self.memory[policy] = reward
            return True
        else:
            return False

    def act(self, state):
        if self.alwaysRandomAction:
            return False, random.sample(range(self.actionSize), self.numOfOptionsToChangePerAction)
        else:
            if np.random.rand() <= self.epsilon:
                return False, random.sample(range(self.actionSize), self.numOfOptionsToChangePerAction)
            else:
                return True, self.determineAction(state)

    # Use the prediction network
    def determineAction(self, state):
        qValues = self.predictModel.predict(state)

        # Flip a few options that correspond to top q-values and a few randomly-selected options
        greedyChangeOpIndices=np.argpartition(qValues[0], range(-self.numOfOptionsToChangeByGreedyPerAction, 0))[-self.numOfOptionsToChangeByGreedyPerAction:]
        greedyChangeOpIndices=np.flip(greedyChangeOpIndices) # list(reversed(greedyChangeOpIndices.tolist()))
        leftOptionList=[x for x in range(self.actionSize) if x not in greedyChangeOpIndices]
        randomChanges=random.sample(leftOptionList, self.numOfOptionsToChangePerAction - self.numOfOptionsToChangeByGreedyPerAction)

        return  greedyChangeOpIndices.tolist() + randomChanges

    # Input: record is a tuple, (state, action) where action is a list of options's numerical IDs
    def incrementExplorationCount(self, state, action):
        hashableState=tuple(state[0][0].tolist())
        hashableAction=tuple(action)
        record=(hashableState, hashableAction)
        if record in self.exploredSAs:
            self.exploredSAs[record]=self.exploredSAs[record]+1
        else:
            # for a record that does not exist in the dict, its count value is 1.
            # This helps with the DivisidedByZero issue when calculing the actual exploration bonus.
            self.exploredSAs[record]=2

    # Input: qValues: 1-dimentional tuple / list
    # Output: augmented qValues - using the exploration funciton, f(u, N) = u + k/N
    def explorationFunction(self, state, qValues):
        augQValues=[qv for qv in qValues]
        hashableState=tuple(state[0][0].tolist())
        for actionID in range(self.actionSize):
            N=1
            if (hashableState, actionID) in self.exploredSAs:
                N=self.exploredSAs[(hashableState, actionID)]
            augQValues[actionID]=augQValues[actionID]+1.0*self.explorationBonus/N
        return augQValues

    def replay(self):
        minibatch = random.sample(self.memory.items(), self.updateBatchSize)
        
        for policy, reward in minibatch:
            stateTuple, actionTuple, nextStateTuple = policy
            state=np.expand_dims([stateTuple], axis=0)
            nextState=np.expand_dims([nextStateTuple], axis=0)

            preNetQValues = self.predictModel.predict(state)
            targetNetQValues = self.targetModel.predict(nextState)[0]
            # use exploration function to add exploration bonus
            if self.explorationBonus != 0:
                targetNetQValues = np.array(self.explorationFunction(nextState, targetNetQValues))

            futureQVs=None
            if self.useDoubleDQN: # deal with overestimation of q-values. (https://papers.nips.cc/paper/3964-double-q-learning)
                # use DQN prediction network to determine the action for the nextState
                actionForNextState = self.determineAction(nextState)
                # use DQN target network to calculate the q values of taking the above action
                futureQVs = targetNetQValues[np.array(actionForNextState)]
            else: # Vanila DQN with fixed target network
                # pick the maximum value from the target network
                topK = len(actionTuple)
                topKQVs = targetNetQValues[np.argpartition(targetNetQValues, range(-topK, 0))[-topK:]]
                futureQVs = np.flip(topKQVs)

            preNetQValues[0][np.array(actionTuple)] = reward + self.gamma * futureQVs
            self.predictModel.fit(state, preNetQValues, epochs=1, verbose=0)

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


