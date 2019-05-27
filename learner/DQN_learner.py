import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K



class DQNLearner:
    def __init__(self, DQNAgent, system, resultDir, isDebug = True, memoryInitSize = 10000, optimizationType="min"):
        self.agent = DQNAgent
        self.system = system
        self.optimizationType = optimizationType
        self.memoryInitSize = memoryInitSize
        self.resultDir = resultDir
        self.DQNModelFilepath = os.path.join(resultDir, 'DQN_model.h5')
        self.isDebug = isDebug


    # Input:
    #   config: the input to the DQN. config[0][0] is the list of option values 
    def evaluateConfig(self, config, testSystem):
        sysConfig = config[0][0]
        perf=testSystem.measure(sysConfig)
        return perf

    # Input:
    #   action: list of options to flip. So far, only one option to filp
    def createNewConfig(self, config, action):
        newConfig = [value for value in config[0][0]]
        #assert len(action)==1
        #opID = action[0]
        for opID in action:
            newConfig[opID] = (newConfig[opID]+1)%2
        newConfig=np.expand_dims([newConfig], axis=0)
        return newConfig

    # Use the performance gap to represent the reward
    def rewardCal(self, oldPerf, newPerf):
        if self.optimizationType == "max":
            return round(newPerf - oldPerf, 6)
        else: # minize system performance
            return round(oldPerf - newPerf, 6)

    # [Used in Debug Mode only]
    # Assume the # of actions equals to the # of options
    # Measuing the performnace of the each configuration resulting from one action by flipping one option
    def measureAllActions(self, state, numberOfActions):
        perfsOfEachAction=[]
        for i in range(numberOfActions):
            tempConfig=self.createNewConfig(state, [i])
            perfsOfEachAction.append(self.evaluateConfig(tempConfig, self.system))
        return perfsOfEachAction

    # Calculate the relative performance gap between the optimal action and the selected action
    def calActionError(self, actualPerf, perfsOfEachAction):
        if self.optimizationType == "max":
            bestPerf = max(perfsOfEachAction)
            if bestPerf == 0:
                return -actualPerf
            else:
                return 1.0*(bestPerf-actualPerf)/bestPerf
        else:
            bestPerf = min(perfsOfEachAction)
            if bestPerf == 0:
                return actualPerf
            else:
                return 1.0*(actualPerf-bestPerf)/bestPerf

    # state represents a configuration: state[0][0] is a list of option values
    def transition(self, state, action, testSystem):
        # new configuration is the new state of the DQN agent
        newConfig=self.createNewConfig(state, action)
        newPerf=self.evaluateConfig(newConfig, testSystem)
        return newConfig, newPerf

    # create a random configuration whose dimensionality is (1， 1， self.numOfOptions)
    def getARandomConfig(self):
        config=np.array([])
        for i in range(self.system.numOfOptions):
            config=np.append(config, random.choice([0, 1]))
            
        config=np.expand_dims([config], axis=0)
        return config

    # use testSystem to initialize Replay Memory
    def initializeReplayMemory(self, testSystem):
        memoryCnt = 0
        while memoryCnt < self.memoryInitSize:
            state=self.getARandomConfig()
            perf=self.evaluateConfig(state, testSystem)
            action = random.sample(range(self.agent.actionSize), self.agent.numOfOptionsToChangePerAction)
            nextState, newPerf  = self.transition(state, action, testSystem)
            reward = self.rewardCal(perf, newPerf)
            if self.agent.remember(state, action, reward, nextState):
                memoryCnt += 1 # it is a new experience
                print("[Initialize Replay Memory] SampleID {0:5d} : Reward - {1}.".format(memoryCnt, reward))
       

    # DQN agent learns the good/optimal policies in the system
    def learning(self, totalIters = 1000, iterRatioFirstEpoch = 0.1, iterRatioLastEpoch = 0.05, numEpochs = 50):
        
        # set up learning strategy 
        itersFirstEpoch = int(totalIters*iterRatioFirstEpoch)
        itersLastEpoch = int(totalIters*iterRatioLastEpoch)
        itersPerMidEpoch = int((totalIters - itersFirstEpoch - itersLastEpoch)/(numEpochs-2))
        if itersPerMidEpoch <= 0:
            msg  = "[Error - Invalid Learning Strategy]: itersPerMidEpoch ({}) is not positive.\n".format(itersPerMidEpoch)
            msg += "                                     totalIters-({}), iterRatioFirstEpoch-({}), iterRatioLastEpoch-({}).".format(
                    totalIters, iterRatioFirstEpoch, iterRatioLastEpoch)
            raise Exception(msg)
        ## update the iteractions in the last epoch
        itersLastEpoch = totalIters - itersFirstEpoch - itersPerMidEpoch*(numEpochs-2)

        ## iterations per epoch
        itersPerEpoch = [itersFirstEpoch] + [itersPerMidEpoch]*(numEpochs-2) + [itersLastEpoch]
        print("Iterations Per Epoch: "+str(itersPerEpoch))

        # set learning rate for each epoch
        lrDecay=-1.0*(self.agent.learningRateMax - self.agent.learningRateMin)/(numEpochs-1)
        lrs = np.arange(self.agent.learningRateMax, self.agent.learningRateMin, lrDecay)
        lrs = [round(lr,4) for lr in lrs]
        lrs = lrs + [self.agent.learningRateMin]

        ## set epsilon for each epoch
        epsilonDecayStep = round(-1.0*(self.agent.epsilonMax - self.agent.epsilonMin)/(numEpochs-1), 4)
        epsilons = np.arange(self.agent.epsilonMax, self.agent.epsilonMin, epsilonDecayStep)
        epsilons = [round(ep, 4) for ep in epsilons]
        epsilons = epsilons + [self.agent.epsilonMin]


        if self.isDebug:
            avePerfs=[]
            aveRewards=[]
            allRewards=[]
            # averaged PGPs
            # PGP: percentage of the performance gap over the best-actioned performance in each iteration
            # performance gap: <actually-actioned performance> - <best-actioned performance>
            avePGPs=[]

        allPerfs = []
        usePolicy = False
        # start with a random state (configuration)
        state=self.getARandomConfig()
        perf=self.evaluateConfig(state, self.system)

        # DQN learning process
        for e in range(numEpochs):
            if self.isDebug:
                currentEpochRewards=[]
                sumPGPs = 0

            currentEpochPerfs=[]
            numIters = itersPerEpoch[e]
            self.agent.setEpsilon(epsilons[e])
            lr = lrs[e]
            K.set_value(self.agent.optimizer.lr, lr)
            for i in range(numIters):
                usePolicy, action = self.agent.act(state)
                # Only pick the top action
                #assert len(action)==1
                if self.agent.explorationBonus != 0: # exploration function is enabled
                    self.agent.incrementExplorationCount(state, action)
                nextState, newPerf  = self.transition(state, action, self.system)
                reward = self.rewardCal(perf, newPerf)
                self.agent.remember(state, action, reward, nextState)

                perf = newPerf
                currentEpochPerfs.append(perf)

                if self.isDebug:
                    perfsOfEachAction=self.measureAllActions(state, self.system.numOfOptions)
                    # percent gap between performances of configurations
                    # derived by the best action and the actually selected action    
                    perfGapPercent=self.calActionError(newPerf, perfsOfEachAction)
                    sumPGPs += perfGapPercent

                    print("UsePolicy: {}, Epoch: {}, Iteration: {}, Epsilon: {}, LearningRate: {}, Perf: {}, Reward: {}".format(usePolicy, e, i, epsilons[e], lr, perf, reward))    
                    currentEpochRewards.append(reward)

                state = nextState
                # update the prediction network
                self.agent.replay()

            avePGPs.append(round(sumPGPs/numIters, 4))
            allPerfs.extend(currentEpochPerfs)
            self.agent.updateTargetModel()


            if self.isDebug:
                allRewards.extend(currentEpochRewards)
                with open(os.path.join(self.resultDir, "performence_in_epoch_"+str(e)), "w+") as f:
                    currentAvePerf=0.0
                    for p in currentEpochPerfs:
                        f.write(str(p)+"\n")
                        currentAvePerf+=p
                    currentAvePerf=round(currentAvePerf/len(currentEpochPerfs), 6)
                    avePerfs.append(currentAvePerf)

                with open(os.path.join(self.resultDir, "reward_in_epoch_"+str(e)), "w+") as f:
                    currentAveReward=0.0
                    for r in currentEpochRewards:
                        f.write(str(r)+"\n")
                        currentAveReward+=r
                    currentAveReward=round(currentAveReward/len(currentEpochRewards), 6)
                    aveRewards.append(currentAveReward)


        if self.isDebug:
            # output final state (configuration)
            with open(os.path.join(self.resultDir, "DQN_Learning_Summary.txt"), "w+") as f:
                print("\n===Writing summary===")
                f.write("Epoch,\tEpislon,\tIterations,\tAve Reward, \tAve Perf, \tAve PGP\n")
                print("Epoch,\tEpislon,\tIterations,\tAve Reward, \tAve Perf, \tAve PGP")
                epoch=1
                for epsilon, iters, aveR, avePerf, avePGP in zip(epsilons, itersPerEpoch, aveRewards, avePerfs, avePGPs):
                    f.write(str(epoch)+",\t"+str(epsilon)+",\t"+str(iters)+",\t"+str(aveR)+",\t"+str(avePerf)+",\t"+str(avePGP)+"\n")
                    print(str(epoch)+",\t"+str(epsilon)+",\t"+str(iters)+",\t"+str(aveR)+",\t"+str(avePerf)+",\t"+str(avePGP))
                    epoch+=1
           
            with open(os.path.join(self.resultDir, "allPerfs.txt"), "w+") as f:
                for p in allPerfs:
                    f.write(str(p)+"\n")
            with open(os.path.join(self.resultDir, "allRewards.txt"), "w+") as f:
                for r in allRewards:
                    f.write(str(r)+"\n")

            print("===Ploting the average performance===")
            allIters = np.arange(totalIters) + 1

            expandedAvePerfs=[]
            for p, i in zip(avePerfs, itersPerEpoch):
                expandedAvePerfs.extend([p]*i)
            self.curvePlot2(allIters, allPerfs, allIters, expandedAvePerfs, "Iteration", "Performance of Power Model", os.path.join(self.resultDir, "Ave_Perf_Per_Epoch.png"), ["per_iteration", "per_epoch"], "")

            expandedAveRewards=[]
            for r, i in zip(aveRewards, itersPerEpoch):
                expandedAveRewards.extend([r]*i)
            self.curvePlot2(allIters, allRewards, allIters, expandedAveRewards, "Iteration", "Reward", os.path.join(self.resultDir, "Ave_Reward_Per_Episode.png"), ["per_iteration", "per_epoch"], "")

            self.curvePlot(np.arange(numEpochs)+1, avePGPs, "Epoch", "Ave Perf Gap Percent", os.path.join(self.resultDir, "Ave_Perf_Gap_Percent.png"), "")
            
        # Save the DQN predict model
        self.agent.predictModel.save(self.DQNModelFilepath)
        # Return the performance of all sampled configurations
        return allPerfs


    def curvePlot(self, xs, ys, xlabel, ylabel, outputfile, title):
            print("\t[curvePlot] ploting " + outputfile)

            plt.plot(xs, ys,'-o')
            plt.xlabel(xlabel, fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(outputfile)
            plt.close()
            print("\tploting finished!")


    def curvePlot2(self, x1, y1, x2, y2, xlabel, ylabel, outputfile, linetags, title):
            print("\t[curvePlot] ploting " + outputfile)

            plt.plot(x1, y1, '-.', label=linetags[0])
            plt.plot(x2, y2, '-o', label=linetags[1])
            plt.xlabel(xlabel, fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title)
            plt.legend(loc='upper left', fontsize=18)
            plt.tight_layout()
            plt.savefig(outputfile)
            plt.close()
            print("\tploting finished!")
 
