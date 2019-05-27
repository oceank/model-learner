class powerSystem:
    #   input parameters:
    #       powerModelFilepath: file path of power model
    #       opIDs: a dict that maps the name of an option to its numerical ID, an iteger from 0 to n-1. n is the total number of options
    def __init__(self, powerModelFilepath, opIDs):
        self.numOfOptions = len(opIDs)
        self.opIDs = opIDs
        self.pmfp = powerModelFilepath
        self.powerModel = self.loadPowerModel()
   
    # functionality: load power model from text file
    #   output:
    #       model: a dict where each element is a pair of (key=a polynomal term represented in a tuple, weight)
    #              The constant term       : model[(-1)] = weight1
    #              Single option term      : model[(opID)] = weight2
    #              Interacting options term: model[(opID1, opID2,...)] = weightt3
    def loadPowerModel(self):
        try:
            model={tuple([-1]):0}
            with open(self.pmfp) as f:
                model_txt = f.read(); # there is only one line in the model file
                terms=model_txt.replace(" ", "").rstrip().split("+")
                for term in terms:
                    parts=term.split("*")
                    if len(parts)==1: # It is a constant. Assume there are at most 1 constant
                        model[tuple([-1])] = float(parts[0])
                    else:
                        weight=parts[0]
                        opList=parts[1:]
                        opListNumericalIDs=[]
                        for op in opList:
                             opListNumericalIDs.append(self.opIDs[op])
                        model[tuple(opListNumericalIDs)]=float(weight)
            return model
        except Exception as e:
            raise Exception(e)

    # functionality: get the performance of the power model under the given configuration
    # input:
    #   config: a list of 0 or 1
    # output:
    #   the performance of the power model
    def measure(self, config):
        perf = 0
        for key, value in self.powerModel.items():
            if -1 not in key: # if the key is not the constant term
                for ID in key:
                    if config[ID] == 0:
                        value = 0
                        break
            perf += value
        return perf


