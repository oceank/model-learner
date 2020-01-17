from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
    
class AcquisitionMU(AcquisitionBase):
    
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    
    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(AcquisitionMU, self).__init__(model, space, optimizer)
        
        # --- UNCOMMENT ONE OF THE TWO NEXT BITS
             
        # 1) THIS ONE IF THE EVALUATION COSTS MAKES SENSE
        #
        # if cost_withGradients == None:
        #     self.cost_withGradients = constant_cost_withGradients
        # else:
        #     self.cost_withGradients = cost_withGradients 

        # 2) THIS ONE IF THE EVALUATION COSTS DOES NOT MAKE SENSE
        #
        # if cost_withGradients == None:
        #     self.cost_withGradients = constant_cost_withGradients
        # else:
        #     print('LBC acquisition does now make sense with cost. Cost set to constant.')  
        #     self.cost_withGradients = constant_cost_withGradients
        if cost_withGradients is not None:
            print('The set cost function is ignored! MU acquisition does not make sense with cost.')  



    def _compute_acq(self,x):
        
        # --- DEFINE YOUR AQUISITION HERE (TO BE MAXIMIZED)
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array  
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the 
        # values of the acquisition at x.
        #
        m, s = self.model.predict(x)   
        f_acqu = s
        return f_acqu

       
    
    def _compute_acq_withGradients(self, x):
        
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array  
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the 
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = s       
        df_acqu = dsdx
       
        return f_acqu, df_acqu
