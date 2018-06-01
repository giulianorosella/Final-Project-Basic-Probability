import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
boston = datasets.load_boston()


class LR(object):
    '''An implemententation of a linear regression object, that given features is able to calculate optimal parameters to get the best prediction'''
    def __init__(self, predictors, target, num_parameters = 0, initial_parameters = None):
        '''Constructor
        :param predictors: the matrix of predictors (feature matrix)
        :param num_parameters: the number of parameters (theta_0, theta_1, ... , theta_n)
        :param initial_parameters: the initial parameters if provided
        '''

        # total number of parameters
        self.num_parameters = num_parameters
        # parameters setting
        self.parameters = np.zeros(shape=(self.num_parameters,1))
        # matrix of predictors
        self.predictors = predictors
        # matrix with the actual values in it
        self.target = target
        # list that keeps track of the cost function on every iteration
        self.cost_history = []
        if initial_parameters.all():
            self.initialize(initial_parameters)

    def initialize(self, initial_parameters = None):
        '''Sets up the initial parameters that were given.
        :param initial_parameters that the user want this program to use.
        '''
        for param in range(self.num_parameters):
            self.parameters[param,0] = initial_parameters[param]



    def improve(self, step_size = 0.001, improvement = 0.0001):
        ''' improvement functions, does gradient descent until the stop condition is reached
        :param step_size: the step size used in the gradient descent
        :param improvement: we stop improving if the difference of two subsequent cost-function-values is below the improvement rate
        '''
        self.gradient_descent(step_size)
        self.gradient_descent(step_size)
        print(self.cost_history)
        i=1
        while self.cost_history[i-1]-self.cost_history[i] >= improvement:
            self.gradient_descent(step_size)
            i +=1
            print(self.cost_history[i-1] - self.cost_history[i])



    def gradient_descent(self, step_size):
        ''' Gradient descent function. Updates the parameters of the object, using the temporary parameters temp_parameters
        :param step_size: step_size used in the gradient descent
        '''
        temp_parameters= np.zeros(shape=(self.num_parameters,1))
        for param in range(self.num_parameters):
            temp_parameters[param,0] = self.parameters[param,0]

        for param in range(self.num_parameters):
            temp1=np.transpose(self.predictors.dot(temp_parameters) - self.target).dot(self.predictors[:,param])
            self.parameters[param,0] = temp_parameters[param,0] - step_size * (1/len(self.target))*temp1[0]
        #update the ccst history
        self.cost_history.append(self.cost_function())

    def cost_function(self):
        '''
        :return: the value of the cost function for the current parameters of the object.
        '''
        cost = 0
        pred = self.prediction()
        for i in range(len(self.target)):
            cost += (pred[i,0] - self.target[i])**2
        cost = cost/(2*len(self.target))
        return cost

    def test_step_size(self, step_size=0.001, iterations = 50):
        ''' This method helps us choose a proper step size. For a given step size, we can perform gradient descent [iterations] times
        This return two lists, which we then can plot, to see if the cost-function behaves properly (and thus the step_size is appropriate
        :param step_size: step_size used in gradient descent
        :param iterations: number of iterations for which we want to test the cost function
        :return: x_axis, list of the range of our iterations
        :return: y_axis, values of the cost_function per iteration
        '''
        initial_param = self.parameters
        for i in range(iterations):
            self.gradient_descent(step_size)

        y_axis = self.cost_history
        x_axis = list(range(len(y_axis)))

        self.parameters = initial_param
        self.cost_history = []
        return x_axis,y_axis

    def prediction(self):
        ''' calculates y = theta_0 + theta_1 x_1 + ... for given theta and given features
        :return: pred, the prediction of the median housing values given our theta and features
        '''
        pred = self.predictors.dot(self.parameters)
        return pred

    def error_measure(self):
        ''' This method calculates R^2. Observe that the numerator is the cost function times 2*m
        :return: R, the error_measure we use!
        '''
        mean = np.mean(self.target)
        numerator = self.cost_function()*(2*len(self.target))
        denominator = 0
        for i in range(len(self.target)):
            denominator += (self.target[i] - mean)**2

        R = 1 - (numerator/denominator)
        return R


def feature_matrix(feat = boston.data, list_of_pred = [],num_observations = 506):
    ''' this method makes a matrix of features we want to use. It starts with a column of only ones, then adds the feature column we want to use
    :param feat: A matrix of features from which we take colums to make the feature matrix our model will use
    :param list_of_pred: In this list are the predictors our model want to use. For instance [12], means we only use LSTAT as predictor
    :param num_observations: The number of rows in the feature/target matrix
    :return: features: the matrix our model will use
    '''
    features = np.reshape(np.ones(num_observations),(num_observations,1))
    for pred in list_of_pred:
        features = np.c_[features,feat[:,pred]]

    return features

def powered_feature_matrix(feat = boston.data, list_of_pred = [0],power = 2,num_observations = 506):
    ''' This method makes a matrix used for polynomial regression.
    :param feat: A matrix of features from which we take colums to make the feature matrix our model will use
    :param list_of_pred: In this list are the predictors our model want to use. For instance [12], means we only use LSTAT as predictor
    :param power: the power of our polynomial: y = theta_0 + theta_1 x_1 + theta_2 x_1^2 + ... + theta_n x_1^n
    :param num_observations: The number of rows in the feature/target matrix
    :return: power_matrix: the matrix our model will use
    '''
    power_matrix = feature_matrix(feat,list_of_pred,num_observations)
    for i in range(power-1):
        power_matrix = np.c_[power_matrix,power_matrix[:,1]**(i+2)]

    return power_matrix

def erase_corrupted_values(feat, tar, num_observations = 506):
    ''' This method deletes the values of 50 in the target and the corresponding features in the feature matrix.
    :param feat: A matrix of features from which we want to delete the (corrupted) features
    :param tar: The target matrix of which we want to delete the target with value 50
    :param num_observations: The number of rows in the feature/target matrix
    :return: feat, tar: the feature and target matrix, with corrupted values removed.
    '''
    for i in range(num_observations):
        if tar[num_observations - 1 - i] == 50:
            tar = np.delete(tar, num_observations - 1 - i, 0)
            feat = np.delete(feat, num_observations - 1 - i, 0)

    return feat, tar

def standardization(feat):
    ''' This method standardizes all the values in the feature matrix. In every column, it substracts the mean of the column from every value
    and divides by the standard deviation
    :param feat: A matrix of features which we want to standardize
    :return: feat: A standardized matrix
    '''
    for i in range(np.shape(feat)[1]):
        if i>0:
            feat[:,i] = (feat[:,i] - np.mean(feat[:,i]))/np.std(feat[:,i])
    return feat






def main(features, target,num_parameters=1,initial_parameters = None, initial_step_size = 0.001, initial_improvement = 0.0001):
    ''' The main method makes a LR-object en tells it to improve. Also code to plot our results is in this method.
    :param features: the feature matrix
    :param target:  the target matrix
    :param num_parameters: the number of parameters
    :param initial_parameters: the initial parameters
    :param initial_step_size: the initial step size
    :param initial_improvement: the initial improvement measure
    '''

    learner = LR(features,target,num_parameters,initial_parameters)

    step_size = initial_step_size
    improvement = initial_improvement

    #unhash to test the step size
    #x,y=learner.test_step_size(step_size)
    #plt.plot(x,y)
    #plt.show()

    #improve the parameters, calculate the error measure, calculate the prediction
    learner.improve(step_size,improvement)
    R = learner.error_measure()
    prediction = learner.prediction()

    print("Parameters are: " + str(learner.parameters))
    print("R = " + str(R))

    #for baseline, use:
    #plt.title("R="+str(R)+"   "+"step-size="+str(step_size))
    #plt.ylabel("Predicted values (blue)- Target values (red)")
    #plt.xlabel("Target values")
    #plt.plot(list(features[:, 0]), target,'ro')
    #plt.plot(list(features[:,0]),prediction,'bs')
    #plt.show()

    # for one feature, or the polynomial use these ones:
    #plt.title("R=" + str(R) + "   " + "step-size=" + str(step_size))
    #plt.ylabel("Predicted values (blue)- Target values (red)")
    #plt.xlabel("Feature values")
    #plt.plot(list(features[:, 1]), target, 'ro')
    #plt.plot(list(features[:, 1]), prediction, 'bs')
    #plt.show()

    #for more than one feature: use this one
    plt.title("R="+str(R)+"   "+"step-size="+str(step_size))
    plt.ylabel("Predicted values")
    plt.xlabel("Target values")
    plt.plot(target,prediction,'bo')
    plt.show()

    #plot of the cost_function
    plt.plot(range(len(learner.cost_history)),learner.cost_history,'bs')
    plt.show()



if __name__ == "__main__":
    #our starting stuff. The whole dataset, separated in a features and target values part
    feature_matr = boston.data
    target = np.reshape(boston.target, (506, 1))

    #If we don't want to take into consideration the cases where MEDV = 50
    feature_matr, target = erase_corrupted_values(feature_matr,target)

    num_of_observations = len(target)

    #List here which features we want to test


    #If we want to try polynomial regression on only one feature
    #list_of_predictors = [1]
    #power = 3
    #features = powered_feature_matrix(feature_matr,list_of_predictors,power, num_of_observations)

    #If we want to test on one or several features
    list_of_predictors = [0,1,2,3,4,5,6,7,8]
    features = feature_matrix(feature_matr,list_of_predictors,num_of_observations)

    #If we want to standardize, uncomment (scaling)
    features = standardization(features)



    number_of_parameters = np.shape(features)[1]
    initial_parameters = np.zeros((number_of_parameters,))

    step_size = 0.3
    improvement = 0.0001


    main(features,target,number_of_parameters,initial_parameters,step_size,improvement)













