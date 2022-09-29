import comp140_module3 as stocks
from collections import defaultdict
import random

def my_helper_probability(list1):
    """
    Computes the probability of each item in the list
    Input: a list of integers 
    Output: a dictionary that maps the probability of each item to the item
    """
    #Use for loop to calculate the probability of each individual item in the list
    new_dict = defaultdict(int)
    
    for item1 in list1:
        new_dict[item1] += 1/(len(list1))
        
    return dict(new_dict)

def markov_chain(data, order):
    """
    Create a Markov chain with the given order from the given data.

    inputs:
        - data: a list of ints or floats representing previously collected data
        - order: an integer repesenting the desired order of the markov chain

    returns: a dictionary that represents the Markov chain
    """
    markov_chain1 = {}
    
    for item in range(len(data)-order):
        keys = []
        
        for item2 in range (item, item + order):
            keys.append(data[item2])
            key = tuple(keys)
        #Conditional to check weather the given bin is in the markov chain
        #If in it, it is mapped, else, new key is created

        if key in markov_chain1:
            markov_chain1[key].append(data[item + order])
            
        else:
            markov_chain1[key] = [data[item + order]]
            
    for key in markov_chain1:
        markov_chain1[key] = my_helper_probability(markov_chain1[key])
        
    return markov_chain1


### Predict

def predict(model, last, num):
    """
    Predict the next num values given the model and the last values.

    inputs:
        - model: a dictionary representing a Markov chain
        - last: a list (with length of the order of the Markov chain)
                representing the previous states
        - num: an integer representing the number of desired future states

    returns: a list of integers that are the next num states
    """
    #Now, we predict the future values and put them into pred_vals
    pred_vals = []

    for item in range(num):

        tuple1 = tuple(last)
        val_dist = random.uniform(0,1)
        
        if tuple1 in model:
            probability = 0
            
            for key_val in model[tuple1]:
                
                if probability <= val_dist and probability < probability + model[tuple1][key_val]:
                    num2 = key_val
                probability = probability + model[tuple1][key_val]
        
        else:
            probability_dict = {0:0.25, 1:0.5, 2:0.75, 3:1}
            probability = 0
            
            for key_val, values in probability_dict.items():
                
                if probability <= val_dist < probability + values:
                    num2 = key_val
                    probability = probability + values
                    
                   
        pred_vals.append(num2)
        last = list(tuple1)
        last.remove(last[0])
        last.append(num2)
        item = item + 1
        
        
    return pred_vals


def mse(result, expected):
    """
    Calculate the mean squared error between two data sets.

    The length of the inputs, result and expected, must be the same.

    inputs:
        - result: a list of integers or floats representing the actual output
        - expected: a list of integers or floats representing the predicted output

    returns: a float that is the mean squared error between the two data sets
    """
    #Find the mean squared error by summing up the difference between
    #expected and result values and dividing by total number
    
    mserror = 0

    for result_val, expected_val in zip(result, expected):
        
        mserror = mserror + (result_val - expected_val)**2
        regerr = mserror/(len(result))
        
    return regerr
    
    
def run_experiment(train, order, test, future, actual, trials):
    """
    Run an experiment to predict the future of the test
    data given the training data.

    inputs:
        - train: a list of integers representing past stock price data
        - order: an integer representing the order of the markov chain
                 that will be used
        - test: a list of integers of length "order" representing past
                stock price data (different time period than "train")
        - future: an integer representing the number of future days to
                  predict
        - actual: a list representing the actual results for the next
                  "future" days
        - trials: an integer representing the number of trials to run

    returns: a float that is the mean squared error over the number of trials
    """
    mean_square_error = 0

    for item in range(trials):
        model_val = markov_chain(train, order)
        res = predict(model_val, test, future)
        mean_square_error = mean_square_error + mse(res, actual)
        item = item + 1
    
    err_per_trial = mean_square_error/trials
    
    return err_per_trial


### Application

def run():
    """
    Run application.

    You do not need to modify any code in this function.  You should
    feel free to look it over and understand it, though.
    """
    # Get the supported stock symbols
    symbols = stocks.get_supported_symbols()

    # Get stock data and process it

    # Training data
    changes = {}
    bins = {}
    for symbol in symbols:
        prices = stocks.get_historical_prices(symbol)
        changes[symbol] = stocks.compute_daily_change(prices)
        bins[symbol] = stocks.bin_daily_changes(changes[symbol])

    # Test data
    testchanges = {}
    testbins = {}
    for symbol in symbols:
        testprices = stocks.get_test_prices(symbol)
        testchanges[symbol] = stocks.compute_daily_change(testprices)
        testbins[symbol] = stocks.bin_daily_changes(testchanges[symbol])

    # Display data
    #   Comment these 2 lines out if you don't want to see the plots
    stocks.plot_daily_change(changes)
    stocks.plot_bin_histogram(bins)

    # Run experiments
    orders = [1, 3, 5, 7, 9]
    ntrials = 500
    days = 5

    for symbol in symbols:
        print(symbol)
        print("====")
        print("Actual:", testbins[symbol][-days:])
        for order in orders:
            error = run_experiment(bins[symbol], order,
                                   testbins[symbol][-order-days:-days], days,
                                   testbins[symbol][-days:], ntrials)
            print("Order", order, ":", error)
        print()
