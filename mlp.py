import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from PIL import Image

from logistic_sgd import LogisticRegression
from load_data import load_umontreal_data, load_mnist

def ReLU(x):
    y = T.maximum(0.0, x)
    return y

def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return y

def Tanh(x):
    y = T.tanh(x)
    return y
    
def Average(x):
    size = len(x)
    y = 0
    for i in range(0,size):
        y += x[i]
    return y/size
    
def Geometric_mean(x):
    size = len(x)
    y = 1
    for i in range(0,size):
        y *= x[i]
    return pow(y,-size)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):    

        self.input = input
        self.activation = activation
        

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX                
            )
            if activation == Sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True,
            params=None,
            temperature=1.0,
            alpha=1.0):

        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            W = None
            b = None
            if params is not None:
                i = layer_counter * 2 if use_bias is True else layer_counter
                j = layer_counter * 2 + 1 if use_bias is True else None
                W = params[i]
                b = params[j] if use_bias is True else None
                
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias, W=W, b=b,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_counter += 1
    
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out, 
                W = params[-2] if params is not None else None, 
                b = params[-1] if params is not None else None,
                temperature=temperature,
                alpha=alpha)
        self.dropout_layers.append(dropout_output_layer)

        output_layer = LogisticRegression(
            input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out, temperature=temperature, alpha=alpha)
        self.layers.append(output_layer)

        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors
        self.dark_dropout_negative_log_likelihood = self.dropout_layers[-1].dark_negative_log_likelihood
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.dark_negative_log_likelihood = self.layers[-1].dark_negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.results = self.layers[-1].results
        self.outputs = self.layers[-1].outputs
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]        

def write_on_file_and_console(text, results_file):
    print(text)
    results_file.write(text)
    results_file.write('\n')
    results_file.flush()

def elastic_distortion(input, x):
    image = Image.fromarray(np.array(np.split(np.array(input)*255,28)))
    image = image.rotate(np.random.randint(-30,30))
    x = (x % 10)*2+10
    image = image.transform((x,x), Image.EXTENT, [np.random.randint(0, 4), np.random.randint(0, 4), 28, 28])
    image = image.resize((28,28))
    return np.array(list(image.getdata()))/255
    
def create_distortion_dataset():
    dataset = 'data/mnist_batches.npz'
    
    mnist = np.load(dataset)
    train_set_x = mnist['train_data']
    mnist = None
    for i in xrange(len(train_set_x)):
        train_set_x[i] = elastic_distortion(input=train_set_x[i], x=np.random.choice(9, 1, p=[0.03, 0.03, 0.04, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])[0])
    
    return train_set_x

def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        n_matches,
        patient,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        datasets,
        use_bias,
        random_seed=1234):
    assert len(layer_sizes) - 1 == len(dropout_rates)

    results_file_train = open("train" + results_file_name, 'wb')
    results_file_validation = open("validation" + results_file_name, 'wb')
    results_file_test = open("test" + results_file_name, 'wb')
    
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    rng = np.random.RandomState(random_seed)
                                  
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################

    write_on_file_and_console('... building the model',results_file_train)

    index = T.lscalar()
    epoch = T.scalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    learning_rate = theano.shared(np.asarray(initial_learning_rate,dtype=theano.config.floatX))        

    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias,)

    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
                
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    gparams = []
    for param in classifier.params:
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    mom = ifelse(epoch < mom_epoch_interval,mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),mom_end)

    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param + updates[gparam_mom]
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param

    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=[classifier.errors(y), output],
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############

    write_on_file_and_console('... training',results_file_train)

    best_params = None
    this_train_score = 60000.
    this_train_errors = 60000
    best_validation_score = this_validation_score = 10000.
    best_validation_errors = this_validation_errors = 10000
    best_test_score = this_test_score = 0.
    best_test_errors = this_test_errors = 0
    best_iter = 0
    epoch_counter = 0
    start_time = time.clock()

    match = 1
    best_match = 0
    original_epochs = n_epochs
    total_epochs = 0

    while match <= n_matches:
        epoch_counter = 0
        n_epochs = original_epochs
        
        while epoch_counter < n_epochs and this_validation_errors > 0:
            train_set_x_with_distortion = theano.shared(np.asarray(create_distortion_dataset(),dtype=theano.config.floatX))
            train_model_distortion = theano.function(inputs=[epoch, index], outputs=[classifier.errors(y), output],
                    updates=updates,
                    givens={
                        x: train_set_x_with_distortion[index * batch_size:(index + 1) * batch_size],
                        y: train_set_y[index * batch_size:(index + 1) * batch_size]})
                
            epoch_counter = epoch_counter + 1
            train_distort_output = [train_model_distortion(epoch_counter, i) for i in xrange(1, (n_train_batches/n_matches)*match)]
            train_output = [train_model(epoch_counter, i) for i in xrange(1, (n_train_batches/n_matches)*match)]
            
            train_distort_losses = [train_distort_output[i][0] for i in xrange(len(train_distort_output))]
            train_distort_cost = [train_distort_output[i][1] for i in xrange(len(train_distort_output))]
            train_losses = [train_output[i][0] for i in xrange(len(train_output))]
            train_cost = [train_output[i][1] for i in xrange(len(train_output))]
            
            this_train_cost = np.sum((train_cost, train_distort_cost))
            this_train_errors = np.sum((train_losses, train_distort_losses))
            this_train_score = np.mean((train_losses, train_distort_losses))

            write_on_file_and_console("match {} epoch {}, train cost {}".format(
                    match, epoch_counter, this_train_cost),results_file_train)
            write_on_file_and_console("match {} epoch {}, train error {}, score {}%, lr {}".format(
                    match, epoch_counter, this_train_errors, this_train_score,
                    learning_rate.get_value(borrow=True)),results_file_train)

            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_errors = int(np.sum(validation_losses))
            this_validation_score = np.mean(validation_losses)

            write_on_file_and_console("match {} epoch {}, validation error {}, score {}%, lr {}{}".format(
                    match, epoch_counter, this_validation_errors, this_validation_score,
                    learning_rate.get_value(borrow=True),
                    " **" if this_validation_errors < best_validation_errors else ""),results_file_validation)

            if this_validation_errors < best_validation_errors:
                best_params = classifier.params
                best_iter = epoch_counter
                best_match = match
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                best_test_errors = this_test_errors = int(np.sum(test_losses))
                best_test_score = this_test_score = np.mean(test_losses)
                n_epochs = max(n_epochs, epoch_counter+patient)
                write_on_file_and_console("match {} epoch {}, test error {}, score {}%, lr {}".format(
                    match, epoch_counter, this_test_errors, this_test_score,
                    learning_rate.get_value(borrow=True)),results_file_test)

            best_validation_errors = min(best_validation_errors, this_validation_errors)
            best_validation_score = min(best_validation_score, this_validation_score)
            decay_learning_rate()
            total_epochs += 1

            if this_train_errors <= 10 and epoch_counter >= n_epochs/2 and match < n_matches:
                epoch_counter = n_epochs
            if this_train_errors == 0 and match < n_matches:
                epoch_counter = n_epochs

        match+=1

    end_time = time.clock()
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_train)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_train)
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_validation)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_validation)
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_test)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_test)
                          
    return best_params
    
def test_committee_machine(
        batch_size,        
        datasets,
        networks,        
        results_file_name,
        use_bias=True,
        random_seed=1234):
    results_file = open(results_file_name, 'wb')
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    rng = np.random.RandomState(random_seed)
    
    #########################################
    # GRAB RESULTS FROM MODELS ACTUAL MODEL #
    #########################################

    write_on_file_and_console('...Grabing results from models',results_file)

    classifiers = []
    train_models = []
    train_results = []
    test_models = []
    test_results = []
    validation_models = []
    validation_results = []
    model_counter = 0
    
    for params, layer_sizes, dropout_rates, activations in networks:
        assert len(layer_sizes) - 1 == len(dropout_rates)  
        model_counter += 1
        
        index = T.lscalar()
        x = T.matrix('x')  
        y = T.ivector('y')

        validation_errors = test_errors = 10000
        validation_score = test_score = 0.
        
        classifier = MLP(rng=rng, 
                         input=x,
                         layer_sizes=layer_sizes,
                         dropout_rates=dropout_rates,
                         activations=activations,
                         use_bias=use_bias,
                         params=params)

        train_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
                         
        validate_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
                
        test_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

        train_model = theano.function(inputs=[index],
            outputs=classifier.results(),
            givens={ x: train_set_x[index * batch_size:(index + 1) * batch_size]} )
                
        validation_model = theano.function(inputs=[index],
            outputs=classifier.results(),
            givens={ x: valid_set_x[index * batch_size:(index + 1) * batch_size]} )
                
        test_model = theano.function(inputs=[index],
            outputs=classifier.results(),
            givens={ x: test_set_x[index * batch_size:(index + 1) * batch_size]} )                
                
        train_losses = [train_error_model(i) for i in xrange(n_train_batches)]
        validation_losses = [validate_error_model(i) for i in xrange(n_valid_batches)]
        test_losses = [test_error_model(i) for i in xrange(n_test_batches)]
        train_errors = int(np.sum(train_losses))
        train_score = train_errors/60000.*100.
        validation_errors = np.sum(validation_losses)
        validation_score = np.mean(validation_losses)
        test_errors = np.sum(test_losses)
        test_score = np.mean(test_losses)

        write_on_file_and_console("Model {}, Train error {}, train error {}%".format(
            model_counter, train_errors, train_score),results_file)
        write_on_file_and_console("Model {}, Validation error {}, validation error {}%".format(
            model_counter, validation_errors, validation_score),results_file)
        write_on_file_and_console("Model {}, Test error {}, test error {}%".format(
            model_counter, test_errors, test_score),results_file)
                         
        train_result = np.concatenate([train_model(i) for i in xrange(n_train_batches)])
        validation_result = np.concatenate([validation_model(i) for i in xrange(n_valid_batches)])
        test_result = np.concatenate([test_model(i) for i in xrange(n_test_batches)])
                         
        classifiers.append(classifier)
        train_models.append(train_model)
        train_results.append(train_result)
        test_models.append(test_model)
        test_results.append(test_result)
        validation_models.append(validation_model)
        validation_results.append(validation_result)
        
    ########################
    # CREATE FINAL RESULTS #
    ########################
    
    write_on_file_and_console('...Create final results from committee',results_file)    
    
    temp_train_result = np.vstack(train_results).T
    temp_validation_result = np.vstack(validation_results).T
    temp_test_result = np.vstack(test_results).T
    final_train_result = []
    final_validation_result = []
    final_test_result = []
    
    for i in xrange(0, len(temp_train_result)):
        final_train_result.append(np.argmax(np.bincount(temp_train_result[i].astype(np.int32))))
        
    for i in xrange(0,len(temp_validation_result)):        
        final_validation_result.append(np.argmax(np.bincount(temp_validation_result[i].astype(np.int32))))
        
    for i in xrange(0, len(temp_test_result)):
        final_test_result.append(np.argmax(np.bincount(temp_test_result[i].astype(np.int32))))
        
    final_train_result = np.array(final_train_result)
    final_validation_result = np.array(final_validation_result)
    final_test_result = np.array(final_test_result)
        
    #############################
    # EVALUATE COMMITTEE RESULT #
    #############################
        
    write_on_file_and_console('...Evaluate committee',results_file)
    
    real_train_set_y = train_set_y.eval()
    real_valid_set_y = valid_set_y.eval()
    real_test_set_y = test_set_y.eval()
    
    committee_train_errors = T.sum(T.neq(final_train_result, real_train_set_y))
    committee_validation_errors = T.sum(T.neq(final_validation_result, real_valid_set_y))
    committee_test_errors = T.sum(T.neq(final_test_result, real_test_set_y))
    committee_train_score = T.mean(T.neq(final_train_result, real_train_set_y))
    committee_validation_score = T.mean(T.neq(final_validation_result, real_valid_set_y))
    committee_test_score = T.mean(T.neq(final_test_result, real_test_set_y))
    
    write_on_file_and_console("Train errors {}, Train score {}%".format(
        committee_train_errors.eval(), committee_train_score.eval() * 100), results_file)
    write_on_file_and_console("Validate errors {}, Validate score {}%".format(
        committee_validation_errors.eval(), committee_validation_score.eval() * 100), results_file)
    write_on_file_and_console("Test errors {}, Test score {}%".format(
        committee_test_errors.eval(), committee_test_score.eval() * 100), results_file)
        
    write_on_file_and_console('... end',results_file)

    return committee_test_errors.eval()
    
def test_dark_knowledge(
        networks,
        mean,
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        n_matches,
        patient,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        datasets,
        use_bias,
        temperature,
        alpha,
        random_seed=1234):

    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    results_file_train = open("train" + results_file_name, 'wb')
    results_file_validation = open("validation" + results_file_name, 'wb')
    results_file_test = open("test" + results_file_name, 'wb')
    
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    rng = np.random.RandomState(random_seed)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    #################################################
    # GRABING DARK KNOWLEDGE FROM PREVIOUS NETWORKS #
    #################################################
    
    write_on_file_and_console('...Grabing dark knowledge from previous networks',results_file_train)
    
    classifiers = []
    train_models = []
    train_outputs = []
    model_counter = 0
    
    for params, layer_sizes, dropout_rates, activations in networks:
        assert len(layer_sizes) - 1 == len(dropout_rates)
        model_counter += 1
        
        index = T.lscalar()
        x = T.matrix('x')  
        y = T.ivector('y')
        
        classifier = MLP(rng=rng, input=x,
                         layer_sizes=layer_sizes,
                         dropout_rates=dropout_rates,
                         activations=activations,
                         use_bias=use_bias,
                         params=params)
        
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        
        output = dropout_cost if dropout else cost
        train_model = theano.function(inputs=[index], 
            outputs=classifier.outputs(),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size]})

        train_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
                
        validate_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
                
        test_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

        train_losses = [train_error_model(i) for i in xrange(n_valid_batches)]
        train_errors = int(np.sum(train_losses))
        train_score = np.mean(train_losses)
        write_on_file_and_console("Model {}, Train error {}, validation error {}%".format(
            model_counter, train_errors, train_score),results_file_train)
        write_on_file_and_console("Model {}, Train error {}, validation error {}%".format(
            model_counter, train_errors, train_score),results_file_validation)
        write_on_file_and_console("Model {}, Train error {}, validation error {}%".format(
            model_counter, train_errors, train_score),results_file_test)
                
        validation_losses = [validate_error_model(i) for i in xrange(n_valid_batches)]
        validation_errors = int(np.sum(validation_losses))
        validation_score = np.mean(validation_losses)
        write_on_file_and_console("Model {}, Validation error {}, validation error {}%".format(
            model_counter, validation_errors, validation_score),results_file_train)
        write_on_file_and_console("Model {}, Validation error {}, validation error {}%".format(
            model_counter, validation_errors, validation_score),results_file_validation)
        write_on_file_and_console("Model {}, Validation error {}, validation error {}%".format(
            model_counter, validation_errors, validation_score),results_file_test)
        
        test_losses = [test_error_model(i) for i in xrange(n_test_batches)]
        test_errors = int(np.sum(test_losses))
        test_score = np.mean(test_losses)
        write_on_file_and_console("Model {}, Test error {}, test error {}%".format(
            model_counter, test_errors, test_score),results_file_train)
        write_on_file_and_console("Model {}, Test error {}, test error {}%".format(
            model_counter, test_errors, test_score),results_file_validation)
        write_on_file_and_console("Model {}, Test error {}, test error {}%".format(
            model_counter, test_errors, test_score),results_file_test)
                         
        train_output = np.concatenate([train_model(i) for i in xrange(n_train_batches)])
                         
        classifiers.append(classifier)
        train_models.append(train_model)
        train_outputs.append(train_output)

    write_on_file_and_console('...Moving Dark Knowledge from CPU to GPU',results_file_validation)
        
    final_output = []
    if model_counter > 0:
        for i in range(0,len(train_outputs[0])):
            outputs = []
            for output in train_outputs:
                outputs.append(output[i])
            outputs = np.dstack(outputs)[0]
            step_final_output = []
            for j in range(0,len(outputs)):
                step_final_output.append(mean(outputs[j]))
            final_output.append(step_final_output)
        final_output = np.vstack(final_output)
        
        final_output = theano.shared(final_output)
        
    ######################
    # BUILD ACTUAL MODEL #
    ######################

    write_on_file_and_console('... building the model',results_file_validation)

    index = T.lscalar()
    epoch = T.scalar()
    x = T.matrix('x')
    y = T.ivector('y')
    dark_y = T.dmatrix('dark_y')
    
    learning_rate = theano.shared(np.asarray(initial_learning_rate,dtype=theano.config.floatX))

    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias,
                     temperature=temperature,
                     alpha=alpha)

    cost = classifier.dark_negative_log_likelihood(dark_y, y) if model_counter > 0 else classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dark_dropout_negative_log_likelihood(dark_y, y) if model_counter > 0 else classifier.dropout_negative_log_likelihood(y)

    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    
    gparams = []
    for param in classifier.params:
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    mom = ifelse(epoch < mom_epoch_interval,mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),mom_end)

    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param + updates[gparam_mom]
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param

    output = dropout_cost if dropout else cost
    train_model = None
    if model_counter > 0:
        train_model = theano.function(inputs=[epoch, index], outputs=[classifier.errors(y), output],
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size],
                    dark_y: final_output[index * batch_size:(index + 1) * batch_size]})
    else:
        train_model = theano.function(inputs=[epoch, index], outputs=[classifier.errors(y), output],
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############

    write_on_file_and_console('... training',results_file_validation)

    best_params = None
    this_train_score = 60000.
    this_train_errors = 60000
    best_validation_score = this_validation_score = 10000.
    best_validation_errors = this_validation_errors = 10000
    best_test_score = this_test_score = 0.
    best_test_errors = this_test_errors = 0
    best_iter = 0
    epoch_counter = 0
    start_time = time.clock()

    match = 1
    best_match = 0
    original_epochs = n_epochs
    total_epochs = 0

    while match <= n_matches:
        epoch_counter = 0
        n_epochs = original_epochs
        
        while epoch_counter < n_epochs and this_validation_errors > 0:
            epoch_counter = epoch_counter + 1
            train_output = [train_model(epoch_counter, i) for i in xrange(1, (n_train_batches/n_matches)*match)]
            
            train_losses = [train_output[i][0] for i in xrange(len(train_output))]
            train_cost = [train_output[i][1] for i in xrange(len(train_output))]
            
            this_train_cost = np.sum(train_cost)
            this_train_errors = np.sum(train_losses)
            this_train_score = np.mean(train_losses)

            write_on_file_and_console("match {} epoch {}, train cost {}".format(
                    match, epoch_counter, this_train_cost),results_file_train)
            write_on_file_and_console("match {} epoch {}, train error {}, score {}%, lr {}".format(
                    match, epoch_counter, this_train_errors, this_train_score,
                    learning_rate.get_value(borrow=True)),results_file_train)

            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_errors = int(np.sum(validation_losses))
            this_validation_score = np.mean(validation_losses)

            write_on_file_and_console("match {} epoch {}, validation error {}, score {}%, lr {}{}".format(
                    match, epoch_counter, this_validation_errors, this_validation_score,
                    learning_rate.get_value(borrow=True),
                    " **" if this_validation_errors < best_validation_errors else ""),results_file_validation)

            if this_validation_errors < best_validation_errors:
                best_params = classifier.params
                best_iter = epoch_counter
                best_match = match
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                best_test_errors = this_test_errors = int(np.sum(test_losses))
                best_test_score = this_test_score = np.mean(test_losses)
                n_epochs = max(n_epochs, epoch_counter+patient)
                write_on_file_and_console("match {} epoch {}, test error {}, score {}%, lr {}".format(
                    match, epoch_counter, this_test_errors, this_test_score,
                    learning_rate.get_value(borrow=True)),results_file_test)

            best_validation_errors = min(best_validation_errors, this_validation_errors)
            best_validation_score = min(best_validation_score, this_validation_score)
            decay_learning_rate()
            total_epochs += 1

            if this_train_errors <= 10 and epoch_counter >= n_epochs/2 and match < n_matches:
                epoch_counter = n_epochs
            if this_train_errors == 0 and match < n_matches:
                epoch_counter = n_epochs

        match+=1

    end_time = time.clock()
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_train)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_train)
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_validation)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_validation)
    write_on_file_and_console(('Optimization complete with %i iterations. Best validation score of %f %% '
           'obtained at iteration %i - %i, with test score %f %%') %
          (total_epochs, best_validation_score, best_match, best_iter, best_test_score), results_file_test)
    write_on_file_and_console('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.), results_file_test)
                          
    return best_params
    
if __name__ == '__main__':
    errors = 10000
    networks = []
    names = []
    attempts = 0

    while attempts < 21:
        initial_learning_rate = 1.0
        
        learning_rate_decay = 0.999
        squared_filter_length_limit = 15.0
        n_epochs = 200
        n_matches = 1
        patient = n_epochs*0.75
        batch_size = 100

        layer_sizes = [ 28*28, 1200, 1200, 10 ]
        dropout_rates = [ 0.2, 0.5, 0.5 ]
        activations = [ ReLU, ReLU ]
        dropout = True
        use_bias = True
        mom_start = 0.5
        mom_end = 0.99
        mom_epoch_interval = 500
        mom_params = {"start": mom_start,
                      "end": mom_end,
                      "interval": mom_epoch_interval}                  
        dataset = 'data/mnist_batches.npz'
        datasets = load_mnist(dataset)
        random_seed = np.random.randint(1,9999)
        
        attempts += 1

        results_file_name = str(attempts) + ".txt"
        print results_file_name
        names.append(results_file_name)
        
        params = test_mlp(initial_learning_rate=initial_learning_rate,
                 learning_rate_decay=learning_rate_decay,
                 squared_filter_length_limit=squared_filter_length_limit,
                 n_epochs=n_epochs,
                 n_matches=n_matches,
                 patient=patient,
                 batch_size=batch_size,
                 layer_sizes=layer_sizes,
                 mom_params=mom_params,
                 activations=activations,
                 dropout=dropout,
                 dropout_rates=dropout_rates,
                 datasets=datasets,
                 results_file_name=results_file_name,
                 use_bias=use_bias,
                 random_seed=random_seed)
        
        network = [params, layer_sizes, dropout_rates, activations]
        networks.append(network)

        results_file_name = "Committee" + str(attempts) + ".txt"
        names.append(results_file_name)
        errors = test_committee_machine(
            batch_size=batch_size,        
            datasets=datasets,
            networks=networks,        
            results_file_name=results_file_name,
            use_bias=use_bias,
            random_seed=random_seed)

        results_file = open('resultsFinal.txt', 'wb')
        for name in names:
            write_on_file_and_console(name,results_file)
    
        
        temperature = ((attempts % 10) + 1)
        alpha = 0.9
        
        layer_sizes = [ 28*28, 800, 800, 10 ]
        dropout_rates = [ 0.0, 0.0, 0.0 ]
        activations = [ ReLU, ReLU ]
        mean=Geometric_mean
        results_file_name = "Original" + str(attempts) + "_800_800.txt"
        test_dark_knowledge(
            networks=[],
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=1,
            alpha=0.0,
            random_seed=random_seed)
        
        layer_sizes = [ 28*28, 800, 800, 10 ]
        dropout_rates = [ 0.0, 0.0, 0.0 ]
        activations = [ ReLU, ReLU ]
        mean=Geometric_mean
        results_file_name = "Dark_geometric" + str(attempts) + "_800_800.txt"    
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=temperature,
            alpha=alpha,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 800, 800, 10 ]
        dropout_rates = [ 0.0, 0.0, 0.0 ]
        activations = [ ReLU, ReLU ]
        mean=Average
        results_file_name = "Dark_average" + str(attempts) + "_800_800.txt"
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=temperature,
            alpha=alpha,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 1200, 10 ]
        dropout_rates = [ 0.0, 0.0]
        activations = [ ReLU ]
        results_file_name = "Original" + str(attempts) + "_1200.txt"
        test_dark_knowledge(
            networks=[],
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=1,
            alpha=0.0,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 1200, 10 ]
        dropout_rates = [ 0.0, 0.0 ]
        activations = [ ReLU ]
        mean=Geometric_mean
        results_file_name = "Dark_geometric" + str(attempts) + "_1200.txt"
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=temperature,
            alpha=alpha,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 1200, 10 ]
        dropout_rates = [ 0.0, 0.0 ]
        activations = [ ReLU ]
        mean=Average
        results_file_name = "Dark_average" + str(attempts) + "_1200.txt"
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=temperature,
            alpha=alpha,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 800, 10 ]
        dropout_rates = [ 0.0, 0.0]
        activations = [ ReLU ]
        results_file_name = "Original" + str(attempts) + "_800.txt"
        test_dark_knowledge(
            networks=[],
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=1,
            alpha=0.0,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 800, 10 ]
        dropout_rates = [ 0.0, 0.0 ]
        activations = [ ReLU ]
        mean=Geometric_mean
        results_file_name = "Dark_geometric" + str(attempts) + "_800.txt"
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            temperature=temperature,
            alpha=alpha,
            random_seed=random_seed)
            
        layer_sizes = [ 28*28, 800, 10 ]
        dropout_rates = [ 0.0, 0.0 ]
        activations = [ ReLU ]
        mean=Average
        results_file_name = "Dark_average" + str(attempts) + "_800.txt"
        test_dark_knowledge(
            networks=networks,
            mean=mean,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            squared_filter_length_limit=squared_filter_length_limit,
            n_epochs=n_epochs,
            n_matches=n_matches,
            patient=patient,
            batch_size=batch_size,
            mom_params=mom_params,
            activations=activations,
            dropout=dropout,
            dropout_rates=dropout_rates,
            results_file_name=results_file_name,
            layer_sizes=layer_sizes,
            datasets=datasets,
            use_bias=use_bias,
            random_seed=random_seed)