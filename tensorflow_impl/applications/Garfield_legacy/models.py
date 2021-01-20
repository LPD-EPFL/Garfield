# coding: utf-8
###
 # @file   models.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #         Arsany Guirguis <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
###

import math
import numpy
import os
import time

import tensorflow as tf
update_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), './update_model.so'))

# ---------------------------------------------------------------------------- #

# Feed-forward model class
class FeedForward:
    # Builder class
    class __Builder:
        # Dual name/variable scope context class
        class __DualScope:
            def __init__(self, name):
                """ Dual name/variable scope space constructor
                Args:
                    name Scope name
                """
                # Finalization
                self.__vars = tf.variable_scope(name)
                self.__name = tf.name_scope(name)

            def __enter__(self):
                """ Enter the dual scope.
                Returns:
                    Scope name
                """
                self.__vars.__enter__()
                return self.__name.__enter__()

            def __exit__(self, *args, **kwargs):
                """ Leave the dual scope.
                Args:
                    ... Forwarded arguments
                """
                try:
                    self.__name.__exit__(*args, **kwargs)
                finally:
                    self.__vars.__exit__(*args, **kwargs)

        # Shared name scope context class
        class __SharedScope:
            def __init__(self, scope):
                """ Shared name scope space constructor
                Args:
                    scope Scope of the builder, to use as root
                """
                cws = tf.get_default_graph().get_name_scope()
                nws = cws if cws[:len(scope)] != scope else cws[len(scope) + 1:]
                if len(nws) == 0:
                    nws = "shared"
                else:
                    nws = "shared/" + nws
                self.__root = tf.name_scope(None)
                self.__path = tf.name_scope(nws)

            def __enter__(self):
                """ Enter the shared scope.
                Returns:
                    Shared scope name
                """
                self.__root.__enter__()
                return self.__path.__enter__()

            def __exit__(self, *args, **kwargs):
                """ Leave the shared scope.
                Args:
                    ... Forwarded arguments
                """
                try:
                    self.__path.__exit__(*args, **kwargs)
                finally:
                    self.__root.__exit__(*args, **kwargs)

        def __init__(self, model):
            """ Bind constructor.
            Args:
                model Model to bind
            """
            self.__model = model
            self.__scope = tf.get_default_graph().get_name_scope()

        def scope(self, name):
            """ Return a simple dual name/variable scope context for tensors.
            Args:
                name Scope name
            Returns:
                Dual name/variable scope context for tensors
            """
            return type(self).__DualScope(name)

        def shared(self):
            """ Return a name scope context for shared tensors.
            Returns:
                Name scope context for shared tensors
            """
            return type(self).__SharedScope(self.__scope)

        def inputholder(self, dtype, inputs, *args, **kwargs):
            """ Instanciate the input placeholder.
            Args:
                dtype  Expected type
                inputs Default tensor (may be None)
                ...    Forwarded arguments
            Returns:
                Instanciated placeholder
            """
            if self.__model.tn_inputs is not None:
                raise RuntimeError("Model building failed: input placeholder exists")
            if inputs is None:
                tensor = tf.placeholder(dtype, *args, name="inputs", **kwargs)
            else:
                if inputs.dtype != dtype:
                    raise RuntimeError("Model building failed: input placeholder type mismatch")
                tensor = tf.placeholder_with_default(inputs, *args, name="inputs", **kwargs)
            self.__model.tn_inputs = tensor
            return tensor

        def labelholder(self, dtype, labels, *args, **kwargs):
            """ Instanciate the label placeholder.
            Args:
                dtype  Expected type
                labels Default tensor (may be None)
                ...    Forwarded arguments
            Returns:
                Instanciated placeholder
            """
            if self.__model.tn_labels is not None:
                raise RuntimeError("Model building failed: label placeholder exists")
            if labels is None:
                tensor = tf.placeholder(dtype, *args, name="labels", **kwargs)
            else:
                if labels.dtype != dtype:
                    raise RuntimeError("Model building failed: label placeholder type mismatch")
                tensor = tf.placeholder_with_default(labels, *args, name="labels", **kwargs)
            self.__model.tn_labels = tensor
            return tensor

        def paramfeeder(self, tensor):
            """ Use the given tensor as "placeholder" for the parameter assignment.
            Args:
                tensor Tensor to use as "placeholder", None to reset
            """
            self.__model.tn_param_in = tensor

        def gradfeeder(self, tensor):
            """ Use the given tensor as "placeholder" for 'update_gradient'.
            Args:
                tensor Tensor to use as "placeholder", None to reset
            """
            self.__model.tn_grad_in = tensor

        def epochfeeder(self, tensor):
            """ Use the given tensor as "placeholder" for the epoch counter assignment.
            Args:
                tensor Tensor to use as "placeholder", None to reset
            """
            self.__model.tn_epoch_in = tensor

        def variable(self, *args, **kwargs):
            """ Instanciate or get a variable.
            Args:
                ... Forwarded arguments
            Returns:
                Instanciated/gotten variable
            """
            tn_variable = tf.get_variable(*args, **kwargs)
            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options)
            sess = tf.Session(config=config)
            self.__model.tn_variables.append(tn_variable)
            with sess.as_default():
              self.__model.var_sizes.append(tf.size(tn_variable).eval())
              self.__model.var_shapes.append(tf.shape(tn_variable).eval())
            return tn_variable

        def logit(self, tn_logit):
            """ Register the optional logit tensor.
            Args:
                tn_logit Logit tensor to register
            Returns:
                Forward logit tensor
            """
            if self.__model.tn_logit is not None:
                raise RuntimeError("Model building failed: logit tensor already registered")
            self.__model.tn_logit = tn_logit
            return tn_logit

        def prob(self, tn_prob):
            """ Register the optional label probabilities tensor.
            Args:
                tn_prob Label probabilities tensor to register
            Returns:
                Forward label probabilities tensor
            """
            if self.__model.tn_prob is not None:
                raise RuntimeError("Model building failed: label probabilities tensor already registered")
            self.__model.tn_prob = tn_prob
            return tn_prob

        def loss(self, tn_loss):
            """ Register the loss tensor.
            Args:
                tn_loss Loss tensor to register
            Returns:
                Forward loss tensor
            """
            if self.__model.tn_loss is not None:
                raise RuntimeError("Model building failed: loss tensor already registered")
            self.__model.tn_loss = tn_loss
            return tn_loss

        def opt(self, optimizer):
            """ Register the optimizer.
            Args:
                optimizer Optimizer to register
            Returns:
                Forward optimizer
            """
            if self.__model.optimizer is not None:
                raise RuntimeError("Model building failed: optimizer already registered")
            self.__model.optimizer = optimizer
            return optimizer

        def eval(self, tn_eval):
            """ Register an evaluation tensor.
            Args:
                tn_eval Evaluation tensor to register
            Returns:
                Evaluation tensor
            """
            self.__model.tn_evals.append(tn_eval)
            return tn_eval

        def epoch(self):
            """ Instanciate/get a 64-bit epoch counter (incrementing on update).
            Returns:
                Instanciate/gotten epoch counter
            """
            if self.__model.tn_epoch_out is not None:
                return self.__model.tn_epoch_out
            tn_epoch_out = tf.Variable(0, dtype=tf.int64, trainable=False, expected_shape=())
            self.__model.tn_epoch_out = tn_epoch_out
            return tn_epoch_out

    @staticmethod
    def __get_sess(sess):
        """ Return the session to use for the computations.
        Args:
            sess Session to prefer (optional)
        Returns:
            Session to use
        """
        if sess is None:
            return tf.get_default_session()
        return sess

    @staticmethod
    def __flatten(variables, gradients=None):
        """ Build the flattened tensor of the given list of variables.
        Args:
            variables Variable node(s)
            gradients List (gradient, variable) to flatten (optional)
        Returns:
            Flattened parameter, or flattened gradient if 'gradients' is set
        """
        if gradients is None:
            return tf.concat([tf.reshape(variable, (-1,)) for variable in variables], 0)
        else:
            nodes = [None] * len(variables)
            for grad, var in gradients:
                nodes[variables.index(var)] = grad
            return tf.concat([tf.reshape(node, (-1,)) for node in nodes], 0)

    @staticmethod
    def __inflate(variables, flattened):
        """ Slice the flattened tensor according to the given list of variables.
        Args:
            variables Variable node(s)
            flattened Flattened tensor
        Returns:
            Map variable -> tensor segment
        """
        res = dict()
        pos = 0
        for var in variables:
            shape = var.shape
            size = shape.num_elements()
            res[var] = tf.reshape(tf.slice(flattened, [pos], [size]), shape)
            pos += size
        return res

    def __init__(self, *args, **kwargs):
        """ Feed-forward network constructor.
        Args:
            ... Arguments (see code)
        """
        self.__sealed = False
        if len(args) > 0:
            # Arguments "parsing"
            tn_inputs, tn_labels, tn_vars, tn_logit, tn_prob, tn_loss, optimizer = args
            tn_epoch_out = kwargs.get("tn_epoch_out", None)
            tn_epoch_in  = kwargs.get("tn_epoch_in", None)
            tn_param_in  = kwargs.get("tn_param_in", None)
            tn_grad_in   = kwargs.get("tn_grad_in", None)
            tn_evals = kwargs.get("tn_evals", list())
            # Application
            self.tn_variables = tn_vars
            self.var_sizes = [0 for i in range(len(tn_vars))]
            self.var_shapes = [[] for i in range(len(tn_vars))]
            self.tn_inputs    = tn_inputs
            self.tn_labels    = tn_labels
            self.tn_logit     = tn_logit
            self.tn_prob      = tn_prob
            self.tn_loss      = tn_loss
            self.tn_epoch_out = tn_epoch_out
            self.tn_epoch_in  = tn_epoch_in
            self.tn_param_in  = tn_param_in
            self.tn_grad_in   = tn_grad_in
            self.tn_evals     = tn_evals
            self.optimizer    = optimizer
            self.__exit__(None, None, None) # Validation and finalization
        else:
            self.tn_variables = list()
            self.var_sizes = list()
            self.var_shapes = list()
            self.tn_inputs    = None
            self.tn_labels    = None
            self.tn_logit     = None
            self.tn_prob      = None
            self.tn_loss      = None
            self.tn_epoch_out = None
            self.tn_epoch_in  = None
            self.tn_param_in  = None
            self.tn_grad_in   = None
            self.tn_evals     = list()
            self.optimizer    = None

    def __enter__(self):
        """ With statement enter, ensure the model is not sealed yet.
        Returns:
            self
        """
        if self.__sealed:
            raise RuntimeError("Model already sealed")
        return type(self).__Builder(self)

    def __exit__(self, err_type, err_value, tb):
        """ With statement exit, finalize model initialization and seal it.
        Args:
            err_type  Raised exception type (None for none)
            err_value Raised exception value (None for none)
            tb        Associated traceback (None for none)
        """
        if err_value is not None:
            return # Do nothing, and continue error propagation
        # Validation
        if len(self.tn_variables) == 0:
            raise RuntimeError("Model sealing failed: missing trainable variable(s)")
        if self.tn_inputs is None:
            raise RuntimeError("Model sealing failed: missing input placeholder")
        if self.tn_labels is None:
            raise RuntimeError("Model sealing failed: missing label placeholder")
        if self.tn_loss is None:
            raise RuntimeError("Model sealing failed: missing loss function")
        if self.optimizer is None:
            raise RuntimeError("Model sealing failed: missing optimizer")
        # Finalization: parameter (flat) vector
        self.tn_param_out = type(self).__flatten(self.tn_variables)
        if self.tn_param_in is None:
            self.tn_param_in = tf.placeholder(tf.float32, shape=self.tn_param_out.shape, name="param_in")
        else:
            self.tn_param_in = tf.placeholder_with_default(self.tn_param_in, self.tn_param_out.shape, name="param_in")
        params_in = type(self).__inflate(self.tn_variables, self.tn_param_in)
        # Finalization: gradient (flat) vector
        grads_out = self.optimizer.compute_gradients(self.tn_loss)
        self.tn_grad_out = type(self).__flatten(self.tn_variables, grads_out)
        if self.tn_grad_in is None:
            self.tn_grad_in = tf.placeholder(tf.float32, shape=self.tn_grad_out.shape, name="grad_in")
        else:
            self.tn_grad_in = tf.placeholder_with_default(self.tn_grad_in, self.tn_grad_out.shape, name="grad_in")
        grads_in = [(grad, var) for var, grad in type(self).__inflate(self.tn_variables, self.tn_grad_in).items()]
        # Finalization: operations
        if self.tn_epoch_out is None:
            self.op_epoch = None
        else:
            if self.tn_epoch_in is None:
                self.tn_epoch_in = tf.placeholder(tf.int64, shape=(), name="epoch_in")
            else:
                self.tn_epoch_in = tf.placeholder_with_default(self.tn_epoch_in, (), name="epoch_in")
            self.op_epoch = tf.assign(self.tn_epoch_out, self.tn_epoch_in)
        self.op_init   = tf.variables_initializer(self.tn_variables)
        self.op_write  = [tf.assign(var, params_in[var]) for var in self.tn_variables]
        self.op_update = self.optimizer.apply_gradients(grads_in, global_step=self.tn_epoch_out)
        self.op_fit    = self.optimizer.apply_gradients(grads_out, global_step=self.tn_epoch_out)
        self.op_saver  = tf.train.Saver(self.tn_variables, allow_empty=True)
        self.__savepth = "saved/"
        self.__sealed  = True

    def init(self, sess=None):
        """ (Re)initialize the variable(s) of the model and the epoch counter, if any.
        Args:
            sess Session to use (optional, use the default one)
        """
        sess = type(self).__get_sess(sess)
        if self.op_epoch is None:
            sess.run(self.op_init)
        else:
            sess.run((self.op_init, self.op_epoch), feed_dict={self.tn_epoch_in: 0})

    def read(self, sess=None):
        """ Read the (flat) parameter vector.
        Args:
            sess Session to use (optional, use the default one)
        Returns:
            Parameter vector as a numpy array
        """
        sess = type(self).__get_sess(sess)
        return sess.run(self.tn_param_out)

    def write(self, param, sess=None):
        """ Write/initialize the (flat) parameter vector.
        Args:
            param Parameter vector as a numpy array
            sess  Session to use (optional, use the default one)
        """
        sess = type(self).__get_sess(sess)
        sess.run(self.op_write, feed_dict={self.tn_param_in: param})

    def eval(self, x=None, y=None, sess=None):
        """ Compute the evaluation tensors of the model.
        Args:
            x    Input data to use (optional, default to None)
            y    Label data to use (optional, default to None)
            sess Session to use (optional, use the default one)
        Returns:
            Tuple with each evaluation node value
        """
        sess = type(self).__get_sess(sess)
        feed = dict()
        if x is not None:
            feed[self.tn_inputs] = x
        if y is not None:
            feed[self.tn_labels] = y
        return sess.run(self.tn_evals, feed_dict=feed)

    def backprop(self, x=None, y=None, sess=None):
        """ Compute the (flat) gradient of the parameter vector.
        Args:
            x    Input data to use (optional, default to None)
            y    Label data to use (optional, default to None)
            sess Session to use (optional, use the default one)
        Returns:
            Gradient as a numpy array
        """
        sess = type(self).__get_sess(sess)
        feed = dict()
        if x is not None:
            feed[self.tn_inputs] = x
        if y is not None:
            feed[self.tn_labels] = y
        return sess.run(self.tn_grad_out, feed_dict=feed)

    def update(self, grad, sess=None):
        """ Update the network with the given (flat) gradient.
        Args:
            grad Flat gradient
            sess Session to use (optional, use the default one)
        """
        sess = type(self).__get_sess(sess)
        return sess.run(self.op_update, feed_dict={self.tn_grad_in: grad})

    def update2(self, grad, sess=None):
        sess = type(self).__get_sess(sess)
        cur_ptr = 0
        lr = -0.001
        try:
          ind=0
          for var in self.tn_variables:
            size = self.var_sizes[ind]
            shape = self.var_shapes[ind]
            var.load(var.eval() - lr*(grad[cur_ptr:cur_ptr+size].reshape(shape)), sess)
#            sess.run(var.assign_add(lr*(grad[cur_ptr:cur_ptr+size].reshape(shape))))
            cur_ptr+=size
            ind+=1
        except Exception as e:
          print("Exception: ", e)
        addone = self.tn_epoch_out.assign_add(1)
        sess.run(addone)

    def update3(self, grad, sess=None):
        sess = type(self).__get_sess(sess)
        cur_ptr = 0
        try:
          ind=0
          for var in self.tn_variables:
            size = self.var_sizes[ind]
            shape = self.var_shapes[ind]
            ref = tf.convert_to_tensor(grad[cur_ptr:cur_ptr+size].reshape(shape))
            out = update_module.update_model(var, ref)
            sess.run(tf.assign(var, out))
            cur_ptr+=size
            ind+=1
        except Exception as e:
          print("Exception: ", e)
        addone = self.tn_epoch_out.assign_add(1)
        sess.run(addone)

    def fit(self, x=None, y=None, grad=False, sess=None):
        """ Compute the (flat) gradient and update the model accordingly in one run.
        Args:
            x    Input data to use (optional, default to None)
            y    Label data to use (optional, default to None)
            grad Whether to read the gradient (optional, default to False)
            sess Session to use (optional, use the default one)
        Returns:
            Gradient as a numpy array, if 'grad' is True
        """
        sess = type(self).__get_sess(sess)
        feed = dict()
        if x is not None:
            feed[self.tn_inputs] = x
        if y is not None:
            feed[self.tn_labels] = y
        if grad:
            return sess.run((self.op_fit, self.tn_grad_out), feed_dict=feed)[1]
        sess.run(self.op_fit, feed_dict=feed)

    def epoch(self, sess=None):
        """ Read the epoch counter.
        Args:
            sess Session to use (optional, use the default one)
        Returns:
            Counter value
        """
        if self.tn_epoch_out is None:
            raise RuntimeError("Epoch read unsupported by this model")
        sess = type(self).__get_sess(sess)
        return sess.run(self.tn_epoch_out)

    def reset(self, value=0, sess=None):
        """ Write the epoch counter.
        Args:
            value Value to set (optional, default to 0)
            sess  Session to use (optional, use the default one)
        """
        if self.op_epoch is None:
            raise RuntimeError("Epoch write unsupported by this model")
        sess = type(self).__get_sess(sess)
        sess.run(self.op_epoch, feed_dict={self.tn_epoch_in: value})

    def load(self, savepath=None, sess=None):
        """ Load the variables from a given checkpoint file.
        Args:
            savepath Path (prefix) of the checkpoint file (optional, latest checkpoint by default)
            sess     Session to load variables into (optional, use default session if missing)
        """
        if savepath is None:
            paths = self.op_saver.last_checkpoints
            if len(paths) == 0:
                raise RuntimeError("Model load: no last checkpoint to restore")
            savepath = paths[-1]
        if sess is None:
            sess = tf.get_default_session()
        self.op_saver.restore(sess, savepath)

    def save(self, basepath=None, sess=None, **kwargs):
        """ Save the variables to a given checkpoint file.
        Args:
            basepath Base path of the checkpoint file (optional)
            sess     Session to save variables from (optional, use default session if missing)
            ...      'Saver.save' forwarded keyword arguments (optional)
        Returns:
            Path prefix of the saved file to use with 'load'
        """
        if basepath is None:
            basepath = self.__savepth
        else:
            self.__savepth = basepath
        if sess is None:
            sess = tf.get_default_session()
        return self.op_saver.save(sess, basepath, **kwargs)

# ---------------------------------------------------------------------------- #

class NoOpCtx:
    @staticmethod
    def noopctx():
        """ Return a new NoOpCtx.
        Returns:
            New NoOpCtx
        """
        return NoOpCtx()
    def __enter__(self):
        """ Do nothing.
        """
        pass
    def __exit__(self, err_type, err_value, tb):
        """ Do nothing.
        """
        pass

def dense_classifier(dims, inputs=None, labels=None, act_fn=tf.nn.relu, optimizer=None, eta=0.02, epoch=False, shared=False):
    """ Build a new dense, feed-forward classifier neural network.
    Args:
        dims     Dimension array: [input, hiddens..., output]
        inputs   Input tensor (optional, default to None)
        labels   Label tensor (optional, default to None)
        act_fn   Activation function (optional, default to 'tf.nn.relu')
        node_opt Optimizer node to use (optional, default to SGD with 'eta')
        eta      Learning rate (optional, ignored if 'node_opt' is set)
        epoch    Whether an epoch counter must be created (optional, default to False)
        shared   Whether the trainable parameters (i.e. variables) are shared (optional, default to False)
    Returns:
        New FeedFoward instance
    """
    model = FeedForward()
    with model as builder:
        # Context creator to use
        shared_ctx = builder.shared if shared else NoOpCtx.noopctx
        # Input/label placeholders
        tn_input = builder.inputholder(tf.float32, inputs, shape=(None, dims[0]))
        tn_label = builder.labelholder(tf.int32, labels, shape=(None,))
        # Inference layers
        tn_hidden = tn_input
        length = len(dims)
        for i in range(length - 1):
            with builder.scope("dense_" + str(i + 1)):
                dim_in  = dims[i]
                dim_out = dims[i + 1]
                with shared_ctx():
                    weights = builder.variable("weights", shape=(dim_in, dim_out), dtype=tf.float32)
                    biases  = builder.variable("biases", shape=(dim_out,), dtype=tf.float32)
                if i == length - 2: # Output linear layer
                    tn_linear = tf.matmul(tn_hidden, weights) + biases
                else: # Hidden, dense layer
                    tn_hidden = act_fn(tf.matmul(tn_hidden, weights) + biases)
        # Classification probability (only for classifier), loss (cross-entropy on softmax), evaluation (proportion of well-classed inputs) and update counter
        builder.logit(tn_linear)
        builder.prob(tf.nn.softmax(tn_linear))
        builder.loss(tf.losses.sparse_softmax_cross_entropy(labels=tn_label, logits=tn_linear))
        builder.eval(tf.reduce_mean(tf.cast(tf.nn.in_top_k(tn_linear, tn_label, 1), tf.float32)))
        if epoch:
            builder.epoch()
        # Optimizer (basic SGD as default)
        builder.opt(tf.train.GradientDescentOptimizer(eta) if optimizer is None else optimizer)
    return model

def conv_cifar10(inputs=None, labels=None, act_fn=tf.nn.relu, optimizer=None, eta=0.02, epoch=True, shared=False):
    """ Build a new convolutional, feed-forward classifier neural network for CIFAR-10.
    Args:
        inputs   Input tensor (optional, default to None)
        labels   Label tensor (optional, default to None)
        act_fn   Activation function (optional, default to 'tf.nn.relu')
        node_opt Optimizer node to use (optional, default to SGD with 'eta')
        eta      Learning rate (optional, ignored if 'node_opt' is set)
        epoch    Whether an epoch counter must be created (optional, default to True)
    Returns:
        New FeedFoward instance
    """
    model = FeedForward()
    with model as builder:
        # Context creator to use
        shared_ctx = builder.shared if shared else NoOpCtx.noopctx
        # Input/label placeholders
        tn_input = builder.inputholder(tf.float32, inputs, shape=(None, 32, 32, 3))
        tn_label = builder.labelholder(tf.int32, labels, shape=(None,))
        # Convolutional layer 1
        with builder.scope("conv1") as scope:
            with shared_ctx():
                kernel = builder.variable("weights", shape=(5, 5, 3, 64), dtype=tf.float32)
                biases = builder.variable("biases", shape=(64,), dtype=tf.float32)
            conv   = tf.nn.conv2d(tn_input, kernel, [1, 1, 1, 1], padding="SAME")
            preact = tf.nn.bias_add(conv, biases)
            conv1  = act_fn(preact, name=scope)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        # Convolutional layer 2
        with builder.scope("conv2") as scope:
            with shared_ctx():
                kernel = builder.variable("weights", shape=(5, 5, 64, 64), dtype=tf.float32)
                biases = builder.variable("biases", shape=(64), dtype=tf.float32)
            conv   = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
            preact = tf.nn.bias_add(conv, biases)
            conv2  = act_fn(preact, name=scope)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        end_dim = numpy.prod(pool2.get_shape().as_list()[1:])
        reshape = tf.reshape(pool2, [-1, end_dim])
        # Dense layer 3
        with builder.scope("dense3") as scope:
            with shared_ctx():
                weights = builder.variable("weights", shape=(end_dim, 384), dtype=tf.float32)
                biases  = builder.variable("biases", shape=(384,), dtype=tf.float32)
            local3  = act_fn(tf.matmul(reshape, weights) + biases, name=scope)
        # Dense layer 4
        with builder.scope("dense4") as scope:
            with shared_ctx():
                weights = builder.variable("weights", shape=(384, 192), dtype=tf.float32)
                biases  = builder.variable("biases", shape=(192,), dtype=tf.float32)
            local4  = act_fn(tf.matmul(local3, weights) + biases, name=scope)
        # Output layer
        with builder.scope("linear") as scope:
            with shared_ctx():
                weights = builder.variable("weights", shape=(192, 10), dtype=tf.float32)
                biases  = builder.variable("biases", shape=(10,), dtype=tf.float32)
            tn_linear = tf.add(tf.matmul(local4, weights), biases, name=scope)
        # Classification probability (only for classifier), loss (cross-entropy on softmax), evaluation (proportion of well-classed inputs) and update counter
        builder.logit(tn_linear)
        builder.prob(tf.nn.softmax(tn_linear))
        builder.loss(tf.losses.sparse_softmax_cross_entropy(labels=tn_label, logits=tn_linear))
        builder.eval(tf.reduce_mean(tf.cast(tf.nn.in_top_k(tn_linear, tn_label, 1), tf.float32)))
        if epoch:
            builder.epoch()
        # Optimizer (basic SGD as default)
        builder.opt(tf.train.GradientDescentOptimizer(eta) if optimizer is None else optimizer)
    return model
