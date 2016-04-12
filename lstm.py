import theano
import theano.tensor as T
import numpy as np
import theano.tensor.nnet as nnet
from keras import backend as K
from keras import initializations

# SETUP
floatX = theano.config.floatX
np.random.seed(1)
init = initializations.get('uniform')


def sigmoid(a):
	return 1 / (1 + np.exp(-a))

# DIMS
x_size = 2
h_size = 3

# WEIGHT MATRICES;
W_ix = init((h_size, x_size))
W_ih = init((h_size, h_size))
# W_ic = init((h_size, x_size))
b_i = K.zeros(h_size)

W_fx = init((h_size, x_size))
W_fh = init((h_size, h_size))
# W_fc = init((h_size, x_size))
b_f = K.zeros(h_size)

W_cx = init((h_size, x_size))
W_ch = init((h_size, h_size))
# W_cc = init((h_size, x_size))
b_c = K.zeros(h_size)

# PARAMETERIZING W_ox TO EXAMINE GRADIENTS
# W_ox = init((h_size, x_size))
W_ox = T.matrix('W_ox', dtype=floatX)
W_oh = init((h_size, h_size))
# W_oc = init((h_size, x_size))
b_o = K.zeros(h_size)

# MODEL I/O
data_in = T.matrix('data_in', dtype=floatX)
data_out = T.matrix('data_out', dtype=floatX)
c0 = T.vector('c0', dtype=floatX)
h0 = T.vector('h0', dtype=floatX)

# ACTIVATION FUNCTIONS
inner_activation = nnet.sigmoid
outer_activation = T.tanh


# LOOPING THROUGH DATA POINTS
def _step(x, y, c_tm1, h_tm1):
	G_i = inner_activation(K.dot(W_ix, x) + K.dot(W_ih, h_tm1) + b_i)
	G_f = inner_activation(K.dot(W_fx, x) + K.dot(W_fh, h_tm1) + b_f)
	c_tilde = outer_activation(K.dot(W_cx, x) + K.dot(W_ch, h_tm1) + b_c)
	cell = G_f * c_tm1 + G_i * c_tilde
	in_o = K.dot(W_ox, x) + K.dot(W_oh, h_tm1) + b_o
	G_o = inner_activation(in_o)
	hidden = G_o * cell
	err = K.dot(hidden, y)  # INNER PRODUCT as LOSS
	# err = K.dot(hidden, y) ** 2 / 2  # MSE of INNER PRODUCT as LOSS
	return c_tilde, cell, hidden, err, theano.gradient.grad(err, in_o), G_i, G_o

[c_tilde, c, h, e, d_ox, gi, go], _ = theano.scan(_step,
								sequences=[data_in, data_out],
								outputs_info=[None, c0, h0, None, None, None, None])  # RECURRENCE ON c AND h ONLY

# CALCULATING THEANO GRADIENT delta(t) = dE(t) / dw
err = e[-1]
theano_d = theano.gradient.grad(err, W_ox)

# COMPILING THEANO FUNCTION
lstm = theano.function(inputs=[data_in, data_out, c0, h0, W_ox],
						outputs=[c_tilde, c, h, e, d_ox, theano_d, gi, go, W_ch, W_oh], on_unused_input='ignore')

# MODEL I/O
N = 5
data_in = np.random.random((N, x_size)).astype(floatX)
data_out = np.random.random((N, h_size)).astype(floatX)

graves_gradient_x = np.array(data_in, copy=True)
# SHIFTING X FOR CALCULATING GRAVES GRADIENT delta(t) * x(t - 1)
for i in range(N - 1, 0, -1):  # [0] and [1] will be the same but [0] will not be used
	graves_gradient_x[i] = graves_gradient_x[i - 1]
graves_gradient_x = graves_gradient_x.reshape(N, 1, x_size)

# INIT STATES
c0 = np.zeros(h_size).astype(floatX)
h0 = np.zeros(h_size).astype(floatX)
# WEIGHT MATRIX
w = np.random.random((h_size, x_size)).astype(floatX)

eps = 1e-4

# for pos in xrange(w.size):
xshape = w.shape
# xshape = (1, 1)
for ir in xrange(xshape[0]):
	for ic in xrange(xshape[1]):
		# CONSTRUCTING DELTA AT POSITION pos
		delta = np.zeros(w.shape).astype(floatX)
		delta[ir][ic] = w[ir][ic] * eps

		# y- AND y+
		[c_tilde, c1, h1, e1, d_ox1, theano_d1, gi_1, go_1, W_ch_1, W_oh_1] = lstm(data_in, data_out, c0, h0, w - delta)
		[c_tilde, c2, h2, e2, d_ox2, theano_d2, gi_2, go_2, W_ch_2, W_oh_2] = lstm(data_in, data_out, c0, h0, w + delta)

		print('err1', e1)
		print('err2', e2)

		'''
		######### THEANO GRADIENT #########
		'''
		print('theano:\t%.6e' % (theano_d1[ir][ic]))

		'''
		######### NUMERICAL GRADIENT #########
		'''
		de = e2[-1] - e1[-1]
		print("numerical:\t%.6e" % (de / delta[ir][ic] / 2))

		'''
		######### GRAVES GRADIENT #########
		'''
		# print("delta:", outputs1[3])
		# print("x:", data_in)
		graves_gradient = d_ox1.reshape(N, h_size, 1) * graves_gradient_x
		graves_gradient = graves_gradient[1:].sum(axis=0)
		print("graves:\t%.6e" % (graves_gradient[ir][ic]))

		'''
		######### WANG GRADIENT #########
		'''
		dEdy_t = data_out[-1]  # INNER PRODUCT LOSS:/dy = y_hat (h_size)

		dy_dw = (c1[0] * go_1[0] * (1 - go_1[0])).reshape(h_size, 1) * data_in[0].reshape(1, x_size)  # h_size X x_size
		for t in xrange(1, N):
			templ = (c1[t] * go_1[t] * (1 - go_1[t])).reshape(h_size, 1)
			tempr = data_in[t].reshape(1, x_size) + np.dot(W_oh_1, dy_dw)
			dy_dw = templ * tempr

		dy_dw *= dEdy_t.reshape(h_size, 1)
		print("wang:\t%.6e" % (dy_dw[ir][ic]))

		# print(mult.shape, dy_0dw.shape, wang_gradient.shape)

		# print(mult.shape)
		# print(gi_1.shape, go_1.shape, in_c1.shape, W_ch_1.shape)
		# print(gi_1.shape, go_1.shape)
		# print(W_ch_1, W_ch_2)

		# print(graves_gradient)
		# print(graves_gradient.shape)
