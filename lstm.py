import theano
import theano.tensor as T
import numpy as np
import theano.tensor.nnet as nnet
from keras import backend as K
from keras import initializations
import sys

# SETUP
floatX = theano.config.floatX
np.random.seed(1)
init = initializations.get('uniform')

# SPECIFY THE PARAMETER TO TAKE GRADIENT OVER
# i: 0, 1, 2, 3 corresponding to input, forget, output, and cell weighted inputs
# j: 0, 1, 2 corresponding to weights of x, h_tm1, and b
d_gate = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d_component = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def sigmoid(a):
	return 1 / (1 + np.exp(-a))

# DIMS
x_size = 2
h_size = 3

w_shape = (h_size, x_size)
v_shape = (h_size, h_size)
b_shape = h_size

weights = []

for i in xrange(4):
	weights.append([])
	weights[i].append(init(w_shape))
	weights[i].append(init(v_shape))
	weights[i].append(T.zeros(b_shape))

# INPUT PERTURBATION FOR NUMERICAL GRADIENTS
delta = T.matrix('delta', dtype=floatX)

# MODEL I/O
data_in = T.matrix('data_in', dtype=floatX)
data_out = T.matrix('data_out', dtype=floatX)
# INITIAL STATES FOR CELL AND HIDDEN
c0 = T.vector('c0', dtype=floatX)
h0 = T.vector('h0', dtype=floatX)

# ACTIVATION FUNCTIONS
inner_activation = nnet.sigmoid
outer_activation = T.tanh


# LOOPING THROUGH DATA POINTS
def _step(x, y, c_tm1, h_tm1):
	'''
	return: cell, hidden, error, de/dcell_in (for graves gradients), four gates (for wang gradients)
	'''
	# CONSTRUCT WEIGHTED INPUTS FOR GATES
	gates_in = []
	for i in xrange(4):
		w = weights[i][0]
		v = weights[i][1]
		b = weights[i][2]

		# ADD PERTURBATION TO TARGET GATE FOR GRADIENT CALCULATION
		if d_gate == i:
			if 0 == d_component:
				w += delta
			elif 1 == d_component:
				v += delta

		gates_in.append(T.dot(w, x) + T.dot(v, h_tm1) + b)

	# CALCULATE GATES
	gates = []
	for i in xrange(3):
		gates.append(inner_activation(gates_in[i]))
	gates.append(outer_activation(gates_in[3]))  # [3] is cell, using tanh

	# CALCULATE CELL
	cell = gates[0] * gates[3] + gates[1] * c_tm1

	# CALCULATE OUTPUT
	hidden = gates[2] * cell

	# CALCULATE ERROR
	err = T.dot(hidden, y)  # INNER PRODUCT as LOSS

	return cell, hidden, err, theano.gradient.grad(err, gates_in[d_gate]), gates[0], gates[1], gates[2], gates[3]

[c, h, e, graves_delta, gi, gf, go, gc], _ = theano.scan(_step,
								sequences=[data_in, data_out],
								outputs_info=[c0, h0, None, None, None, None, None, None])  # RECURRENCE ON c AND h ONLY

# CALCULATING THEANO GRADIENT delta(t) = dE(t) / dw (last time step)
err = e[-1]
theano_d = theano.gradient.grad(err, weights[d_gate][d_component])

# COMPILING THEANO FUNCTION
# returning weights[:][1]: the weight matrix for the recurrent term h_tm1, which is often required by wang gradients
lstm = theano.function(inputs=[data_in, data_out, c0, h0, delta],
						outputs=[c, h, e, theano_d, graves_delta, gi, gf, go, gc, weights[0][1], weights[1][1], weights[2][1], weights[3][1]], on_unused_input='ignore')

# MODEL I/O
N = 5
data_in = np.random.random((N, x_size)).astype(floatX)
data_out = np.random.random((N, h_size)).astype(floatX)

graves_x = np.array(data_in, copy=True)
# SHIFTING X FOR CALCULATING GRAVES GRADIENT delta(t) * x(t - 1)
for i in range(N - 1, 0, -1):  # [0] and [1] will be the same but [0] will not be used
	graves_x[i] = graves_x[i - 1]
graves_x = graves_x.reshape(N, 1, x_size)

# INITIAL STATES
c0 = np.zeros(h_size).astype(floatX)
h0 = np.zeros(h_size).astype(floatX)

eps = 1e-4

params_shape = w_shape if 0 == d_component else v_shape


def dg(gates, gid):
	gate = gates[gid]
	if 3 == gid:  # cell, tanh
		return 1 - gate * gate
	else:
		return gate * (1 - gate)  # gates, sigmoid

# for pos in xrange(w.size):
for ir in xrange(params_shape[0]):
	for ic in xrange(params_shape[1]):
		# CONSTRUCTING DELTA AT POSITION pos
		delta = np.zeros(params_shape).astype(floatX)
		delta[ir][ic] = eps
		# delta[ir][ic] = w[ir][ic] * eps

		# y- AND y+
		[c1, h1, e1, theano_d1, graves_delta1, gi_1, gf_1, go_1, gc_1, wy_i_1, wy_f_1, wy_o_1, wy_c_1] = lstm(data_in, data_out, c0, h0, -delta)
		[c2, h2, e2, theano_d2, graves_delta2, gi_2, gf_2, go_2, gc_2, wy_i_2, wy_f_2, wy_o_2, wy_c_2] = lstm(data_in, data_out, c0, h0, delta)

		# print('err1', e1)
		# print('err2', e2)
		print("R%dC%d" % (ir, ic))
		'''
		######### THEANO GRADIENT #########
		'''
		print('theano:\t%.6e' % (theano_d1[ir][ic]))

		'''
		######### NUMERICAL GRADIENT #########
		'''
		de = e2[-1] - e1[-1]
		print("num.:\t%.6e" % (de / eps / 2))

		'''
		######### GRAVES GRADIENT #########
		'''
		# print("delta:", outputs1[3])
		# print("x:", data_in)
		graves_gradient = graves_delta1.reshape(N, h_size, 1) * graves_x
		graves_gradient = graves_gradient[1:].sum(axis=0)
		print("graves:\t%.6e" % (graves_gradient[ir][ic]))

		'''
		######### WANG GRADIENT #########
		'''
		dEdy_t = data_out[-1]  # INNER PRODUCT LOSS:/dy = y_hat (h_size)
		# print("dEdy_t.shape", dEdy_t.shape)

		def dg_alpha_dw_beta_x(gates, alpha, beta, x_t, wys, dy_tm1):
			dg_alpha = dg(gates, alpha)
			if alpha != beta:
				x_t = np.zeros_like(x_t)
			return dg_alpha.reshape(h_size, 1) * (x_t + np.array([np.dot(wys[alpha], tmp) for tmp in dy_tm1]))

		def dc_dw_beta_x(gates, c_tm1, dc_tm1, beta, x_t, wys, dy_tm1):
			retval = gates[0].reshape(h_size, 1) * dg_alpha_dw_beta_x(gates, 3, beta, x_t, wys, dy_tm1)
			retval += gates[3].reshape(h_size, 1) * dg_alpha_dw_beta_x(gates, 0, beta, x_t, wys, dy_tm1)
			retval += gates[1].reshape(h_size, 1) * dc_tm1
			retval += c_tm1.reshape(h_size, 1) * dg_alpha_dw_beta_x(gates, 1, beta, x_t, wys, dy_tm1)
			return retval

		def dy_dw_beta_x(gates, beta, cell, x_t, wys, dy_tm1, dc):
			retval = cell.reshape(h_size, 1) * dg_alpha_dw_beta_x(gates, 2, beta, x_t, wys, dy_tm1)
			retval += gates[2].reshape(h_size, 1) * dc
			return retval

		dy = np.zeros((h_size, params_shape[0], params_shape[1])).astype(floatX)
		dc = np.zeros((h_size, params_shape[0], params_shape[1])).astype(floatX)
		c_tm1 = c0
		wys = [wy_i_1, wy_f_1, wy_o_1, wy_c_1]
		for t in xrange(0, N):
			c_tm1 = c1[t - 1] if t > 0 else c0
			gates = [gi_1[t], gf_1[t], go_1[t], gc_1[t]]
			dc = dc_dw_beta_x(gates, c_tm1, dc, d_gate, data_in[t], wys, dy)
			dy = dy_dw_beta_x(gates, d_gate, c1[t], data_in[t], wys, dy, dc)

		dy = dEdy_t.reshape(h_size, 1) * dy
		print("wang:\t%.6e" % (dy[0][ir][ic]))
