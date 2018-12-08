import numpy as np
import scipy as sp
from scipy.linalg import expm
from numpy.random import randn
import matplotlib.pyplot as plt
from numpy import shape
import time
from numpy.linalg import matrix_rank
from numpy.random import randn
######################
## program controls (only do one thing at a time!)
linear = 0 # if running linear sim, do KF or UKF (no EKF!)
nonlinear = 1 # if running nonlinear sim, run EKF or UKF (no KF!)
KF = 0
EKF = 1
UKF = 0

#TODO implement bias tracking if we only have velocity or acceleration measurement
#TODO find a non-linear example for space applications
######################
## define the transition and observation functions
def f(x, dt):
    x_next = np.zeros([3,1])
    x_next[0] = np.cos(tOUT[i] + x[1]*x[0])
    x_next[1] = -np.sin(tOUT[i] + x[0]*x[2])
    x_next[2] = -np.cos(tOUT[i] + x[1])

    return x_next

def jacobian(x, t):
    # evaluates the jacobian at a point x
    J = np.zeros([num_states, num_states])
    J[0, 0] = -1.*np.sin(t + x[1]) 
    J[0, 1] = -1.*np.sin(t + x[0])
    J[0, 2] = 0.
    J[1, 0] = -1.*np.cos(t + x[2]) 
    J[1, 1] = 0.
    J[1, 2] = -1.*np.cos(t + x[0]) 
    J[2, 0] = 0.
    J[2, 1] = np.sin(t + x[1])
    J[2, 2] = 0.

    return J

def f_linear(x,dt):
    A = np.array([
        [0, 1, 0],
        [0 ,0, 1],
        [0, 0, 0]
    ])

    A_discrete = expm(A*dt)
    return  A_discrete @ x

def phi_discrete(dt):
    A = np.array([
        [0, 1, 0],
        [0 ,0, 1],
        [0, 0, 0]
    ])

    A_discrete = expm(A*dt)
    return A_discrete

def h(x):
    #TODO make a nonlinear observation?
    C = np.array([1, 0, 0])
    return C @ x

def h_linear():
    return np.array([1, 0, 0]).reshape(1, num_states)
######################
## define UKF parameters
# set ICs
x = np.expand_dims(np.array([0, -20, 1.5]), 1)

num_states = len(x)
num_obs = 1

x_hat = 5*np.ones([num_states,1])

# confidence in process model (deals with unmodeled dynamics)
var_process = 0.1
Q = np.eye(num_states)*var_process

# confidence in sensor model (deals with sensor noise, bias)
var_sensor = 1.
R = var_sensor

P = np.eye(num_states) * 0.1

# sim time
dt = 0.01
tf = 25
num_steps = int(tf / dt)

# UKF parameters
L = num_states
kappa = 3-L
alpha = 10**-2
beta = 2
lambda_ = alpha**2 * (L+kappa)

def sigma_points(x, P):
    x = x.squeeze(1)
    sigmas = np.zeros([2*L+1, L])
    sigmas[0] = x
    
    # U = sp.linalg.cholesky((L+lambda_)*P)
    U = sp.linalg.sqrtm((L+lambda_)*P)
    for k in range(L):
        sigmas[k+1] = x+U[k]
        sigmas[L+k+1] = x-U[k]
    return sigmas

def weights():
    Wc = np.full(2*L+1, 1./(2*(L+lambda_)))
    Wm = np.full(2*L+1, 1./(2*(L+lambda_)))
    
    Wc[0] = lambda_ / (L+lambda_) + (1. - alpha**2 + beta)
    Wm[0] = lambda_ / (L+lambda_)

    return Wc, np.expand_dims(Wm, 1)

def plot_sigmas(x, P):
    sigmas = sigma_points(x, P)
    num_points = len(sigmas)

    for i in range(num_points):
        if i == 0:
            plt.scatter(sigmas[i, 0], sigmas[i, 1], c = 'b', label = 'previous sigmas')
        else:
            plt.scatter(sigmas[i, 0], sigmas[i, 1], c = 'b')

######################
## simulate the real system
# storage variables
xOUT = np.zeros([num_states, num_steps])
x_hatOUT = np.zeros([num_states, num_steps])
pOUT = np.zeros([num_states, num_steps])
yOUT = np.zeros([num_obs, num_steps])
innovationOUT = np.zeros([num_obs, num_steps])
tOUT = np.arange(0, tf, dt)

# run sim
if linear:
    for i in range(num_steps):
        x = f_linear(x, dt)
        xOUT[:, i] = x.squeeze(1)

if nonlinear:
    for i in range(num_steps):
        x = f(x, dt)
        xOUT[:, i] = x.squeeze(1)

######################
## run UKF
if linear and UKF:
    ## run UKF
    Wc, Wm = weights()
    Wc_diag = np.diag(Wc)

    for i in range(num_steps):
        #################
        ## get sigma points
        sigmas_old = sigma_points(x_hat, P)
        # plot_sigmas(x_hat, P)

        num_sig = len(sigmas_old)

        sigmas = np.zeros([2*L+1, L])
        obs_sigmas = np.zeros(num_sig)
        #################
        ## propagate through system model
        # propagate sigma points through nonlinaer system 
        for j in range(num_sig):
            sig = np.reshape(sigmas_old[j, :], [num_states,1])
            sigmas[j, :] = f_linear(sig, dt).squeeze(1)
            
            obs_sig = h(sigmas[j, :])
            obs_sigmas[j] = obs_sig
            
        #     if j == 0:
        #         plt.scatter(sigmas[j, 0], sigmas[j, 1], c = 'r', label = 'current sigmas')
        #     else:
        #         plt.scatter(sigmas[j, 0], sigmas[j, 1], c = 'r')
        # plt.legend()
        # plt.show()

        # mean of propogated sigma points is the state estimate
        x_hat_minus = (Wm.T @ sigmas).reshape(num_states, 1)
        y_hat_minus = (Wm.T @ obs_sigmas)
        
        a = (sigmas - x_hat_minus.T)
        P_minus = a.T @ Wc_diag @ a + Q
        
        #################
        ## update estimates
        b = (obs_sigmas - y_hat_minus)
        P_yy = b.T @ Wc_diag @ b + R
        P_xy = a.T @ Wc_diag @ b
        P_yy = np.expand_dims(np.expand_dims(P_yy, 0), 0)
        
        # Kalman Gain
        K = (P_xy * np.linalg.inv(P_yy)).T #TODO replace with matrix multiplication for different cases
        
        # sensor reading + noise
        y = h(xOUT[:, i]) + randn()*var_sensor

        e = np.expand_dims((y - y_hat_minus), 1)
        x_hat_plus = x_hat_minus + K @ e
        P_plus = P_minus - K @ P_yy @ K.T
        P_diag_elements = np.diag(P_plus).reshape([num_states, 1])
        # P_plus_PSD = np.diag(P_diag_elements)

        #################
        ## update for next iteration
        x_hat = x_hat_plus
        P = P_plus

        x_hatOUT[:, i] = x_hat.squeeze(1)
        innovationOUT[:, i] = e
        pOUT[:, i] = P_diag_elements.squeeze(1)
        yOUT[0, i] = y

######################
## run KF
if linear and KF:
    phi = phi_discrete(dt)
    C = h_linear()
    for i in range(num_steps):
        ###############
        ## predict
        x_hat_minus = phi @ x_hat
        P_minus = phi @ P @ phi.T

        ###############
        ## correct
        y = h(xOUT[:, i]) + randn()*var_sensor
        # residual / innovation / error 
        e = y - h(x_hat_minus)
        # system uncertainty
        S = C @ P_minus @ C.T + R
        # K = P_minus @ C.T @ np.linalg.inv(S)
        K = (P_minus @ C.T * S**-1).reshape(num_states, 1)

        ## update
        # x_hat_plus = x_hat_minus + K @ e
        x_hat_plus = x_hat_minus + K * e
        P_plus = (np.eye(num_states) - K @ C) @ P_minus

        ###############
        ## update for next iteration, save current step
        x_hat = x_hat_plus
        P = P_plus

        P_diag_elements = np.diag(P_plus).reshape([num_states, 1])
        x_hatOUT[:, i] = x_hat.squeeze(1)
        innovationOUT[:, i] = e
        pOUT[:, i] = P_diag_elements.squeeze(1)
        yOUT[0, i] = y

######################
## run UKF
if nonlinear and UKF:
    ## run UKF
    Wc, Wm = weights()
    Wc_diag = np.diag(Wc)

    for i in range(num_steps):
        #################
        ## get sigma points
        sigmas_old = sigma_points(x_hat, P)
        # plot_sigmas(x_hat, P)

        num_sig = len(sigmas_old)

        sigmas = np.zeros([2*L+1, L])
        obs_sigmas = np.zeros(num_sig)
        #################
        ## propagate through system model
        # propagate sigma points through nonlinaer system 
        for j in range(num_sig):
            sig = np.reshape(sigmas_old[j, :], [num_states,1])
            sigmas[j, :] = f(sig, dt).squeeze(1)
            
            obs_sig = h(sigmas[j, :])
            obs_sigmas[j] = obs_sig
            
        #     if j == 0:
        #         plt.scatter(sigmas[j, 0], sigmas[j, 1], c = 'r', label = 'current sigmas')
        #     else:
        #         plt.scatter(sigmas[j, 0], sigmas[j, 1], c = 'r')
        # plt.legend()
        # plt.show()

        # mean of propogated sigma points is the state estimate
        x_hat_minus = (Wm.T @ sigmas).reshape(num_states, 1)
        y_hat_minus = (Wm.T @ obs_sigmas)
        
        a = (sigmas - x_hat_minus.T)
        P_minus = a.T @ Wc_diag @ a + Q
        
        #################
        ## update estimates
        b = (obs_sigmas - y_hat_minus)
        P_yy = b.T @ Wc_diag @ b + R
        P_xy = a.T @ Wc_diag @ b
        P_yy = np.expand_dims(np.expand_dims(P_yy, 0), 0)
        
        # Kalman Gain
        K = (P_xy * np.linalg.inv(P_yy)).T #TODO replace with matrix multiplication for different cases
        
        # sensor reading + noise
        y = h(xOUT[:, i]) + randn()*var_sensor

        e = np.expand_dims((y - y_hat_minus), 1)
        x_hat_plus = x_hat_minus + K @ e
        P_plus = P_minus - K @ P_yy @ K.T
        P_diag_elements = np.diag(P_plus).reshape([num_states, 1])
        # P_plus_PSD = np.diag(P_diag_elements)

        #################
        ## update for next iteration
        x_hat = x_hat_plus
        P = P_plus

        x_hatOUT[:, i] = x_hat.squeeze(1)
        innovationOUT[:, i] = e
        pOUT[:, i] = P_diag_elements.squeeze(1)
        yOUT[0, i] = y

######################
## run EKF
if nonlinear and EKF:
    phi = phi_discrete(dt)
    C = h_linear()
    for i in range(num_steps):
        t = tOUT[i]
        ###############
        ## predict
        x_hat_minus = f(x_hat, dt)
        J = jacobian(x_hat, t)
        P_minus = J @ P @ J.T + Q

        
        ###############
        ## correct
        y = h(xOUT[:, i]) + randn()*var_sensor
        # residual / innovation / error 
        e = y - h(x_hat_minus)
        # system uncertainty
        S = C @ P_minus @ C.T + R
        # K = P_minus @ C.T @ np.linalg.inv(S)
        K = (P_minus @ C.T * S**-1).reshape(num_states, 1)

        ## update
        # x_hat_plus = x_hat_minus + K @ e
        x_hat_plus = x_hat_minus + K * e
        P_plus = (np.eye(num_states) - K @ C) @ P_minus

        ###############
        ## update for next iteration, save current step
        x_hat = x_hat_plus
        P = P_plus

        P_diag_elements = np.diag(P_plus).reshape([num_states, 1])
        x_hatOUT[:, i] = x_hat.squeeze(1)
        innovationOUT[:, i] = e
        pOUT[:, i] = P_diag_elements.squeeze(1)
        yOUT[0, i] = y

######################
eOUT = np.abs(xOUT - x_hatOUT)

plt.figure(figsize = [10,10])
plt.subplot(311)
plt.plot(tOUT, xOUT[0, :], label = 'Real State', c = 'blue')
plt.plot(tOUT, x_hatOUT[0, :], label = 'Noisy Measurement', c = 'orange', alpha = .5)
# plt.scatter(tOUT, yOUT[0 ,:], label = 'Noisy Measurement', c = 'green', marker = 'X',alpha = .35)
plt.ylabel('x1')
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(tOUT, xOUT[1, :], label = 'Real State')
plt.plot(tOUT, x_hatOUT[1, :], label = 'Estimated State', alpha = .5)
plt.ylabel('x2')
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(tOUT, xOUT[2, :], label = 'Real State')
plt.plot(tOUT, x_hatOUT[2, :], label = 'Estimated State', alpha = .5)
plt.xlabel('Time')
plt.ylabel('x3')
plt.legend(loc='upper right')
plt.savefig('state_estimate.png', dpi=300)
plt.close()

plt.figure(figsize = [10,10])
plt.subplot(211)
plt.plot(tOUT, pOUT[0, :], label = 'x1 Covariance')
plt.plot(tOUT, pOUT[1, :], label = 'x2 Covariance')
plt.plot(tOUT, pOUT[2, :], label = 'x3 Covariance')
plt.ylabel('Covariance')
plt.legend(loc='upper right')

plt.subplot(212)
plt.plot(tOUT, eOUT[0, :], label = 'x1 Error')
plt.plot(tOUT, eOUT[1, :], label = 'x2 Error', alpha = .5)
plt.plot(tOUT, eOUT[2, :], label = 'x3 Error', alpha = .5)
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.savefig('cov_error.png', dpi=300)

# plt.subplot(222)
# plt.plot(tOUT, xOUT[1, :], label = 'Real State')
# plt.plot(tOUT, x_hatOUT[1, :], label = 'Estimated State', alpha = .45)
# plt.ylabel('Velocity')
# plt.legend()
# plt.show()
