"""
Same as acrobot.py, except that the functions inside acro_utils are
now copied into this file to avoid problems with imports from my code
in Python3 (the import statements in acrobot.py are for Python 2)
"""

import numpy as np
import sys
import os
import numbers

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
			   "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range

    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar

    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

def fromAtoB(x1, y1, x2, y2, color='k', connectionstyle="arc3,rad=-0.4",
             shrinkA=10, shrinkB=10, arrowstyle="fancy", ax=None):
    """
    Draws an arrow from point A=(x1,y1) to point B=(x2,y2) on the (optional)
    axis ``ax``.

    .. note::

        See matplotlib documentation.

    """
    if ax is None:
        return pl.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )
    else:
        return ax.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )

class Acrobot(object):
	"""
	Acrobot is a 2-link pendulum with only the second joint actuated
	Intitially, both links point downwards. The goal is to swing the
	end-effector at a height at least the length of one link above the base.

	Both links can swing freely and can pass by each other, i.e., they don't
	collide when they have the same angle.

	**STATE:**
	The state consists of the two rotational joint angles and their velocities
	[theta1 theta2 thetaDot1 thetaDot2]. An angle of 0 corresponds to corresponds
	to the respective link pointing downwards (angles are in world coordinates).

	**ACTIONS:**
	The action is either applying +1, 0 or -1 torque on the joint between
	the two pendulum links.

	.. note::

		The dynamics equations were missing some terms in the NIPS paper which
		are present in the book. R. Sutton confirmed in personal correspondance
		that the experimental results shown in the paper and the book were
		generated with the equations shown in the book.

		However, there is the option to run the domain with the paper equations
		by setting book_or_nips = 'nips'

	**REFERENCE:**

	.. seealso::
		R. Sutton: Generalization in Reinforcement Learning:
		Successful Examples Using Sparse Coarse Coding (NIPS 1996)

	.. seealso::
		R. Sutton and A. G. Barto:
		Reinforcement learning: An introduction.
		Cambridge: MIT press, 1998.

	.. warning::

		This version of the domain uses the Runge-Kutta method for integrating
		the system dynamics and is more realistic, but also considerably harder
		than the original version which employs Euler integration,
		see the AcrobotLegacy class.
	"""
	
	state_names = ("LINK1_POS","LINK2_POS", "VEL_1","VEL_2")
	
	def __init__(self, dt=.2, model_derivatives=None,**kw):
		
		continuous_dims = np.arange(4)
		discount_factor = 1.

		self.LINK_LENGTH_1 = 1.  # [m]
		self.LINK_LENGTH_2 = 1.  # [m]
		self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
		self.LINK_MASS_2 = 1.  #: [kg] mass of link 2
		self.LINK_COM_POS_1 = 0.5*self.LINK_LENGTH_1  #: [m] position of the center of mass of link 1
		self.LINK_COM_POS_2 = 0.5*self.LINK_LENGTH_2  #: [m] position of the center of mass of link 2
		self.LINK_MOI = 1.  #: moments of inertia for both links

		self.perturb_params = ('p_LINK_LENGTH_1','p_LINK_LENGTH_2','p_LINK_MASS_1','p_LINK_MASS_2')

		self.MAX_VEL_1 = 4 * np.pi
		self.MAX_VEL_2 = 9 * np.pi

		self.AVAIL_TORQUE = [-1., 0., +1.]
		self.num_actions = 3

		self.torque_noise_max = 0. # Can add random jitter on the order of this variable to the applied torque
		statespace_limits = np.array([[-np.pi, np.pi]] * 2
										+ [[-self.MAX_VEL_1, self.MAX_VEL_1]]
										+ [[-self.MAX_VEL_2, self.MAX_VEL_2]])

		self.target = np.array([np.pi,0,0,0]) # The target state that we are hoping to get the Acrobot to


		#: use dynamics equations from the nips paper or the book
		self.book_or_nips = "book"
		self.action_arrow = None
		self.domain_fig = None
		self.actions_num = 3


		if model_derivatives is None:
			model_derivatives = self.dsdt
		self.model_derivatives = model_derivatives
		self.dt = dt
		self.reset(**kw) 

	def reset(self,random_start_state=False, assign_state = False, n=None, k = None, \
		perturb_params = False, p_LINK_LENGTH_1 = 0, p_LINK_LENGTH_2 = 0, \
		p_LINK_MASS_1 = 0, p_LINK_MASS_2 = 0, **kw):
		self.t = 0
		self.state = np.random.uniform(low=-0.1,high=0.1,size=(4,))

		self.LINK_LENGTH_1 = 1.  # [m]
		self.LINK_LENGTH_2 = 1.  # [m]
		self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
		self.LINK_MASS_2 = 1.

		if perturb_params:
			self.LINK_LENGTH_1 += (self.LINK_LENGTH_1 * p_LINK_LENGTH_1)  # [m]
			self.LINK_LENGTH_2 += (self.LINK_LENGTH_2 * p_LINK_LENGTH_2)  # [m]
			self.LINK_MASS_1 += (self.LINK_MASS_1 * p_LINK_MASS_1)  #: [kg] mass of link 1
			self.LINK_MASS_2 += (self.LINK_MASS_2 * p_LINK_MASS_2)  #: [kg] mass of link 2
		
		# The idea here is that we can initialize our batch randomly so that we can get
		# more variety in the state space that we attempt to fit a policy to.
		if random_start_state:
			self.state[:2] = np.random.uniform(-np.pi,np.pi,size=2)

		if assign_state:
			self.state[0] = wrap((2*k*np.pi)/(1.0*n),-np.pi,np.pi)

	def observe(self):
		return self.state

	def is_terminal(self, state=None):
		if state is None:
			s = self.state
		else:
			s = state

		hinge, foot = self.get_cartesian_points(s)
		return bool( foot[0] >= (self.LINK_LENGTH_1) )

	def is_done(self, episode_length = 250, state=None):
		if state is None:
			s = self.observe()
		else:
			s = state
		if (self.t >= episode_length) or self.is_terminal(s):
			return True
		else:
			return False

	def perform_action(self, a , **kw):
		self.t += 1
		s = self.state
		torque = self.AVAIL_TORQUE[a]

		# Add noise to the force action
		if self.torque_noise_max > 0:
			torque += self.random_state.uniform(-self.torque_noise_max, self.torque_noise_max)

		# Now, augment the state with our force action so it can be passed to _dsdt
		s_augmented = np.append(s, torque)

		ns = rk4(self.dsdt, s_augmented, [0, self.dt])
		# only care about final timestep of integration returned by integrator
		ns = ns[-1]
		ns = ns[:4]  # omit action
		# ODEINT IS TOO SLOW! [rlpy note]
		# ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
		# self.s_continuous = ns_continuous[-1] # We only care about the state
		# at the ''final timestep'', self.dt

		ns[0] = wrap(ns[0], -np.pi, np.pi)
		ns[1] = wrap(ns[1], -np.pi, np.pi)
		ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
		ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
		self.state = ns.copy()
		
		reward = self.calc_reward()
		return reward, self.observe()

	def calc_reward(self, action = None, state = None , **kw ):
		'''Calculates the continuous reward based on the height of the foot (y position) 
		with a penalty applied if the hinge is moving (we want the acrobot to be upright
		and stationary!), which is then normalized by the combined lengths of the links'''
		t = self.target
		if state is None:
			s = self.state
		else:
			s = state
			# Make sure that input state is clipped/wrapped to the given bounds (not guaranteed when coming from the BNN)
			s[0] = wrap( s[0] , -np.pi , np.pi )
			s[1] = wrap( s[1] , -np.pi , np.pi )
			s[2] = bound( s[2] , -self.MAX_VEL_1 , self.MAX_VEL_1 )
			s[3] = bound( s[3] , -self.MAX_VEL_1 , self.MAX_VEL_1 )
		
		hinge, foot = self.get_cartesian_points(s)
		reward = -0.05 * (foot[0] - self.LINK_LENGTH_1)**2

		terminal = self.is_terminal(s)
		return 10 if terminal else reward

	def get_cartesian_points(self, s):
		"""Return the state as the cartesian location of the hinge and foot."""
		# Position of the hinge
		p1 = [-self.LINK_LENGTH_1 * np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])] 
		# Position of the foot
		p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]), p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])] 
	
		return p1, p2
		
	def dsdt(self, s_augmented, t):
		derivs = np.empty_like(s_augmented)
		self._dsdt(derivs, s_augmented, t)
		return derivs

	def _dsdt(self, out, s_augmented, t):
		m1 = self.LINK_MASS_1
		m2 = self.LINK_MASS_2
		l1 = self.LINK_LENGTH_1
		lc1 = self.LINK_COM_POS_1
		lc2 = self.LINK_COM_POS_2
		I1 = self.LINK_MOI
		I2 = self.LINK_MOI
		g = 9.8
		a = s_augmented[-1]
		s = s_augmented[:-1]
		theta1 = s[0]
		theta2 = s[1]
		dtheta1 = s[2]
		dtheta2 = s[3]
		d1 = m1 * lc1 ** 2 + m2 * \
			(l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
		d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
		phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
		phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
			   - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
			+ (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
		if self.book_or_nips == "nips":
			# the following line is consistent with the description in the
			# paper
			ddtheta2 = (a + d2 / d1 * phi1 - phi2)/(m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
		else:
			# the following line is consistent with the java implementation and the
			# book
			ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
		
		ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
		out[0] = dtheta1
		out[1] = dtheta2
		out[2] = ddtheta1
		out[3] = ddtheta2
		out[4] = 0
