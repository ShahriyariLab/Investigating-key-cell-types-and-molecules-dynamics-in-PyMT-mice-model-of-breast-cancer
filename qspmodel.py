'''
qspmodel: classes and methods for analysis of
          Mathematical model of immune response in cancer*

Breast_QSP_Functions: class containing functions and jacobians related to
            Data-driven Mathematical model of immune response in colon cancer
QSP: general class containing methods for analysis of
          Mathematical model of immune response in cancer

Author: Arkadz Kirshtein, https://sites.google.com/site/akirshtein/
(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
Modified By: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei

Conceptualization of sensitivity algorithm by Wenrui Hao, http://personal.psu.edu/wxh64/
'''

# from odesolver import *
import numpy as np
import scipy.optimize as op
from itertools import permutations
from scipy.integrate import odeint
import pandas as pd


def Sensitivity_function(x,t,QSP):
    nparam=QSP.qspcore.nparam
    nvar=QSP.qspcore.nvar
    x=x.reshape(nparam+1,nvar)
    dxdt=np.empty(x.shape)
    dxdt[0]=QSP(x[0],t)
    dxdt[1:]=np.dot(QSP.Ju(x[0],t),x[1:].T).T+ QSP.Jp(x[0],t)
    return dxdt.flatten()


# Object class that defines the functions for the appropriate QSP Model
class Breast_QSP_Functions(object):
    def __init__(self,SSrestrictions=np.ones(28)):
        self.nparam=60
        self.nvar=15
        self.SSscale = SSrestrictions
        self.variable_names=['Naive T-cells', 'helper T-cells', 'cytotoxic cells', 'Treg-cells', 'Naive Dendritic cells', 'Dendritic cells', 'Naive Macrophages', 'Macrophages',
                        'Cancer cells', 'Necrotic cells','Adipocytes', 'HMGB1','IL-12','IL-10', 'IL-6']
        self.parameter_names=['lambda_{T_hH}','lamda_{T_hD}','lambda_{T_hIL_{12}}',
                         'lambda_{T_cD}','lambda_{T_cIL_{12}}',
                         'lambda_{T_rD}',
                         'lambda_{DC}','lambda_{DH}',
                         'lambda_{MIL_{10}}','lambda_{MIL_{12}}','lambda_{MT_h}',
                         'lambda_{C}','lambda_{CIL_6}','lambda_{CA}',
                         'lambda_{A}',
                         'lambda_{HD}','lambda_{HN}','lambda_{HM}','lambda_{HT_c}','lambda_{HC}',
                         'lambda_{IL_{12}M}','lambda_{IL_{12}D}','lambda_{IL_{12}T_h}','lambda_{IL_{12}T_c}',
                         'lambda_{IL_{10}M}','lambda_{IL_{10}D}','lambda_{IL_{10}T_r}','lambda_{IL_{10}T_h}','lambda_{IL_{10}T_c}','lambda_{IL_{10}C}',
                         'lambda_{IL_6A}','lambda_{IL_6M}','lambda_{IL_6D}',
                         'delta_{T_hT_r}','delta_{T_hIL_{10}}','delta_{T_h}',
                         'delta_{T_CT_r}','delta_{T_cIL_{10}}','delta_{T_c}',
                         'delta_{T_r}',
                         'delta_{T_N}',
                         'delta_{DC}','delta_{D}',
                         'delta_{D_N}',
                         'delta_{M}',
                         'delta_{M_N}',
                         'delta_{CT_c}','delta_{C}',
                         'delta_{A}',
                         'delta_{N}',
                         'delta_{H}',
                         'delta_{IL_{12}}',
                         'delta_{IL_{10}}',
                         'delta_{IL_6}',
                         'A_{T_N}','A_{D_N}','A_{M}',
                         'alpha_{NC}','C_0','A_0']

    def __call__(self,x,t,p):
        # ODE right-hand side
        return np.array([p[54]-(p[0]*x[11]+p[1]*x[5]+p[2]*x[12])*x[0]-(p[3]*x[5]+p[4]*x[12])*x[0]-(p[5]*x[5]+p[40])*x[0],\
                        (p[0]*x[11]+p[1]*x[5]+p[2]*x[12])*x[0]-(p[33]*x[3]+p[34]*x[13]+p[35])*x[1],\
                        (p[3]*x[5]+p[4]*x[12])*x[0]-(p[36]*x[3]+p[37]*x[13]+p[38])*x[2],\
                        (p[5]*x[5])*x[0]-p[39]*x[3],\
                        p[55]-(p[6]*x[8]+p[7]*x[11])*x[4]-p[43]*x[4],\
                        (p[6]*x[8]+p[7]*x[11])*x[4]-(p[41]*x[8]+p[42])*x[5],\
                        p[56]-(p[8]*x[13]+p[9]*x[12]+p[10]*x[1]+p[45])*x[6],\
                        (p[8]*x[13]+p[9]*x[12]+p[10]*x[1])*x[6]-p[44]*x[7],\
                        (p[11]+p[12]*x[14]+p[13]*x[10])*(1-x[8]/p[58])*x[8]-(p[46]*x[2]+p[47])*x[8],\
                        p[57]*(p[46]*x[2]+p[47])*x[8]-p[49]*x[9],\
                        (p[14]*x[10])*(1-x[10]/p[59])-p[48]*x[10],\
                        p[15]*x[5]+p[16]*x[9]+p[17]*x[7]+p[18]*x[2]+p[19]*x[8]-p[50]*x[11],\
                        p[20]*x[7]+p[21]*x[5]+p[22]*x[1]+p[23]*x[2]-p[51]*x[12],\
                        p[24]*x[7]+p[25]*x[5]+p[26]*x[3]+p[27]*x[1]+p[28]*x[2]+p[29]*x[8]-p[52]*x[13],\
                        p[30]*x[10]+p[31]*x[7]+p[32]*x[5]-p[53]*x[14]])
    def Ju(self,x,t,p):

        # Jacobian with respect to variables
        return np.array([[-p[40] - p[0]*x[11] - p[2]*x[12] - p[4]*x[12] - p[1]*x[5] - p[3]*x[5] - p[5]*x[5], 0, 0, 0, 0, -p[1]*x[0] - p[3]*x[0] - p[5]*x[0], 0, 0, 0, 0, 0, -p[0]*x[0], -p[2]*x[0] - p[4]*x[0], 0, 0],\
                        [p[0]*x[11] + p[2]*x[12] + p[1]*x[5], -p[35] - p[34]*x[13] - p[33]*x[3], 0, -p[33]*x[1], 0, p[1]*x[0], 0, 0, 0, 0, 0, p[0]*x[0], p[2]*x[0], -p[34]*x[1], 0],\
                        [p[4]*x[12] + p[3]*x[5], 0, -p[38] - p[37]*x[13] - p[36]*x[3], -p[36]*x[2], 0, p[3]*x[0], 0, 0, 0, 0, 0, 0, p[4]*x[0], -p[37]*x[2], 0],\
                        [p[5]*x[5], 0, 0, -p[39], 0, p[5]*x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                        [0, 0, 0, 0, -p[43] - p[7]*x[11] - p[6]*x[8], 0, 0, 0, -p[6]*x[4], 0, 0, -p[7]*x[4], 0, 0, 0],\
                        [0, 0, 0, 0, p[7]*x[11] + p[6]*x[8], -p[42] - p[41]*x[8], 0, 0, p[6]*x[4] - p[41]*x[5], 0, 0, p[7]*x[4], 0, 0, 0],\
                        [0, -p[10]*x[6], 0, 0, 0, 0, -p[45] - p[10]*x[1] - p[9]*x[12] - p[8]*x[13], 0, 0, 0, 0, 0, -p[9]*x[6], -p[8]*x[6], 0],\
                        [0, p[10]*x[6], 0, 0, 0, 0, p[10]*x[1] + p[9]*x[12] + p[8]*x[13], -p[44], 0, 0, 0, 0, p[9]*x[6], p[8]*x[6], 0],\
                        [0, 0, -p[46]*x[8], 0, 0, 0, 0, 0, -p[47] - p[46]*x[2] - ((p[11] + p[13]*x[10] + p[12]*x[14])*x[8])/p[58] + (p[11] + p[13]*x[10] + p[12]*x[14])*(1 - x[8]/p[58]), 0, p[13]*x[8]*(1 - x[8]/p[58]), 0, 0, 0, p[12]*x[8]*(1 - x[8]/p[58])],\
                        [0, 0, p[46]*p[57]*x[8], 0, 0, 0, 0, 0, p[57]*(p[47] + p[46]*x[2]), -p[49], 0, 0, 0, 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -p[48] - (p[14]*x[10])/p[59] + p[14]*(1 - x[10]/p[59]), 0, 0, 0, 0],\
                        [0, 0, p[18], 0, 0, p[15], 0, p[17], p[19], p[16], 0, -p[50], 0, 0, 0],\
                        [0, p[22], p[23], 0, 0, p[21], 0, p[20], 0, 0, 0, 0, -p[51], 0, 0],\
                        [0, p[27], p[28], p[26], 0, p[25], 0, p[24], p[29], 0, 0, 0, 0, -p[52], 0],\
                        [0, 0, 0, 0, 0, p[32], 0, p[31], 0, 0, p[30], 0, 0, 0, -p[53]]])
    def Jp(self,x,t,p):
        # Jacobian with respect to the parameters
        return np.array([[-x[0]*x[11], x[0]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[12], x[0]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], 0, x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[12], 0, x[0]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], 0, 0, x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, -x[4]*x[8], x[4]*x[8], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, -x[11]*x[4], x[11]*x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[13]*x[6], x[13]*x[6], 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[12]*x[6], x[12]*x[6], 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[1]*x[6], x[1]*x[6], 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, x[8]*(1 - x[8]/p[58]), 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, x[14]*x[8]*(1 - x[8]/p[58]), 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, x[10]*x[8]*(1 - x[8]/p[58]), 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[10]*(1 - x[10]/p[59]), 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[9], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[10]],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7]],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5]],\
                      [0, -x[1]*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, -x[1]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, -x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[2]*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[13]*x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, -x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [-x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, -x[5]*x[8], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, -x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, -x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, -x[7], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, -x[6], 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, -x[2]*x[8], p[57]*x[2]*x[8], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, -x[8], p[57]*x[8], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[10], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -x[9], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[11], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[12], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[13], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[14]],\
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, (p[47] + p[46]*x[2])*x[8], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, ((p[11] + p[13]*x[10] + p[12]*x[14])*(x[8]**2))/(p[58]**2), 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (p[14]*(x[10]**2))/(p[59]**2), 0, 0, 0, 0]])


class QSP:
    def __init__(self,parameters,qspcore=Breast_QSP_Functions()):
        self.qspcore=qspcore
        self.p=parameters;
    def steady_state(self):
        # compute steady state with current parameters
        IC=np.ones(self.qspcore.nvar);
        return op.fsolve((lambda x: self.qspcore(x,0,self.p)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,self.p)),xtol=1e-7,maxfev=10000)  #This might need to change
    def Sensitivity(self,method='steady',t=None,IC=None,params=None,variables=None):
        # Sensitivity matrix
        # method: (default) 'steady' - steady state sensitivity
                # 'time' - time-integrated sensitivity
                        # requires time array t and initial conditions IC
                # 'split' - steady state sensitivity with respect to chosen parameters
                        # requires initiate_parameter_split to have been run
                        # takes optional argument 'variables' for sensitivity of specific variables.
        if method=='time':
            if IC is None:
                raise Exception('Error: Need initial conditions for time integration. Set IC=')
                return None
            if t is None:
                raise Exception('Error: Need time values for time integration. Set t=')
                return None

            nparam=self.qspcore.nparam
            nvar=self.qspcore.nvar
            initial=np.zeros((nparam+1,nvar));
            initial[0]=IC
            return np.mean(odeint(Sensitivity_function, initial.flatten(), t, args=(self, )) ,axis=0).reshape(nparam+1,nvar)[1:]
        elif method=='split':
            if not hasattr(self,'variable_par'):
                raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
                return None
            if params is None:
                raise Exception('error: Need parameter values for split sensitivity. Set params=')
                return None
            elif len(params)!=sum(self.variable_par):  #what is variable_par?
                raise Exception('error: wrong number of parameters given')
                return None

            if IC is None:
                IC=np.ones(self.qspcore.nvar);
            par=np.copy(self.p)
            par[self.variable_par]=np.copy(params)

            u=op.fsolve((lambda x: self.qspcore(x,0,par)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,par)),xtol=1e-7,maxfev=10000)
            if variables is None:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par]
            else:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par,variables]
        else:
            u=self.steady_state()
            return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))
    def __call__(self,x,t):
        return self.qspcore(x,t,self.p)
    def Ju(self,x,t):
        return self.qspcore.Ju(x,t,self.p)
    def Jp(self,x,t):
        return self.qspcore.Jp(x,t,self.p)
    def variable_names(self):return self.qspcore.variable_names
    def parameter_names(self):return self.qspcore.parameter_names
    def solve_ode(self, t, IC, method='default'):
        # Solve ode system with either default 1e4 time steps or given time discretization
        # t - time: for 'default' needs start and end time
        #           for 'given' needs full array of time discretization points
        # IC - initial conditions
        # method: 'default' - given interval divided by 10000 time steps
        #         'given' - given time discretization
        if method=='given':
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, t,
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), t
        else:
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, np.linspace(min(t), max(t), 10001),
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), np.linspace(min(t), max(t), 10001)

    def initiate_parameter_split(self,variable_par):
        # splits the parameters into fixed and variable for further fittin
        # variable_par - boolean array same size as parameter array indicating which parameters are variable
        if (variable_par.dtype!='bool') or (len(variable_par)!=self.qspcore.nparam):
            raise Exception('error: wrong parameter indicator')
            return None
        self.variable_par=np.copy(variable_par)

    def solve_ode_split(self, t, IC, params):
        # Solve ode system with adjusted variable parameters
        #   using either default 1e4 time steps or given time discretization
        # t - time: needs full array of time discretization points
        # IC - initial conditions
        # params - parameters to update for this solution
        if not hasattr(self,'variable_par'):
            raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
            return None
        if len(params)!=sum(self.variable_par):
            raise Exception('error: wrong number of parameters given')
            return None
        par=np.copy(self.p)
        par[self.variable_par]=np.copy(params)
        return odeint((lambda x,t: self.qspcore(x,t,par)), IC, t,Dfun=(lambda x,t: self.qspcore.Ju(x,t,par)))