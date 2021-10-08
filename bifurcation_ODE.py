'''
Bifurcation of cancer based on user picked parameters

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''

from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qspmodel import *
import csv

###############################################################
#Reading files
max_values= pd.read_csv('input/input/max_values.csv')
max_values = max_values.to_numpy()
IC=np.array(pd.read_csv('input/input/IC_data_ND.csv'))
parameters = pd.read_csv('input/input/parameters.csv').to_numpy()
###############################################################

###############################################################
#ODE Parameters non-dimensional
Pars =['\lambda_{T_hH}','\lambda_{T_hD}','\lambda_{T_hIL_{12}}',
                        '\lambda_{T_cD}','\lambda_{T_cIL_{12}}',
                        '\lambda_{T_rD}',
                        '\lambda_{DC}','\lambda_{DH}',
                        '\lambda_{MIL_{10}}','\lambda_{MIL_{12}}','\lambda_{MT_h}',
                        '\lambda_{C}','\lambda_{CIL_6}','\lambda_{CA}',
                        '\lambda_{A}',
                        '\lambda_{HD}','\lambda_{HN}','\lambda_{HM}','\lambda_{HT_c}','\lambda_{HC}',
                        '\lambda_{IL_{12}M}','\lambda_{IL_{12}D}','\lambda_{IL_{12}T_h}','\lambda_{IL_{12}T_c}',
                        '\lambda_{IL_{10}M}','\lambda_{IL_{10}D}','\lambda_{IL_{10}T_r}','\lambda_{IL_{10}T_h}','\lambda_{IL_{10}T_c}','\lambda_{IL_{10}C}',
                        '\lambda_{IL_6A}','\lambda_{IL_6M}','\lambda_{IL_6D}',
                        '\delta_{T_hT_r}','\delta_{T_hIL_{10}}','\delta_{T_h}',
                        '\delta_{T_cIL_{10}}','\delta_{T_CT_r}','\delta_{T_c}',
                        '\delta_{T_r}',
                        '\delta_{T_N}',
                        '\delta_{DC}','\delta_{D}',
                        '\delta_{D_N}',
                        '\delta_{M}',
                        '\delta_{M_N}',
                        '\delta_{CT_c}','\delta_{C}',
                        '\delta_{A}',
                        '\delta_{N}',
                        '\delta_{H}',
                        '\delta_{IL_{12}}',
                        '\delta_{IL_{10}}',
                        '\delta_{IL_6}',
                        'A_{T_N}','A_{D_N}','A_{M}',
                        '\\alpha_{NC}','C_0','A_0']
par_list = ["$"+x+"$" for x in Pars]
Vars = ['Tn','Th','Tc','Tr','Dn','D','Mn','M','C','N','A','H','IL12','IL10','IL6']
###############################################################

###############################################################
idx = input("Type in the number for the parameter you want to bifurcate:")
###############################################################

#########################################################################################
def ODE_sys(x,t,p):
    RHS_Tn = p[54]-(p[0]*x[11]+p[1]*x[5]+p[2]*x[12])*x[0]-(p[3]*x[5]+p[4]*x[12])*x[0]-(p[5]*x[5]+p[40])*x[0]
    RHS_Th = (p[0]*x[11]+p[1]*x[5]+p[2]*x[12])*x[0]-(p[33]*x[3]+p[34]*x[13]+p[35])*x[1]
    RHS_Tc = (p[3]*x[5]+p[4]*x[12])*x[0]-(p[36]*x[3]+p[37]*x[13]+p[38])*x[2]
    RHS_Tr = (p[5]*x[5])*x[0]-p[39]*x[3]
    RHS_Dn = p[55]-(p[6]*x[8]+p[7]*x[11])*x[4]-p[43]*x[4]
    RHS_D = (p[6]*x[8]+p[7]*x[11])*x[4]-(p[41]*x[8]+p[42])*x[5]
    RHS_Mn = p[56]-(p[8]*x[13]+p[9]*x[12]+p[10]*x[1]+p[45])*x[6]
    RHS_M = (p[8]*x[13]+p[9]*x[12]+p[10]*x[1])*x[6]-p[44]*x[7]
    RHS_C = (p[11]+p[12]*x[14]+p[13]*x[10])*(1-x[8]/p[58])*x[8]-(p[46]*x[2]+p[47])*x[8]
    RHS_N = p[57]*(p[46]*x[2]+p[47])*x[8]-p[49]*x[9]
    RHS_A = (p[14]*x[10])*(1-x[10]/p[59])-p[48]*x[10]
    RHS_H = p[15]*x[5]+p[16]*x[9]+p[17]*x[7]+p[18]*x[2]+p[19]*x[8]-p[50]*x[11]
    RHS_IL12 = p[20]*x[7]+p[21]*x[5]+p[22]*x[1]+p[23]*x[2]-p[51]*x[12]
    RHS_IL10 = p[24]*x[7]+p[25]*x[5]+p[26]*x[3]+p[27]*x[1]+p[28]*x[2]+p[29]*x[8]-p[52]*x[13]
    RHS_IL6 = p[30]*x[10]+p[31]*x[7]+p[32]*x[5]-p[53]*x[14]
    return [RHS_Tn ,RHS_Th ,RHS_Tc ,RHS_Tr ,RHS_Dn ,RHS_D ,RHS_Mn ,RHS_M ,RHS_C ,RHS_N ,RHS_A ,RHS_H ,RHS_IL12 ,RHS_IL10,RHS_IL6]
#########################################################################################

########################################################################################
Time interval
t_span = np.linspace(0, 3000, 30000)
#########################################################################################

#########################################################################################
solution1 = odeint(ODE_sys, IC[0,], t_span,args=(parameters[0],))
solution2 = odeint(ODE_sys, IC[1,], t_span,args=(parameters[0],))
solution3 = odeint(ODE_sys, IC[2,], t_span,args=(parameters[0],))

#Save the value of each variable at time 3000 for using in the code BreastCancerAnalysis.
c=csv.writer(open('input/input/steady_state_ODE.csv',"w"))
c.writerows([solution1[30000-1,:],solution2[30000-1,:],solution3[30000-1,:]])
del c
########################################################################################


#########################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


r = np.linspace(0,0.2,100)
t_span = np.linspace(0, 1008, 2016)
bif_1 = [];
bif_2 = [];
bif_3 = [];

#Getting the solutions at the end time
for i in range(len(r)):
    parameters[0,int(idx)]=r[i]
    solution1 = odeint(ODE_sys, IC[0,], t_span,args=(parameters[0],))
    solution2 = odeint(ODE_sys, IC[1,], t_span,args=(parameters[0],))
    solution3 = odeint(ODE_sys, IC[2,], t_span,args=(parameters[0],))
    bif_1.append(solution1[len(t_span)-1,8]*max_values[0,8])
    bif_2.append(solution2[len(t_span)-1,8]*max_values[0,8])
    bif_3.append(solution3[len(t_span)-1,8]*max_values[0,8])
    print(i)


#Plotting
palette = {0:'#FF0000', 1:'#0000FF', 2:'#000000'}
sns.set(font_scale=1.5)
sns.set_style("ticks")
fig, axs = plt.subplots(1, 3, sharey=False, figsize=(15.5,5))
fig.subplots_adjust(wspace=0.5)
axs = axs.flatten()
custom_lines = [Line2D([0], [0], color='#FF0000', lw=2.5),
                Line2D([0], [0], color='#0000FF', lw=2.5),
                Line2D([0], [0], color='#000000', lw=2.5)]


for cluster in range(3):
    axs[cluster].margins(x=0)
    axs[cluster].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if cluster==0:
        axs[cluster].plot(r, bif_1, color=palette[cluster])
    if cluster==1:
        axs[cluster].plot(r, bif_2, color=palette[cluster])
    if cluster==2:
        axs[cluster].plot(r, bif_3, color=palette[cluster])

    axs[cluster].set_xlabel(par_list[int(idx)],fontsize=14)
    axs[cluster].set_ylabel('Cancer cells',fontsize=14)

axs[0].legend(custom_lines,['Mouse 1', 'Mouse 2', 'Mouse 3'], bbox_to_anchor=(4.1, 0.5),loc='center left')

plt.show()
#########################################################################################
