import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy as sp
from qspmodel import *
import numpy as np

# Checking or creating necessary output folders
if not os.path.exists('Data/'):
    os.makedirs('Data/Dynamic/')
    os.makedirs('Data/GlobalSensitivity/')
else:
    if not os.path.exists('Data/Dynamic/'):
        os.makedirs('Data/Dynamic/')
    if not os.path.exists('Data/GlobalSensitivity/'):
        os.makedirs('Data/GlobalSensitivity/')

# some global parameters
lmod=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #indices of immune cells variables in cell data
clusters=3 #number of clusters

T=3000
t=np.linspace(0, T, 30001)

nvar=Breast_QSP_Functions().nvar # number of variables
nparam=Breast_QSP_Functions().nparam # number of parameters

################################################################################
###########################Reading data#########################################
clustercells = pd.read_csv('input/input/steady_state_ODE.csv',header=None)
clustercells = clustercells.to_numpy()

max_values= pd.read_csv('input/input/max_values.csv')
max_values = max_values.to_numpy()

parameters = pd.read_csv('input/input/parameters.csv').to_numpy()


IC=np.array(pd.read_csv('input/input/IC_data_ND.csv'))  #non-dimensional IC

################################################################################
#######################Sensitivity Analysis#####################################

print('Starting steady state global sensitivity analysis')


# Read the local parameter perturbation grid
# level 1 or 0 corresponds to no local perturbation
gridlevel=2
sensitivity_radius=1 # percentage for local perturbation
if gridlevel>1:
    localgridname='Local-level'+str(gridlevel)
    filename='grid60-level'+str(gridlevel)
    data = pd.read_csv('input/input/'+filename+'.csv', header=None).to_numpy()
    w=data[:,0]
    x=data[:,1:]
    del data, filename
else:
    localgridname='Local-nogrid'
    w=np.array([1])
    x=[0]

# coefficients for variable sensitivity
lambda0=np.zeros((nvar,2))
lambda0[8,0]=1 # just cancer

usr_inpt1 = input("Type in 1 if you have already calculated the global sensitivity. Otherwise type int 0:")

if int(usr_inpt1)!=1:

    import time

    start = time.time()
    k_filter_all_clusters = []

    for mouse in range(3):

        print('Starting computations for cluster '+str(mouse+1))

        filename='V60-'+'-'+localgridname+'-cluster-'+str(mouse+1)+'-results'

        lambda0[1:11,1]=clustercells[mouse,lmod]/np.sum(clustercells[mouse,lmod]) # all cells

        #We use one set of parameters for all subjects
        params = parameters[0]
        print(' Parameters set. Computing steady state sensitivity')

        dudp=np.zeros((nparam,2))

        sensitivity_radius=1 # perturbation percentage for sensitivity integration
        for k in range(w.size):
            # set parameter sample
            param_sample=params*(1+(sensitivity_radius*1e-2)*x[k,:])
            QSP_=QSP(param_sample);
            #Calculate cancer and total cell sensitivity analytically
            dudp[:,:2]=dudp[:,:2]+w[k]*np.dot(QSP_.Sensitivity(method='time',t=np.linspace(0,1008,30000),IC=IC[mouse,]),lambda0)

        print(' Writing to file')

        c=csv.writer(open('Data/GlobalSensitivity/'+filename+'sensitivity_mouse_'+str(mouse+1)+'.csv',"w"))
        c.writerows(dudp)
        del c

    end = time.time()
    print('Run time: ', end - start)
    print('Global sensitivity analysis complete')



Par_list =['\lambda_{T_hH}','\lambda_{T_hD}','\lambda_{T_hIL_{12}}',
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
par_list = ["$"+x+"$" for x in Par_list]



################################################################################
######################Plotting Sensitivities####################################
usr_inpt2 = input("Do you want to plot the sensitivies?(yes=1, no=0)")



if int(usr_inpt2)==1:
   for mouse in range(3):
       filename='V60-'+'-'+localgridname+'-cluster-'+str(mouse+1)+'-results'
       sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_mouse_'+str(mouse+1)+'.csv', header=None)
       sensitivity_df.index = par_list
       sensitive_ids = np.abs(sensitivity_df)[0][abs(sensitivity_df)[0]>0].nlargest(n=20).index
       print('Mouse ', str(mouse+1))
       print('Sensitivities:\n', sensitivity_df.loc[sensitive_ids])
       import matplotlib.pyplot as plt
       import seaborn as sns
       plt.rc('xtick',labelsize=13)
       fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,2))
       fig.subplots_adjust(wspace=0.75, hspace=0.75)
       axs[0].set_title('Sensitivity of Cancer')
       axs[1].set_title('Sensitivity of Total cells count')


       filename='V60-'+'-'+localgridname+'-cluster-'+str(mouse+1)+'-results'
       sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_mouse_'+str(mouse+1)+'.csv', header=None)
       sensitivity_df.index = par_list
       for i in range(2):
           sensitive_ids = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index 
           sensitivity_df[i][sensitive_ids[:6]].plot.bar(ax=axs[i], rot=0, width=0.7)
           axs[i].axhline()
           axs[i].set_ylabel('Mouse '+str(mouse+1))
       plt.savefig('fig/sensitivity1_Mouse'+str(mouse+1)+'.eps', format='eps',dpi=300)
       # plt.rc('xtick',labelsize=11)

       fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,2))
       fig.subplots_adjust(wspace=0.75, hspace=0.75)
       axs[0].set_title('Sensitivity of Cancer')
       axs[1].set_title('Sensitivity of Total cells count')


       filename='V60-'+'-'+localgridname+'-cluster-'+str(mouse+1)+'-results'
       sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_mouse_'+str(mouse+1)+'.csv', header=None)
       sensitivity_df.index = par_list
       ids_to_remove = []
       for i in range(2):
           sensitive_ids1 = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index  
           sensitivity_df[i][sensitive_ids1[6:12]].plot.bar(ax=axs[i], rot=0, width=0.7)
           axs[i].axhline()
           axs[i].set_ylabel('Mouse '+str(mouse+1))
       plt.savefig('fig/sensitivity2_Mouse'+str(mouse+1)+'.eps', format='eps',dpi=300)

################################################################################
########################Varying parameters######################################
def plot_cancer_vary_assumption_perturb_sensitive_params(T):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = {0:'#FF0000', 1:'#0000FF', 2:'#000000'}
    alphas = [0.2, 0.2, 0.2]
    restrictions = np.ones(28)

    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    fig, axs = plt.subplots(1, 3, sharey=False, figsize=(15.5,5))
    fig.subplots_adjust(wspace=0.25)
    axs = axs.flatten()
    t = np.linspace(0, T, 10*T+1)
    custom_lines = [Line2D([0], [0], color='#FF0000', lw=2.5),
                    Line2D([0], [0], color='#0000FF', lw=2.5),
                    Line2D([0], [0], color='#000000', lw=2.5)]


    for cluster in range(3):
        perturb_scale=1
        # print('For cluster '+str(cluster)+' and scale='+str(newscale)+' we have \n deltaCIgamma = '+str(new_params[58])+'\n deltaCTc = '+str(new_params[57])+'\n deltaC= '+str(new_params[59])+'.')
        QSP_ = QSP(parameters[0])
        u, _ = QSP_.solve_ode(t, IC[cluster,], 'given')
        u = max_values[0]*u
        umax = umin = u

        max_params={};
        min_params={};
        for param_id in sensitive_param_ids:
            for j in [-perturb_scale, perturb_scale]:
                perturb_arr = np.zeros(len(parameters[0]))
                perturb_arr[param_id] = j
                QSP_ = QSP(parameters[0]*(1+(1e-1)*perturb_arr))
                u_perturb, _ = QSP_.solve_ode(t, IC[cluster,], 'given')
                u_perturb = max_values[0]*u_perturb

                umax = np.maximum(u_perturb, umax)
                umin = np.minimum(u_perturb, umin)

        axs[cluster].margins(x=0)
        axs[cluster].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axs[cluster].fill_between(t, umax[:,8], umin[:,8], facecolor=palette[cluster], alpha=alphas[cluster])
        axs[cluster].plot(t, u[:,8], color=palette[cluster])
        axs[cluster].set_xlabel('time (hours)',fontsize=14)
        axs[cluster].set_ylabel('Cancer cells',fontsize=14)
    axs[0].legend(custom_lines,['Mouse 1', 'Mouse 2', 'Mouse 3'], bbox_to_anchor=(4.1, 0.5),loc='center left')

    plt.show()

usr_inpt3 = input("Do you want to plot the varying dynamics?(yes=1, no=0)")

if int(usr_inpt3)==1:
   sensitive_param_ids = [47,11,12,13,48,14,30,53,31,46,44,58,29,0,8]
   plot_cancer_vary_assumption_perturb_sensitive_params(T=3000)
