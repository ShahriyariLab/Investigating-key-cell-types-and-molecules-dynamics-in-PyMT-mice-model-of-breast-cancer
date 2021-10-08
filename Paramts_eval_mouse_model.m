%**************************************************************************
%**************************************************************************
% A program to evaluate the parameters of breast cancer based on mice data  
% and solve the system of ODEs for the dynamics of the variables
%**************************************************************************
%**************************************************************************
% Author: Navid Changizi
% NIH project, breast cancer, mouse model
% UMass Dartmouth
%**************************************************************************
%**************************************************************************
% If using this or related code please cite: 
% Mohammad Mirzaei, N.; Changizi, N.; Asadpoure, A.; Su, S.; Sufia, D.;
% Tatarova, Z; Zervantonakis, I.K.; Hwan Chang, Y.; Shahriyari, L. 
%     Investigating key cell types and molecules dynamics in  PyMT mice model 
%     of breast cancer through a mathematical model.
%     (Manuscript submitted for publication).
%=================== 15 variables in the system of ODEs ===================
% Tn: Naive T-cells
% Th: Helper T-cells
% Tc: Cytotoxic cells
% Tr: Regulatory cells
% Dn: Naive dendritic cells
% D: Activated dendritic cells
% Mn: Naive macrophages
% M: Macrophages
% C: Cancer cells
% N: Necrotic cells
% A: Cancer associated adipocytes
% H: HMGB1
% IL12: IL-12
% IL10: IL-10
% IL6: IL-6
%==========================================================================
clc; clear;
one_m_out = 0;            % Flag to single out one mouse data in parameter estimation phase
%---------------------------Load the mice data-----------------------------
m1all = xlsread('C:\Users\Mouse data\Mouse1-ND.csv');
m2all = xlsread('C:\Users\Mouse data\Mouse2-ND.csv');
m3all = xlsread('C:\Users\Mouse data\Mouse3-ND.csv');
if one_m_out == 1
    md    = 3;
    Time  = m1all(:,end);   Time_pko  = m3all(:,end);
    mi_p = zeros(8,15);
    % Extract the variables to assemble the matrix with numbering similar to 2020 paper
    mi(1:4,:) = m1all(:,1:15);
    mi(5:8,:) = m2all(:,1:15);
    mi_pko(1:4,:)= m3all(:,1:15);
    % Evaluate the vector of rates
    B1  = bsxfun(@rdivide,mi(3:4,1:15)-mi(1:4-2,1:15),Time(3:end)-Time(1:end-2));
    B2  = bsxfun(@rdivide,mi(7:8,1:15)-mi(5:8-2,1:15),Time(3:end)-Time(1:end-2));
    lvs1 = 2:3;  lvs2 = 6:7;  lvs = [lvs1,lvs2];  nb = 3;
    B = [B1;B2];  B = B';
else
    md    = 3;              % Mouse number to use its initial conditions, and be compared with the dynamics
    Time  = m1all(:,end);   Time_pko  = m3all(:,end);   mi_pko(1:4,:)= m3all(:,1:15);
    if md == 1
        mi_pko(1:4,:)= m1all(:,1:15);
    elseif md == 2
        mi_pko(1:4,:)= m2all(:,1:15);
    elseif md == 3
        mi_pko(1:4,:)= m3all(:,1:15);
    end
    mi_p = zeros(12,15);
    % Extract the variables to assemble the matrix with numbering similar to 2020 paper
    mi(1:4,:) = m1all(:,1:15);
    mi(5:8,:) = m2all(:,1:15);
    mi(9:12,:) = m3all(:,1:15);
    % Evaluate the vector of rates
    B1  = bsxfun(@rdivide,mi(3:4,1:15)-mi(1:4-2,1:15),Time(3:end)-Time(1:end-2));
    B2  = bsxfun(@rdivide,mi(7:8,1:15)-mi(5:8-2,1:15),Time(3:end)-Time(1:end-2));
    B3  = bsxfun(@rdivide,mi(11:12,1:15)-mi(9:12-2,1:15),Time(3:end)-Time(1:end-2));
    lvs1 = 2:3;  lvs2 = 6:7;  lvs3 = 10:11;  lvs = [lvs1,lvs2,lvs3];  nb = 3;
    B = [B1;B2;B3];  B = B';
end
Cmax   = max(max(mi(:,9)),max(mi_pko(:,9)));         
Amax   = max(max(mi(:,10)),max(mi_pko(:,10)));
alphNC = 1.5;
%==========================================================================
% Least-squares optimization
%==========================================================================
Ag = [];  bg = [];
% Loop to assemble
for i = lvs
    A1 = [1 -mi(i,1)*mi(i,12) -mi(i,1)*mi(i,6) -mi(i,1)*mi(i,13)...
        -mi(i,1)*mi(i,6) -mi(i,1)*mi(i,13)...
        -mi(i,1)*mi(i,6) -mi(i,1) zeros(1,49)];
    
    A2 = [zeros(1,1) mi(i,1)*mi(i,12) mi(i,1)*mi(i,6) mi(i,1)*mi(i,13)...
        zeros(1,4) -mi(i,2)*mi(i,4) -mi(i,2)*mi(i,14) -mi(i,2)  zeros(1,46)];
    
    A3 = [zeros(1,4) mi(i,1)*mi(i,6) mi(i,1)*mi(i,13)...
        zeros(1,5) -mi(i,3)*mi(i,4) -mi(i,3)*mi(i,14) -mi(i,3) zeros(1,43)];
    
    A4 = [zeros(1,6) mi(i,1)*mi(i,6) zeros(1,7) -mi(i,4) zeros(1,42)];
    
    A5 = [zeros(1,15) 1 -mi(i,5)*mi(i,9) -mi(i,5)*mi(i,12) -mi(i,5) zeros(1,38)];
    
    A6 = [zeros(1,16) mi(i,5)*mi(i,9) mi(i,5)*mi(i,12)...
        zeros(1,1) -mi(i,6)*mi(i,9) -mi(i,6) zeros(1,36)];
    
    A7 = [zeros(1,21) 1 -mi(i,7)*mi(i,14) -mi(i,7)*mi(i,13) -mi(i,7)*mi(i,2) -mi(i,7) zeros(1,31)];
    
    A8 = [zeros(1,22) mi(i,7)*mi(i,14) mi(i,7)*mi(i,13) mi(i,7)*mi(i,2) ...
        zeros(1,1) -mi(i,8) zeros(1,30)];
    
    C0 = 2*Cmax;
    A9 = [zeros(1,27) mi(i,9)*(1-mi(i,9)/C0) mi(i,9)*(1-mi(i,9)/C0)*mi(i,15) ...
        mi(i,9)*(1-mi(i,9)/C0)*mi(i,10) -mi(i,9)*mi(i,3) -mi(i,9) zeros(1,25)];
    
    A0 = 2*Amax;
    A10 = [zeros(1,32) mi(i,10)*(1-mi(i,10)/A0) -mi(i,10) zeros(1,23)];
    
    A11 = [zeros(1,30) alphNC*mi(i,3)*mi(i,9) alphNC*mi(i,9) zeros(1,2) -mi(i,11) zeros(1,22)];
    
    A12 = [zeros(1,35) mi(i,6) mi(i,11) mi(i,8) mi(i,3) mi(i,9) -mi(i,12) zeros(1,16)];
    
    A13 = [zeros(1,41) mi(i,8) mi(i,6) mi(i,2) mi(i,3) -mi(i,13) zeros(1,11)];
    
    A14 = [zeros(1,46) mi(i,8) mi(i,6) mi(i,4) mi(i,2) mi(i,3) mi(i,9) -mi(i,14) zeros(1,4)];
    
    A15 = [zeros(1,53) mi(i,10) mi(i,8) mi(i,6) -mi(i,15)];
    
    A  = [A1;A2;A3;A4;A5;A6;A7;A8;A9;A10;A11;A12;A13;A14;A15];    % Assemble all the equatios
    Ag = [Ag;A];                                                  % Update the global matrix
    if one_m_out == 1
        if i < lvs1(end)+1
            bg = [bg;B(:,i-1)];                                   % Update the global vector
        elseif i >= lvs2(1)
            bg = [bg;B(:,i-nb)];                                  % Update the global vector
        end
    else
        if i < lvs1(end)+1
            bg = [bg;B(:,i-1)];                                   % Update the global vector
        elseif lvs3(1)> i && i >= lvs2(1)
            bg = [bg;B(:,i-nb)];                                  % Update the global vector
        elseif i >= lvs3(1)
            bg = [bg;B(:,i-nb-2)];                                % Update the global vector
        end
    end
end
x = (Ag'*Ag)^(-1)*Ag'*bg;          % Solution of the least squares method, unconstrained
%===============Solve the constrained least square problem=================
% Algorithms available: 1- interior-point (default), 2- trust-region-reflective, 3- active-set
lb = 1e-5*ones(57,1);                                   % Lower bound
% x0 = .1*ones(57,1);                                   % For algorithms other than interior-point
options = optimoptions('lsqlin','Algorithm','interior-point','MaxIter',2000,'Display','iter');
xopt    = lsqlin(Ag,bg,[],[],[],[],lb,[],[],options);   % Linear least square xopt
%======================== Solve the system of ODEs ========================
delt   = .001;                    % In hours
delt2  = delt/2;                  % Half of time-step
Tnm    = 0:delt:3*Time_pko(end);  % All time-steps
% Initial values
yip1   = mi_pko(1,:)';
yip1_M = zeros(15,length(Tnm));
CKM = 0;                          % To use Cash–Karp method
if CKM == 1
    CK_cofs = [37/378	0	250/621	125/594	0	512/1771];  % Cash–Karp method coefficients
end
%================== Main loop to evaluate the response ====================
for i = 1:length(Tnm)            
    if CKM == 0
        % =-=-=-=-=-=- Classic fourth-order Runge Kutta method =-=-=-=-=-=-
        [dydt1] = RKM_state_derivs(yip1,xopt,Cmax,Amax,alphNC);
        [dydt2] = RKM_state_derivs(yip1+dydt1*delt2,xopt,Cmax,Amax,alphNC);
        [dydt3] = RKM_state_derivs(yip1+dydt2*delt2,xopt,Cmax,Amax,alphNC);
        [dydt4] = RKM_state_derivs(yip1+dydt3*delt,xopt,Cmax,Amax,alphNC);
        %------------------------------------------------------------------
        yip1    = yip1 + (dydt1 + 2*(dydt2 + dydt3) + dydt4)*delt/6;       % Update the solution
        %------------------------------------------------------------------
    elseif CKM == 1
        % =-=-=-=-=-=-=-=-= Cash–Karp Runge Kutta method =-=-=-=-=-=-=-=-=-
        [dydt1] = RKM_state_derivs(yip1,xopt,M0,part_set);
        [dydt2] = RKM_state_derivs(yip1+dydt1*1/5*delt,xopt,M0,part_set);
        [dydt3] = RKM_state_derivs(yip1+(dydt1*3/40 + dydt2*9/40)*delt,xopt,M0,part_set);
        [dydt4] = RKM_state_derivs(yip1+(dydt1*3/10 + dydt2*-9/10 + dydt3*6/5)*delt,xopt,M0,part_set);
        [dydt5] = RKM_state_derivs(yip1+(dydt1*-11/54 + dydt2*5/2 + dydt3*-70/27 + dydt4*35/27)*delt,xopt,M0,part_set);
        [dydt6] = RKM_state_derivs(yip1+(dydt1*1631/55296 + dydt2*175/512 + dydt3*575/13824+ dydt4*44275/110592 + dydt5*253/4096)*delt,xopt,M0,part_set);
        %------------------------------------------------------------------
        yip1    = yip1 + (CK_cofs(1)*dydt1 + CK_cofs(2)*dydt2 + CK_cofs(3)*dydt3 + ...
            CK_cofs(4)*dydt4 + CK_cofs(5)*dydt5 + CK_cofs(6)*dydt6)*delt;  % Update the solution
    end
    %----------------------------------------------------------------------
    yip1_M(:,i) = yip1;                 % Store all solutions
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Plot the dynamics
%==========================================================================
% Plot markers, colors, and labels
%==========================================================================
MarkSyl  = {'o','+','*','x','s','d','^','v','>','<','p','h','s','o','+','*','x','s','d','^','v','>','<','p','h','.'};   % Marker symbole
colorset = {[1 0 0],[0 1 0],[0 0 1],[0 1 1],[1 0 1],[0 0.4470 0.7410],[0 0 0],[0.6350 0.0780 0.1840],...                % Marker color
    [0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880],...
    [0.3010 0.7450 0.9330],[0 0 0],[0 0.4470 0.7410],[1 0 1],[0 1 1],[0 0 1],[0 1 0],[1 0 0],[0 0 0],...
    [0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.6350 0.0780 0.1840],[0.4940 0.1840 0.5560],[0.9290 0.6940 0.1250]};

colorset_subset = {colorset{1},colorset{3},[0 0 0]};
% Labels of the variables
ylblvr  = {'[T_N]','[T_h]','[T_C]','[T_r]','[D_N]','[D]','[M_N]','[M]','[C]','[A]','[N]','[H]','[IL_{12}]','[IL_{10}]','[IL_6]'};
%==========================================================================
figure('Units','normalized','Position',[0 0 1 .6])
tlo = tiledlayout(2,8,'TileSpacing','none','Padding','none');
% Load the maximum values for normalization
maxvar = xlsread('C:\Users\Mouse data\max_values.csv');
for w  = 1:15             
    nexttile
    set(gca,'fontsize',19,'FontName','Times New Roman');  hold on
    if w==13
        ms = 5;  lw = 5;
    elseif w == 14
        ms = 5;  lw = 5;
    else
        ms = 9;  lw = 2;
    end
    plot(Time_pko,mi_pko(:,w)*maxvar(w),'Color',colorset_subset{md},'Marker',MarkSyl{w},'LineStyle','none','MarkerSize',ms,'LineWidth',lw);  hold on
    plot(Tnm,yip1_M(w,:)*maxvar(w),'Color',colorset_subset{md},'LineWidth',3,'MarkerSize',10);  hold on
    if w == 15
        legend('Mouse data','Dynamics','Location','layout','Orientation','horizontal','NumColumns',1,'Position', [0.875 0.106 0.12 0.1],'FontSize',22);
    end
    xlabel('Time (hours)');  ylabel(ylblvr{w});   xticks([0 1500 3000 6500]);
end
%==========================================================================
% Save the figure
saveas(gcf,['C:\Users\mouse',int2str(md) '_data_vs_dynmaics'],'epsc')   
%==========================================================================
function [dydti] = RKM_state_derivs(yip1f,xopti,Cmax,Amax,alphNC)
%==================================================================
% To evaluate f(x,t,u) for each time-step
%==================================================================
C0 = 2*Cmax;
A0 = 2*Amax;
% Evaluate the rate for each variable
g1 = xopti(1) -(xopti(2)*yip1f(12) + xopti(3)*yip1f(6) + xopti(4)*yip1f(13))*yip1f(1)...
    -(xopti(5)*yip1f(6) + xopti(6)*yip1f(13))*yip1f(1)...
    -(xopti(7)*yip1f(6) + xopti(8))*yip1f(1);

g2 = (xopti(2)*yip1f(12) + xopti(3)*yip1f(6) + xopti(4)*yip1f(13))*yip1f(1)...
    -(xopti(9)*yip1f(4) + xopti(10)*yip1f(14) + xopti(11))*yip1f(2);

g3 = (xopti(5)*yip1f(6) + xopti(6)*yip1f(13))*yip1f(1)...
    - (xopti(12)*yip1f(4) + xopti(13)*yip1f(14) + xopti(14))*yip1f(3);

g4 = (xopti(7)*yip1f(6))*yip1f(1) - xopti(15)*yip1f(4);

g5 = xopti(16) -(xopti(17)*yip1f(9) + xopti(18)*yip1f(12))*yip1f(5) - xopti(19)*yip1f(5);

g6 = (xopti(17)*yip1f(9) + xopti(18)*yip1f(12))*yip1f(5) -(xopti(20)*yip1f(9) + xopti(21))*yip1f(6);

g7 = xopti(22) - (xopti(23)*yip1f(14) + xopti(24)*yip1f(13) + xopti(25)*yip1f(2) + xopti(26))*yip1f(7);

g8 = (xopti(23)*yip1f(14) + xopti(24)*yip1f(13) + xopti(25)*yip1f(2))*yip1f(7) - xopti(27)*yip1f(8);

g9 = (xopti(28) + xopti(29)*yip1f(15) + xopti(30)*yip1f(10))*yip1f(9)*(1-yip1f(9)/C0)...
    - (xopti(31)*yip1f(3) + xopti(32))*yip1f(9);

g10 = xopti(33)*yip1f(10)*(1-yip1f(10)/A0) - xopti(34)*yip1f(10);

g11 = alphNC*(xopti(31)*yip1f(3) + xopti(32))*yip1f(9) - xopti(35)*yip1f(11);

g12 = xopti(36)*yip1f(6) + xopti(37)*yip1f(11) + xopti(38)*yip1f(8) + xopti(39)*yip1f(3) + xopti(40)*yip1f(9) - xopti(41)*yip1f(12);

g13 = xopti(42)*yip1f(8) + xopti(43)*yip1f(6) + xopti(44)*yip1f(2) + xopti(45)*yip1f(3) - xopti(46)*yip1f(13);

g14 = xopti(47)*yip1f(8) + xopti(48)*yip1f(6) + xopti(49)*yip1f(4) + xopti(50)*yip1f(2) + xopti(51)*yip1f(3) + xopti(52)*yip1f(9) - xopti(53)*yip1f(14);

g15 = xopti(54)*yip1f(10) + xopti(55)*yip1f(8) + xopti(56)*yip1f(6) - xopti(57)*yip1f(15);
%--------------------------------------------------------------------------
dydti = [g1;g2;g3;g4;g5;g6;g7;g8;g9;g10;g11;g12;g13;g14;g15];    
end


