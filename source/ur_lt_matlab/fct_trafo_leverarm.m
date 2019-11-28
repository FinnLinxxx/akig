function [adj,data] = fct_trafo_leverarm(var,data,adj)
%FCT_TRAFO_LEVERARM builds up the functional model for an integrated
%determination of the transformation of a sensor in respective to the robot
%coordinate system as well as the leverarm between the robot arm flange and
%the measured point
%   Input variables:
%       var      ... structure including var.cov or var.identPts
%       data     ... structure including:
%         data.R  ... data of the robot arm
%         data.LT ... data of the sensor
%       adj       ... structure, variables mainly defined in this function
%   Output varibles:
%       adj       ... defines all adjustment variables for the
%                     gauss_helmert.m
%       data      ... if 'var.identPts = True' -> data is extended by LT1
%                     and LT2
%
% Sabine Horvath, TU Wien, 22.05.2019

%% Definition of Measures

% R_LT_T, R_RF_T (SA)
syms eulX eulY eulZ eulX1 eulY1 eulZ1 eulX2 eulY2 eulZ2 om1 phi1 kap1 om2 phi2 kap2 a1 a2 a3 z1 z2 z3 % moX moY moZ 
c1 = cos(eulX); s1 = sin(eulX);
c2 = cos(eulY); s2 = sin(eulY);
c3 = cos(eulZ); s3 = sin(eulZ);
RxA = [1 0 0; 0 c1 s1; 0 -s1 c1];
RyB = [c2 0 -s2; 0 1 0; s2 0 c2];
RzC = [c3 s3 0; -s3 c3 0; 0 0 1];
rotTT = RzC*RyB*RxA;
rotT = rotTT';


% Obserations
t_R_RF = sym('t_R_RF',[3 1]);
R_R_RF = subs(rotT,{eulX,eulY,eulZ},{z1,z2,z3});

R_LT_T = rotT;
t_LT_T = sym('t_LT_T',[3 1]);

R_LT_T1 = subs(rotT,{eulX,eulY,eulZ},{eulX1,eulY1,eulZ1});
t_LT_T1 = sym('t_LT_T1',[3 1]);

R_LT_T2 = subs(rotT,{eulX,eulY,eulZ},{eulX2,eulY2,eulZ2});
t_LT_T2 = sym('t_LT_T2',[3 1]);


% Unknowns
if ~var.leverarmRF
    t_T_RF = sym('t_T_RF',[3 1]);
    % R_RF_T = subs(rotT,{eulX,eulY,eulZ},{moX,moY,moZ});
    t_R_LT = sym('t_R_LT',[3 1]);
    R_R_LT = subs(rotT,{eulX,eulY,eulZ},{a1,a2,a3});
else
    t_LT_R = sym('t_LT_R',[3 1]);
    R_LT_R = subs(rotT,{eulX,eulY,eulZ},{a1,a2,a3});
    t_RF_T = sym('t_RF_T',[3 1]);
end


%% Functional Model

if ~var.leverarmRF
    g = R_R_LT * R_LT_T * t_T_RF + R_R_LT * t_LT_T + t_R_LT - t_R_RF; % = 0
    adj.gg = matlabFunction(g);
    
    % Unkn: R_R_LT(a1,2,3), t_R_LT, t_T_RF
    AA = [diff(g,a1),diff(g,a2),diff(g,a3),diff(g,t_R_LT(1)),...
        diff(g,t_R_LT(2)),diff(g,t_R_LT(3)),diff(g,t_T_RF(1)),...
        diff(g,t_T_RF(2)),diff(g,t_T_RF(3))]; %,diff(g,t_T_RF(3))
    adj.aa = matlabFunction(AA);
    
    BB = [diff(g,eulX), diff(g,eulY), diff(g,eulZ), ...
        diff(g,t_LT_T(1)),diff(g,t_LT_T(2)),diff(g,t_LT_T(3))...
        diff(g,t_R_RF(1)),diff(g,t_R_RF(2)),diff(g,t_R_RF(3))];
    adj.bb = matlabFunction(BB);

elseif var.leverarmRF

    g = R_LT_R * R_R_RF * t_RF_T + R_LT_R * t_R_RF + t_LT_R - t_LT_T; % = 0
    adj.gg = matlabFunction(g,'Vars',[a1;a2;a3;t_LT_R;t_RF_T;t_LT_T;z1;z2;z3;t_R_RF]);
    
    % Unkn: R_R_LT(a1,2,3), t_R_LT, t_T_RF
    AA = [diff(g,a1),diff(g,a2),diff(g,a3),diff(g,t_LT_R(1)),...
        diff(g,t_LT_R(2)),diff(g,t_LT_R(3)),diff(g,t_RF_T(1)),...
        diff(g,t_RF_T(2)),diff(g,t_RF_T(3))]; %,diff(g,t_T_RF(3))
    adj.aa = matlabFunction(AA,'Vars',[a1;a2;a3;t_LT_R;t_RF_T;z1;z2;z3;t_R_RF]);
    
    BB = [diff(g,t_LT_T(1)),diff(g,t_LT_T(2)),diff(g,t_LT_T(3)),...
        diff(g,z1), diff(g,z2), diff(g,z3),diff(g,t_R_RF(1)),...
        diff(g,t_R_RF(2)),diff(g,t_R_RF(3))];
    adj.bb = matlabFunction(BB,'Vars',[a1;a2;a3;t_RF_T;z1;z2;z3]);
end


%% Arrange A, B, w and Qll

adj.naeh = adj.x;

adj.samples = size(data.LT,1);
adj.u = length(adj.x);
adj.bed = adj.samples*3; % number of constraints


if ~var.leverarmRF
    ll = [data.LT(:,5:7),data.LT(:,2:4),data.R(:,2:4)]';
    adj.ll = ll(:);
    adj.nn = 9;
    adj.beob = adj.samples*adj.nn;
     
elseif var.leverarmRF
    ll = [data.LT(:,2:4),data.R(:,5:7),data.R(:,2:4)]';
    adj.ll = ll(:);
    adj.nn = 9;
    adj.beob = adj.samples*adj.nn;
end

adj.A = zeros(adj.bed,adj.u);
adj.B = zeros(adj.bed,adj.beob);
adj.w = zeros(adj.bed,1);
adj.Qll = eye(adj.beob,adj.beob);
adj.l0 = adj.ll;


end