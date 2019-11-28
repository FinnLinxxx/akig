% data_trafo_TTH
% measurement from 2019-03-26
% computation in apply_trafo_leverarm.m

% necessary problem specific inputs are
%    - the robot and sensor data
%    - the approximate values for the parameters
%    - the stochastic model


%% Data
% Units in m and rad!!!

%path = 'H:\Diss\Rohdaten\UR5\Processing\190528_TrafoTTH_2\';
rob = readtable('juri.csv');
for i = 1:size(rob,1)
    r = rob{i,11:19};
    Rot = [r(1) r(2) r(3);...
        r(4) r(5) r(6);...
        r(7) r(8) r(9)];
    angR(i,:) = ltRot(Rot);
end
data.R = [(1:size(rob,1))',rob{:,6:8},angR];


datLT = readtable('lt.csv');
data.LT = [(1:size(rob,1))',datLT{:,3:5}];



%% Approximate Values

MV = [-30 -115 90]*1e-03; % check Sign
for i = 1:length(data.R)
    rot = ltRot(data.R(i,5:7));
    pt(i,:) = [rot * MV' + data.R(i,2:4)']';
end
[tp,rc,ac,tr] = helmert3d(pt(:,:),data.LT(:,2:4),'7p',1,[0,0,0]);
par = [tp(4:6);tp(1:3)]; % in rad

adj.x = [par',MV]';


if var.leverarmRF    
    MV = [30 115 -90]*1e-03;
    for i = 1:length(data.R)
        rot = ltRot(data.R(i,5:7));
        pt(i,:) = [rot * MV' + data.R(i,2:4)']';
    end
    [tp,rc,ac,tr] = helmert3d(pt(:,:),data.LT(:,2:4),'7p',1,[0,0,0]);
    par = [tp(4:6);tp(1:3)]; % in rad
    
    adj.x = [par',MV]';
    adj.x(1:3) = [-0.0552   -0.0125   -1.2719]'; 
end

%% Covariances
A_pol2xyz = A_polar2xyz();

if var.cov == 1
    sAng = 2.6/3600*pi/180;
    sDist = 30e-06;
    adj.Cll_eul = eye(3,3)*(0.01*pi/180)^2; % Kov ^2
    adj.Cll_R = eye(3,3)*0.3e-03^2; % Kov ^2
    adj.Cll_R_axang = eye(3,3)*(0.01*pi/180)^2; % Kov ^2
    
    [LT_roh(:,1),LT_roh(:,2),LT_roh(:,3)] = xyz2pol(data.LT(:,2),data.LT(:,3),data.LT(:,4));
    [adj.Kov,variance] = KVFpolar([LT_roh(:,1),LT_roh(:,2),LT_roh(:,3)],sAng,sDist,A_pol2xyz); % Az,El,s (in rad!!!), Std.Dev. !!
    
end