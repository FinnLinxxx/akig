% data_trafo_TTH
% measurement from 2019-03-26
% computation in apply_trafo_leverarm.m

% necessary problem specific inputs are
%    - the robot and sensor data
%    - the approximate values for the parameters
%    - the stochastic model


%% Data
% Units in m and rad!!!

path = '.\data_add-leverarm-to-LT\';

load('191102_TrafoPts_20-23.mat')
rob = TrafoPts_PoseJuri;
for i = 1:size(rob,1)
    r = rob{i,12:20};
    Rot = [r(1) r(2) r(3);...
        r(4) r(5) r(6);...
        r(7) r(8) r(9)];
    angR(i,:) = ltRot(Rot);
end
data.R = [rob{:,1},rob{:,7:9},angR];


datLT = TrafoPts_PoseLT;
data.LT = [datLT(:,1),datLT(:,2:4)*1e-03,datLT(:,5:7)*pi/180];


% Schwerpunkts-Berechnung - 27.06.2019
meanLT = mean(data.LT(:,2:4));
data.LT(:,2:4) = data.LT(:,2:4) - repmat(meanLT,size(data.LT,1),1);
clear rob datLT


%% Approximate Values


if var.leverarmRF
    MV = [0 +0.02 0.081+0.032];
else
    MV = [0 -0.02 -0.08-0.032];
end

if ~var.leverarmRF
    for i = 1:length(data.LT)
        rot = ltRot(data.LT(i,5:7));
        pt(i,:) = [rot * MV' + data.LT(i,2:4)']';
    end
    [tp,rc,ac,tr] = helmert3d(pt(:,:),data.R(:,2:4),'7p',1,[0,0,0]);
else
    for i = 1:length(data.R)
        rot = ltRot(data.R(i,5:7));
        pt(i,:) = [rot * MV' + data.R(i,2:4)']';
    end
    [tp,rc,ac,tr] = helmert3d(pt(:,:),data.LT(:,2:4),'7p',1,[0,0,0]);
end
par = [tp(4:6);tp(1:3)] % in rad
adj.x = [par',MV]';


%% Covariances
A_pol2xyz = A_polar2xyz();

if var.cov == 1
    sAng = 2.6/3600*pi/180; % statisch; 12.5e-06 bis 2.5 m
    sDist = 30e-06;%sqrt(12.5e-06^2+6e-06^2+15e-06^2);
    adj.Cll_eul = eye(3,3)*(0.01*pi/180)^2; % Kov ^2
    adj.Cll_R = eye(3,3)*0.13e-03^2; % Kov ^2
    adj.Cll_R_axang = eye(3,3)*(0.01*pi/180)^2; % Kov ^2
    
    [LT_roh(:,1),LT_roh(:,2),LT_roh(:,3)] = xyz2pol(data.LT(:,2),data.LT(:,3),data.LT(:,4));
    [adj.Kov,variance] = KVFpolar([LT_roh(:,1),LT_roh(:,2),LT_roh(:,3)],sAng,sDist,A_pol2xyz); % Az,El,s (in rad!!!), Std.Dev. !!
end
