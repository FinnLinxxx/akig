% Script applying the adjustment of the 'trafo_leverarm' approach
%
% Integrated determination of the transformation of a sensor frame with 
% respect to the robot arm frame as well as the leverarm between robot
% flange and the measurement point; Estimated in a GH-model
%
% It needs to be specified if a full observation covariance
% matrix shall be used (cov = 1) or an identity matrix (cov = 0);
% Furthermore, it need to be defined, if the leverarm shall be added to the
% robot flange (leverarmRF = 1) or to a probe (=6DOF-Measurement Device) 
% (leverarmRF = 0).
%
% Reference: see Horvath, Neuner, system identification of a robot arm with
% EKF and ANN; JAG 2/2019
%
% It bases on simulation_UR_2.m

% Sabine Horvath, TU Wien, 22.05.2019

clear all

addpath('GeodeticToolbox')
path_out = 'Results';


%% Processing options

% Leverarm added to the Robot-Flange or to the Probe
var.leverarmRF = 1; % 1 ... RobotFlange; 0 ... Probe

% Covariances of the Measurements
var.cov = 1; % 1 ... including; 0 ... identity matrix

%% Data, Approximate Values and Stochastic Model

% Example var.leverarmRF = 1;
addpath('data_add-leverarm-to-RF')
data_trafo_TTH2   
name = 'trafo_TTH';

% % Example var.leverarmRF = 0;
% addpath('.\data_add-leverarm-to-LT')
% data_trafoANN_190915
% name = 'trafoANN';


%% Functional model (matlabFunction) and Dimensions

[adj,data] = fct_trafo_leverarm(var,data,adj);


%% GH Adjustment

alpha = 0.05;
adj.s02_apri = 1;
adj.ww = 1;
crit = 0; it = 1;
while crit == 0
    
    % A and B
    adj = fill_trafo_leverarm(adj,var);

    
    % Compute x and v
    adj = gauss_helmert(adj);
        
    
    va(:,it) = adj.va;
    x(:,it) = adj.x;
    w(:,it) = adj.w;
    
    it = it+1;
    
    % Breaking criteria
    if abs(adj.ww_it - adj.ww) < 1e-09
        %         if abs(x1-x) < 1e-09 % 1e-09
        break
    end

end


%% Outlier detection

% Tuning of test statistics - multiple test
alpha0 = 1-(1-alpha)^(1/adj.beob); % see Heunecke p. 248; Pelzer p. 144
% alpha0 = alpha/adj.beob;

% Standardised residuals
[od_val,od_ind] = max(abs(adj.va)./adj.stdv);
% ind = find(abs(adj.va)>tinv(1-alpha0/2,adj.frei)*adj.s02_apost*adj.stdv);

down_it = 1;
while od_val > tinv(1-alpha0/2,adj.frei)
    
    outl(down_it) = od_ind;
    
    % Downweighting
    adj.Qll(od_ind,od_ind) = adj.Qll(od_ind,od_ind)*10;
    
    
    % Compute x and v
    adj = gauss_helmert(adj);
    
    % Criterion
    [od_val,od_ind] = max(abs(adj.va)./adj.stdv);
    
    down_it = down_it+1;
end

if exist('outl','var')
    outl_ind = unique(outl);
    outl_samples = outl_ind/adj.nn
end


%% Output

if exist('meanLT')
    if var.leverarmRF
        adj.x(4:6) = adj.x(4:6) + meanLT';
    else
        meanR = ltRot(adj.x(1:3))*meanLT';
        adj.x(4:6) = adj.x(4:6) - meanR;
    end
end
adj.x
it
GT_True = adj.gt
GT_Apost = adj.s02_apost

% Correlations
for i = 1:size(adj.Cxx,1)
    for j = 1:size(adj.Cxx,2)
        corr(i,j) = adj.Cxx(i,j)/sqrt(adj.Cxx(i,i))/sqrt(adj.Cxx(j,j));
    end
end
corr_max = max(abs(corr-eye(size(corr))));
figure
imagesc(corr)
colorbar; caxis([-1 1]);
title('Correlations Qxx')


%% Output result

save([path_out sprintf('CalibData_AdjResult_%s',name)],'adj')

t_LT_R = [adj.x(4:6)';adj.stdX(4:6)'];
ang_LT_R = [adj.x(1:3)';adj.stdX(1:3)'];
Rotmat = ltRot(adj.x(1:3));
R_LT_R = [Rotmat(:)';nan(1,9)];
t_RF_CCR =[ adj.x(7:9)';adj.stdX(7:9)'];
output = table(t_RF_CCR,t_LT_R,ang_LT_R,R_LT_R);
output.Properties.RowNames = {'x','stdX'};
writetable(output,[path_out sprintf('CalibData_AdjResult_%s',name)],'WriteRowNames',1)
