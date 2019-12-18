close all
clear all
clc
format longg
%% Data
load('line1.mat');
load('line2.mat');
load('line3.mat');

lt_lines = {lt_line1, lt_line2, lt_line3};
tscan_lines = {tscan_line1, tscan_line2, tscan_line3};

% TF Parameter zwischen LT und UR5 in der Form Translation [m --> mm],
% Rotationswinkel [rad] und Ma�stab []

% !!!!!!!!!!!!!!!!!!!!!!!!! d3trafo arbeitet mit geodätischer Rotation, für
% mathematische Rotationen alle Radiantwerte *-1.
% !!!!!!!!!!!!!!!!!!!!!!!!!
lt2ur = [
    6105.1
    4492.2
    -614.4
    (-1)*-0.0036
    (-1)*0.0053
    (-1)*(3.7517-pi) % Achsendefinition beachten, flinzer & mmikschi unterschiedlich?!! daher bei flinzer -pi
    1]';


% TF parameter zwischen UR5 Base und Flanschframe in from von Translation
% [m] und Rotationsmatrix (numpy flatten)
ur_base2flansch = [0.14190628 -0.33053733  0.26661017 -0.42212532  0.5597224  -0.71310662  0.85309395  0.51136504 -0.10361716  0.30666094 -0.65208637 -0.69335592
                   0.1790546  -0.29097045  0.19822112 -0.35581447  0.45806736 -0.81459828  0.93432821  0.19362739 -0.29923106  0.02066056 -0.86757289 -0.49688069
                  -0.24913413 -0.13770821  0.28824816 -0.89657148 -0.44096571  0.04133796  -0.35248338  0.65391288 -0.66944262  0.26816981 -0.6147741  -0.74171272
                  -0.23990856 -0.14079256  0.27978861 -0.98700246  0.10405879  0.12246595  -0.01239288  0.71050001 -0.70358806 -0.16022658 -0.69596085 -0.69997567];

% Translation zwischen UR5 Base und Flansch von m in mm
ur_base2flansch(:, 1:3) = ur_base2flansch(:, 1:3)*1e3;

%% Plots
figure(1);
hold on
grid on
title('LT')
xlabel('x')
ylabel('y')
zlabel('z')
figure(2);
hold on
grid on
title('Baseframe')
xlabel('x')
ylabel('y')
zlabel('z')
figure(3);
hold on
grid on
title('Flanschframe')
xlabel('x')
ylabel('y')
zlabel('z')
figure(4);
hold on
grid on
title('Flanschframe - reduced Center')
xlabel('x')
ylabel('y')
zlabel('z')


%% Trafo
lt_line_in_ur_base = cell(numel(lt_lines), 1);
lt_line_in_ur_flansch = cell(numel(lt_lines), 1);

for line_idx = 1:numel(lt_lines) %iter through all data epochs
%line_idx = 2
% TF vom LT ins Baseframe
% �ber die geodetic toolbox:
    lt_line_in_ur_base{line_idx} = d3trafo(lt_lines{line_idx},lt2ur, [0 0 0], 1);
    
    current_translation = ur_base2flansch(line_idx,1:3);
    current_rotmat = ur_base2flansch(line_idx,4:end);
    current_rotmat = reshape(current_rotmat, 3, 3)';
    
    current_roteul = rotm2eul(current_rotmat);
    
    ur2flansch = [
    current_translation(1)
    current_translation(2)
    current_translation(3)
    (-1)*current_roteul(3) % default Achsendefition für rotm2eul (weiter oben) ist "ZYX", daher hier in umgekehrter Reihenfolge eingeben.
    (-1)*current_roteul(2)
    (-1)*current_roteul(1) 
    1]';

    lt_line_in_ur_flansch{line_idx} = d3trafo(lt_line_in_ur_base{line_idx},ur2flansch, [0 0 0], 1); %d3trafo berücksichtigt die konforme Rotation?!
    %lt_line_in_ur_flansch{line_idx} = current_translation' + current_rotmat * (lt_line_in_ur_base{line_idx}-mean(lt_line_in_ur_base{line_idx}))' + mean(lt_line_in_ur_base{line_idx})';
    %lt_line_in_ur_flansch{line_idx} = lt_line_in_ur_flansch{line_idx}';

% Plotten der Scannlinien in den verschiedenen Frames
    figure(1)
    scatter3(lt_lines{line_idx}(:,1), lt_lines{line_idx}(:,2), lt_lines{line_idx}(:,3))
    figure(2)
    scatter3(lt_line_in_ur_base{line_idx}(:,1), lt_line_in_ur_base{line_idx}(:,2), lt_line_in_ur_base{line_idx}(:,3))
    figure(3)
    scatter3(lt_line_in_ur_flansch{line_idx}(:,1), lt_line_in_ur_flansch{line_idx}(:,2), lt_line_in_ur_flansch{line_idx}(:,3))
    
    
% Plotten der Scanline, aber Nullpunktreduziert.    
    xyz_tscan_lineX = [ lt_line_in_ur_flansch{line_idx}(floor(length(lt_line_in_ur_flansch{line_idx})/2),1) ...
                        lt_line_in_ur_flansch{line_idx}(floor(length(lt_line_in_ur_flansch{line_idx})/2),2) ...
                        lt_line_in_ur_flansch{line_idx}(floor(length(lt_line_in_ur_flansch{line_idx})/2),3) ]
    figure(4)
    scatter3(lt_line_in_ur_flansch{line_idx}(:,1)-xyz_tscan_lineX(:,1), lt_line_in_ur_flansch{line_idx}(:,2)-xyz_tscan_lineX(:,2), lt_line_in_ur_flansch{line_idx}(:,3)-xyz_tscan_lineX(:,3))
end