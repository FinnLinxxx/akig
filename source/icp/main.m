clc;
clear all;
close all;

addpath('icp')
addpath('data')

load('Leica_02_0Grad_6mm'); % gemessene Punktwolke 
load('soll_pc'); % Soll-Punktwolke


% Darstellung der beiden Punktwolken
%{
figure('units','normalized','outerposition',[0.5 0 0.5 1]);
hold on
plot3(xyz(:,1), xyz(:,2), xyz(:,3), 'b.')
plot3(soll_pc(:,1), soll_pc(:,2), soll_pc(:,3), 'r.')
legend('gem. Punktwolke','Soll-Punktwolke')
%}

% Grobe Orientierung (Voraussetzung bei ICP!) der gem. Punktwolke zur
% Soll-Punktwolke. Muss noch angepasst werden fuer dieses Beispiel!
wx = deg2rad(190); wy = deg2rad(270); wz = deg2rad(116);
R = Rz(wz)*Ry(wy)*Rx(wx);
transl = repmat([1.5;0.1;6.40], 1, size(xyz,1));
xyz_approx = (transl + R * xyz')';


% Darstellung: Grobe Orientierung der beiden Punktwolken
%{
figure('units','normalized','outerposition',[0.5 0 0.5 1]);
title('Grobe Orientierung')
hold on
plot3(xyz_approx(:,1), xyz_approx(:,2), xyz_approx(:,3), 'b.')
plot3(soll_pc(:,1), soll_pc(:,2), soll_pc(:,3), 'r.')
legend('gem. Punktwolke','Soll-Punktwolke')
%}


% ICP Algorithmus (% Funkt. verlangt die Dim. (3xn), desh. die Transponierte)
[Ricp, Ticp, ER] = icp(soll_pc', xyz_approx', 20); 
% Transformationsparameter anbringen
xyz_icp = Ricp * xyz_approx' + repmat(Ticp, 1, size(xyz_approx,1));
xyz_icp = xyz_icp';

% {
% Darstellung: Punktwolken nach ICP-Algorithmus
figure('units','normalized','outerposition',[0.5 0 0.5 1]);
title('Punktwolken nach ICP-Algorithmus')
hold on
plot3(xyz_icp(:,1), xyz_icp(:,2), xyz_icp(:,3), 'b.')
plot3(soll_pc(:,1), soll_pc(:,2), soll_pc(:,3), 'r.')
legend('gem. Punktwolke','Soll-Punktwolke')
hold off
%}















