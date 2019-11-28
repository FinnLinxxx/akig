function [A] = A_polar2xyz()

syms s Az El
% t_T_RF =  sym('t_T_RF',[3 1]);

% Polarpunkt-Bestimmung = Translation
t_LT_T = [ s * cos(Az) * sin(El);...
    s * sin(Az) * sin(El);...
    s * cos(El)];

A = [diff(t_LT_T,Az) diff(t_LT_T,El) diff(t_LT_T,s)];
A = matlabFunction(A);

end