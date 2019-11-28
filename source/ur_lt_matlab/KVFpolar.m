function [sf, var_f] = KVFpolar(beob,sig_Win,sig_Dist,A)
% [sf] = KVFpolar(beob,sig_Win,sig_Dist)
% Computation of LT point precision due to variance propagation
% Input:
%    beob    ... vector [Az,El,s] (Az, El) in rad
%    sig_Win ... std.dev. in rad
%    sig_Dist... std.dev. in mm
% Output:
%    sf      ... std.dev. of the point
% 12.04.2017 - SHO


sig = [sig_Win; sig_Win; sig_Dist]; %rad and mm
S = diag(sig.^2);

j = 1;
for i = 1:size(beob,1)
    % A generated in A_polar2xyz.m
    AA = A(beob(i,1),beob(i,2),beob(i,3));
        
    sf(j:j+2,:) = AA*S*AA';
    var_f(i,:) = diag(sf(j:j+2,:));
    j = 3*(i-1)+4;
end

end