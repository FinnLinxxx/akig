function [x,y,z] = pol2xyz(az,elev,r)
%pol2xyz Transform 3d polar to cartesian coordinates.
%   Az is clockwise angle in the xy plane measured from the
%   positive x axis.  Elev is the zenithAngle from the zx plane.
%
%   SHO

z = r .* cos(elev);
rsinelev = r .* sin(elev);
x = rsinelev .* cos(az);
y = rsinelev .* sin(az);