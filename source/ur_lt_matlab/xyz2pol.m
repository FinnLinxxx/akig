function [az,elev,r] = xyz2pol(x,y,z)
%xyz2pol Transform Cartesian to 3d polar coordinates.
%[az,elev,r] = xyz2pol(x,y,z)
%   Az is clockwise angle in the xy plane measured from the
%   positive x axis.  Elev is the zenithAngle from the zx plane.


hypotxy = hypot(x,y);
r = hypot(hypotxy,z);
% elev = atan2(z,hypotxy);
elev = acos(z./r);
az = atan2(y,x);