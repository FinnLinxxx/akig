function [rot] = ltRot(X)
% LT eulerXYZ
% in SA User Manual p970
% Unit Rotation Matrices differently chosen as RxA, etc
% if transition to angles output = [om fi ka]

[rows,cols]=size(X);

if rows==3 && cols==3  %extract angles in RAD
    om = atan2(-X(2,3),X(3,3));
    fi = asin(X(1,3));
    ka = atan2(-X(1,2),X(1,1));
    
    rot=[om fi ka];

    
elseif length(X)==3  %construct rotation matrix, using angles in RAD
    c1 = cos(X(1)); % Rx(X)
    s1 = sin(X(1));
    c2 = cos(X(2)); % Ry(Y)
    s2 = sin(X(2));
    c3 = cos(X(3)); % Rz(Z)
    s3 = sin(X(3));
    
    RxA = [1 0 0; 0 c1 s1; 0 -s1 c1];
    RyB = [c2 0 -s2; 0 1 0; s2 0 c2];
    RzC = [c3 s3 0; -s3 c3 0; 0 0 1];
    
    rot = RzC*RyB*RxA;
    rot = rot';
end

