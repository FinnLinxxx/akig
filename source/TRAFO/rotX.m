function [mat] = rotX(a)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    mat = [1 0 0
           0 cos(a) sin(a)
           0 -sin(a) cos(a)];
end

