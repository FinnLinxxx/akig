function [mat] = rotZ(a)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    mat = [cos(a) sin(a) 0
           -sin(a) cos(a) 0
           0 0 1];
end

