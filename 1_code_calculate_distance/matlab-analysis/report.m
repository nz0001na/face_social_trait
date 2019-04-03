clear;clc;
close all;

%addpath('./redist-source');
% autism_data = csvread(['left_autism.csv']); 

fileID = fopen('v4_autism.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec)

fileID2 = fopen('v4_normal.txt','r');
B = fscanf(fileID2,formatSpec)

% VA = cell2mat(A)
% VB = cell2mat(B)

A1 = A'
B1 = B'


reportTstats(A,B)


fclose(fileID)
fclose(fileID2)