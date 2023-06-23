
% trainableModel.m 
% An abstract class for trainableModels
% The trainable models class is a simplified interface inspired by the model
% class in sckit learn. Any trainable model have a definition for train 
% and predict methods and contain a set of trainable prameters and a set of 
% hyperparamters as properties.
%  
% Luis Gonzalo Sanchez Giraldo
% June 2023 

classdef (Abstract) trainableModel < handle
    properties (SetAccess = public)
        trainable_params = struct([]);
        hyperparams = struct([]);
    end
    methods (Abstract)
        train(obj, train_data)
        pred = predict(obj, x)
    end
end