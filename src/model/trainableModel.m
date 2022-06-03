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