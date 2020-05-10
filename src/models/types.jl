abstract type InterpretableModel end

notdef(T::DataType, method) =
    error("Method $(string(method)) not defined for type $(T)")
notdef(o, method) = notdef(typeof(o), method)
istimeseriesmodel(m::InterpretableModel) = false
isclassificationmodel(m::InterpretableModel) = notdef(m, :isclassificationmodel)
isregressionmodel(m::InterpretableModel) = notdef(m, :isregressionmodel)

abstract type InterpretableRegressionModel <: InterpretableModel end
isclassificationmodel(m::InterpretableRegressionModel) = false
isregressionmodel(m::InterpretableRegressionModel) = true

abstract type InterpretableClassificationModel <: InterpretableModel end
isclassificationmodel(m::InterpretableClassificationModel) = true
isregressionmodel(m::InterpretableClassificationModel) = false

abstract type InterpretableTimeseriesModel <: InterpretableModel end
istimeseriesmodel(m::InterpretableTimeseriesModel) = true

"""
    horizon(model)

The number of steps into the future that a time series model can predict.
Returns Inf if it can predict a flexible number of steps into the future.
"""
horizon(m::InterpretableTimeseriesModel) = notdef(m, :horizon)

abstract type InterpretableTimeseriesRegressionModel <: InterpretableTimeseriesModel end
isclassificationmodel(m::InterpretableTimeseriesRegressionModel) = false
isregressionmodel(m::InterpretableTimeseriesRegressionModel) = true
