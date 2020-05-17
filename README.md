# InterpretableModels

[![Build Status](https://travis-ci.com/robertfeldt/InterpretableModels.jl.svg?branch=master)](https://travis-ci.com/robertfeldt/InterpretableModels.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/robertfeldt/InterpretableModels.jl?svg=true)](https://ci.appveyor.com/project/robertfeldt/InterpretableModels-jl)
[![Codecov](https://codecov.io/gh/robertfeldt/InterpretableModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/robertfeldt/InterpretableModels.jl)
[![Coveralls](https://coveralls.io/repos/github/robertfeldt/InterpretableModels.jl/badge.svg?branch=master)](https://coveralls.io/github/robertfeldt/InterpretableModels.jl?branch=master)

Interpretable and explainable machine learning (ML) and AI in Julia. Much current work in ML and AI focus mainly on attaining the highest accuracy. This often leads to complex, opaque models that are hard to interpret and understand. In contrast, this Julia package helps you train small and simple models that are easy to interpret. This can have many benefits in "sensitive" application domains such as e.g. law and medicine.

Currently this package implements the modeling approaches:
- SLIM (Supersparse Linear Integer Model) Scoring systems for classification, and
- a LaggedFactor model for time series regression, and
- a MovingAverage model for time series regression.

All models are currently optimized via heuristic, black-box optimization.

## Usage

For SLIM, see [examples/bankruptcy_dataset.jl](examples/bankruptcy_dataset.jl) for a simple example of its use.

For LaggedFactor LITRM, see [examples/lagged_factor_timeseries_model.jl](examples/lagged_factor_timeseries_model.jl) for a simple example.

For MovingAverage LITRM, see [examples/moving_average_timeseries_model.jl](examples/moving_average_timeseries_model.jl) for a simple example.
