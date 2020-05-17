using Statistics

X = rand(50, 20)

# True generating function has intercept, 2 factors and a lag range of 3-5.
genfn(r, X) = 2.56 - 7.8*mean(X[(r-5):(r-4), 2]) + 9.3*X[r-3, 11] + (0.01 * rand())

y = Float64[genfn(r, X) for r in 6:size(X, 1)]

# Now in reality we don't know the real model se we will have to train it
# from data. Let's do it and try to recover an interpretable model.
using InterpretableModels

# We don't know how many factors so let's aim for 3 and a lag range of 2-10
m = MovingAverageLITRM(3, size(X, 2); minlag=2, maxlag=10)

fit!(m, X, y; MaxTime = 10.0)

println(m) # The model should have been recovered quite well...
