X = rand(100, 20)

# True generating function has intercept, 2 factors and a lag range of 3-5.
genfn(r, X) = 1.56 + 2.4*X[r-5, 3] - 5.6*X[r-3, 13]

y = Float64[genfn(r, X) for r in 6:size(X, 1)]

# Now in reality we don't know the real model se we will have to train it
# from data. Let's do it and try to recover an interpretable model.
using InterpretableModels

# We don't know how many factors so let's aim for 3 and a lag range of 2-10
lm = LaggedFactorLITRM(3, size(X, 2); minlag=2, maxlag=10)

fit!(lm, X, y; MaxTime = 10.0)

println(lm) # The model should have been recovered quite well...
