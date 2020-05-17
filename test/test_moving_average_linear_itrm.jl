using InterpretableModels: getfactor, parambounds, getintercept, featurename
using InterpretableModels: setfeaturenames!, modelasstring, setparams!, hasparams
using InterpretableModels: minlag, actualminlag, maxlag, actualmaxlag, params, horizon
using InterpretableModels: predict, predict!, fit!, isvalidparams, windowspans
using DataFrames

@testset "MovingAverageLITRM" begin

m = MovingAverageLITRM(2, 4;
        minlag = 3, maxlag = 8,
        hasintercept = true, mincoef = -10.0, maxcoef = 97.5)

@test !isclassificationmodel(m)
@test  isregressionmodel(m)
@test  istimeseriesmodel(m)
@test !hasparams(m)
@test minlag(m) == 3
@test actualminlag(m) == 3
@test horizon(m) == 3
@test maxlag(m) == 8
@test actualmaxlag(m) == 8

pserr1 = [1.0, 2.24, 5.5, 2.1, 4.1, 1.2, 8.0, 0.2, 0.6]
@test !isvalidparams(m, pserr1) # Has a window length < 1.0 (0.2)

pserr2 = [1.0, 2.24, 5.5, 0.1, 4.1, 1.2, 8.0, 1.2, 0.6]
@test !isvalidparams(m, pserr2)

ps = [1.0, 2.24, 5.5, 2.1, 4.1, 1.2, 8.0, 3.2, 0.6]
@test windowspans(m, ps) == [(5, 2, 6), (8, 1, 8)]
@test isvalidparams(m, ps)
setparams!(m, ps)
@test hasparams(m)
@test minlag(m) == 3
@test actualminlag(m) == 5 # Since 5.5 < 8.0
@test actualmaxlag(m) == 8 # Since 8.0 > 5.5
@test horizon(m) == 5
@test params(m) == ps

c, f, ws, wlen, we = getfactor(m, 1)
@test c == 1.0
@test isa(f, Int)
@test f == 2
@test isa(ws, Int)
@test ws == 5
@test isa(wlen, Int)
@test wlen == 2
@test isa(we, Int)
@test we == 6

c, f, ws, wlen, we = getfactor(m, 2)
@test c == 4.1
@test isa(f, Int)
@test f == 1
@test isa(ws, Int)
@test ws == 8
@test isa(wlen, Int)
@test wlen == 1
@test isa(we, Int)
@test we == 8

@test getintercept(m) == 0.6

@testset "parambounds" begin
    bs = parambounds(m)
    @test length(bs) == (4*2+1) # 4 values per factor and 1 for intercept
    for fi in 1:2
        o = (fi-1)*4
        @test bs[1+o] == (-10.0, 97.5)
        @test bs[2+o][1] == 1.0  # feature 1 is lowest possible feature number
        @test isapprox(bs[2+o][2], 1.0+4.0, atol=1e-4)  # Should almost be 1+nfeatures
        @test bs[3+o][1] == 3.0                         # minlag
        @test isapprox(bs[3+o][2], 1.0+8.0, atol=1e-4)  # maxlag
        @test bs[4+o][1] == 1.0                         # minlag
        @test isapprox(bs[4+o][2], 2.0+8.0-3.0, atol=1e-4)
    end
    @test bs[end] == (-10.0, 97.5)
end # @testset "parambounds" begin

@testset "featurenames and setfeaturenames!" begin
    @test featurename(m, 1) == "feature_1"
    @test featurename(m, 2) == "feature_2"
    setfeaturenames!(m, ["arn", "be"])
    @test featurename(m, 1) == "arn"
    @test featurename(m, 2) == "be"
end # @testset "featurenames and setfeaturenames!" begin

@testset "modelasstring" begin
    setparams!(m, [1.2, 1.6, 5.8, 2.1, 6.987, 2.3, 7.3, 1.4, 0.762])
    @test modelasstring(m) == "0.76 + 1.2 * MAvg(arn[T-5], 2) + 6.99 * be[T-7]"
    df = DataFrame(MyFactor1 = Int[], AnotherFac = Float64[])
    @test modelasstring(m, df) == "0.76 + 1.2 * MAvg(MyFactor1[T-5], 2) + 6.99 * AnotherFac[T-7]"
end

@testset "predict, predict! and fit!" begin
    m = MovingAverageLITRM(2, 3;
            minlag = 1, maxlag = 3,
            hasintercept = true, mincoef = -10.0, maxcoef = 97.5)

    df = DataFrame(
        FacA = Int[    1,    2,   3,    4,    5],
        Fac2 = Float64[3.14, 5.4, 9.8, 10.0, 25.0],
        Fac3 = Float64[4.00, 6.0, 8.0, 10.0, 12.0])

    X = Array{Float64}(df)

    trueparams = [-2.5, 1.0, 1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 1.34]
    setparams!(m, trueparams)
    # Since we have a lag range from 1-2 we can only get three predictions
    # from this model, i.e. for rows 3, 4 and 5.
    # True expression: y = 1.34 .+ MAVG(X[T-1, 1], 2) .* (-2.5) .+ X[T-2, 3] .* 4.0
    y = [Inf, Inf,
            1.34 + (-2.5)*((X[3-1, 1] + X[3-2, 1])/2) + 4.0*X[3-2, 3],
            1.34 + (-2.5)*((X[4-1, 1] + X[4-2, 1])/2) + 4.0*X[4-2, 3],
            1.34 + (-2.5)*((X[5-1, 1] + X[5-2, 1])/2) + 4.0*X[5-2, 3]]
    yhat = predict(m, X, trueparams)
    @test length(yhat) == 3
    @test yhat[1] == y[3]
    @test yhat[2] == y[4]
    @test yhat[3] == y[5]

    yhatpreds = zeros(Float64, 3)
    predict!(m, yhatpreds, X, trueparams)
    @test yhatpreds[1] == y[3]
    @test yhatpreds[2] == y[4]
    @test yhatpreds[3] == y[5]

    # If we don't have the true params we can fit the model to data to recover them.
    dummyparams = [3.5, 1.5, 1.5, 2.1, 2.1, 2.0, 1.7, 1.6, 1.34]
    setparams!(m, dummyparams) # Set a random param vector to ensure a better one is set after fitting
    fit!(m, X, y; MaxTime = 5.0)
    @test length(params(m)) == (1+2*4)
    @test params(m) != dummyparams

    # Since the problem is underconstrained we cannot expect to find the "true"
    # params since there are many that works...
    yhat = predict(m, X)
    @test isapprox(yhat[end-1], y[4], atol=1e-1)
    @test isapprox(yhat[end], y[5], atol=1e-1)
end

end # @testset "MovingAverageLITRM" begin
