suml2loss(yhat, y) = sum((yhat .- y).^2)
sumabsloss(yhat, y) = sum(abs.(yhat .- y))
l2loss(yhat, y) = mean((yhat .- y).^2)
absloss(yhat, y) = mean(abs.(yhat .- y))
mape(yhat, y, eps = 1.0) =
    100.0 * mean(abs.(yhat .- (y .+ eps)) ./ abs.(y .+ eps))
