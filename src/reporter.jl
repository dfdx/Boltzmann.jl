
abstract AbstractReporter

type TextReporter
end

function report(r::TextReporter, rbm::AbstractRBM,
                epoch::Int, epoch_time::Float64, score::Float64)
    println("[Epoch $epoch] Score: $score [$(epoch_time)s]")
end


