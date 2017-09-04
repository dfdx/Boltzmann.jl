
abstract type AbstractReporter end

struct TextReporter
end

function report(r::TextReporter, rbm::AbstractRBM,
                epoch::Int, epoch_time::Float64, score::Float64)
    println("[Epoch $epoch] Score: $score [$(epoch_time)s]")
end

function report(r::TextReporter, dbn::DBN, epoch::Int, layer::Int)
    println("[Layer $layer] Starting epoch $epoch")
end

