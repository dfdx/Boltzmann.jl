

if !isdefined(:__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{UInt64}()
end

macro runonce(expr)
    h = hash(expr)
    return esc(quote
        if !in($h, __EXPRESSION_HASHES__)
            push!(__EXPRESSION_HASHES__, $h)
            $expr
        end
    end)
end


"""Same as `get` function, but evaluates default_expr only if needed"""
macro get(dict, key, default_expr)
    return quote
        if haskey($dict, $key)
            $dict[$key]
        else
            $default_expr
        end
    end
end


"""
Same as `@get`, but creates new object from `default_expr` if
it didn't exist before
"""
macro get_or_create(dict, key, default_expr)
    return quote
        if !haskey($dict, $key)
            $dict[$key] = $default_expr
        end
        $dict[$key]
    end
end



"""
Same as `@get`, but immediately exits function and return `default_expr`
if key doesn't exist.
"""
macro get_or_return(dict, key, default_expr)
    return quote
        if haskey($dict, $key)
            $dict[$key]
        else
            return $default_expr
            nothing  # not reachable, but without it code won't compile
        end
    end
end

"""
Get array of size `sz` from a `dict` by `key`. If element doesn't exist or
its size is not equal to `sz`, create and return new array
using `default_expr`. If element exists, but is not an error,
throw ArgumentError.
"""
macro get_array(dict, key, sz, default_expr)
    return quote
        if (haskey($dict, $key) && !isa($dict[$key], Array))
            local k = $key
            throw(ArgumentError("Key `$k` exists, but is not an array"))
        end
        if (!haskey($dict, $key) || size($dict[$key]) != $sz)
            # ensure $default_expr results in an ordinary array
            $dict[$key] = convert(Array, $default_expr)
        end
        $dict[$key]
    end
end


function logistic(x)
    return 1 ./ (1 + exp(-x))
end


const KNOWN_OPTIONS =
    [:gradient, :update, :sampler, :scorer, :reporter,
     :batch_size, :n_epochs, :n_gibbs,
     :lr, :momentum, :weight_decay_kind, :weight_decay_rate,
     :sparsity_cost, :sparsity_target,
     :randomize,
     # deprecated options
     :n_iter]
const DEPRECATED_OPTIONS = Dict(:n_iter => :n_epochs)

function check_options(opts::Dict)
    deprecated_keys = keys(DEPRECATED_OPTIONS)
    for opt in keys(opts)
        if !in(opt, KNOWN_OPTIONS)
            warn("Option '$opt' is unknownm ignoring")
        end
        if in(opt, deprecated_keys)
            warn("Option '$opt' is deprecated, " *
                 "use '$(DEPRECATED_OPTIONS[opt])' instead")
        end
    end
end


function split_evenly(n, len)
    n_parts = Int(ceil(n / len))
    parts = Array(Tuple, n_parts)
    for i=1:n_parts
        start_idx = (i-1)*len + 1
        end_idx = min(i*len, n)
        parts[i] = (start_idx, end_idx)
    end
    return parts
end

"""
`tofinite!` takes an array and
1. turns all NaNs to zeros
2. turns all Infs and -Infs to the largets and
   smallest representable values accordingly.
3. turns all zeros to the smallest representable
   non-zero values, if `nozeros` is true
"""
function tofinite!(x::Array; nozeros=false)
    for i in eachindex(x)
        if isnan(x[i])
            x[i] = 0.0
        elseif isinf(x[i])
            if x[i] > 0.0
                x[i] = prevfloat(x[i])
            else
                x[i] = nextfloat(x[i])
            end
        end

        if x[i] == 0.0 && nozeros
            x[i] = nextfloat(x[i])
        end
    end
end


function ensure_type(newT::DataType, A::AbstractArray)
    if eltype(A) != newT
        map(newT, A)
    else
        A
    end
end

function add!{T}(X::Array{T}, inc::T)
    @simd for i=1:length(X)
        @inbounds X[i] += inc
    end
end
