

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
