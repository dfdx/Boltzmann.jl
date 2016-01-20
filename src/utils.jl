

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


"""Same as `get` function, but evaluates default_expr only if needed"""
macro get_or_create(dict, key, default_expr)
    return quote
        if !haskey($dict, $key)
            $dict[$key] = $default_expr
        end
        $dict[$key]
    end
end
