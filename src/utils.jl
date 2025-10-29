module EWSFUtils
export mode_label

function mode_label(labels)
    counts = Dict{Any,Int}()
    for v in labels
        counts[v] = get(counts, v, 0) + 1
    end
    best = nothing
    bestc = -1
    for (k,v) in counts
        if v > bestc
            best = k; bestc = v
        end
    end
    return best
end

end # module
