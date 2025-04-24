struct ConvBlob
    p::AbstractArray
    symmetries
    conv1
    conv2
end
Base.size(m::ConvBlob) = size(m.p)
function (m::ConvBlob)()
    @unpack p, symmetries, conv1, conv2 = m
    # p = _ConvBlob(p, symmetries, conv1, 0.1)
    p = _ConvBlob(p, symmetries, conv1)
    # @debug p |> extrema
    # foo(p)
end

function _ConvBlob(a::AbstractArray{T}, symmetries, conv) where {T}
    @nograd (conv,)

    N = ndims(a)
    R = (size(conv.weight)[1] - 1) ÷ 2
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    a = apply_symmetries(a, symmetries)
    # stepfunc.(a, min(10, 1.5R))
    # stepfunc.(a, 2R)
    # g = pad.(cdiff.([a], 1:N) / 2, :replicate, [1:N .== i for i = 1:N])
    # g = broadcast(1:2) do dims
    #     n =  1:N .== dims
    #     pad(cdiff(a; dims) / 2, :replicate, n)
    # end
    g = [
        pad(cdiff(a; dims=1) / 2, :replicate, 1:N .== 1),
        pad(cdiff(a; dims=2) / 2, :replicate, 1:N .== 2),
    ]
    lb = [a] .- g
    ub = [a] .+ g
    lb, ub = min.(lb, ub), max.(lb, ub)
    v = map(lb, ub) do lb, ub
        map(lb, ub) do lb, ub
            _lb = lb .> 0.5
            _ub = ub .> 0.5
            b = _lb .== _ub
            (ub - T(0.5)) ./ (ub - lb) .* (1 - b) + b .* _lb
        end
    end
    map(v...) do v...
        for (i, x) = enumerate(v)
            ((i == N) || 0 < x < 1) && return x
        end
    end
    # _a .* .!edges + a .* edges
end
