struct ConvBlob
    p::AbstractArray
    symmetries
    conv1
    conv2
end
Base.size(m::ConvBlob) = size(m.p)

function (m::ConvBlob)()
    @unpack p, symmetries, conv1, conv2 = m
    p = _ConvBlob(p, symmetries, conv1, 0.1)
    p = _ConvBlob(p, symmetries, conv2)
    # @debug p |> extrema
    p
end

function _ConvBlob(a::AbstractArray{T}, symmetries, conv, α=0.001) where T
    @nograd (conv,)

    # if bound
    # a = sigmoid.(a - T(0.5))
    # end

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    a = apply_symmetries(a, symmetries)
    stepfunc.(a, α)
end
