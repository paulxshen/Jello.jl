struct ConvBlob
    p::AbstractArray
    symmetries
    conv
end
Base.size(m::ConvBlob) = size(m.p)

function (m::ConvBlob)()
    @unpack p, symmetries, conv = m
    p = _ConvBlob(p, symmetries, conv, 0.1, true)
    p = _ConvBlob(p, symmetries, conv, 0.01, false)
end

function _ConvBlob(a::AbstractArray{T}, symmetries, conv, α=0.01, bound=false) where T
    @nograd (conv,)

    if bound
        a = sigmoid.(a - T(0.5))
    end

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    a = apply_symmetries(a, symmetries)
    stepfunc.(a, α)
end
