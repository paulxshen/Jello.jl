struct ConvBlob
    p::AbstractArray
    symmetries
    conv
end
Base.size(m::ConvBlob) = size(m.p)

function (m::ConvBlob)()
    @unpack p, symmetries, conv = m
    @nograd (conv,)

    T = eltype(p)
    p = sigmoid.(p - T(0.5))
    a = apply_symmetries(p, symmetries)

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a) .|> T
    a = dropdims(a, dims=(N + 1, N + 2))
    stepfunc.(a)
end
