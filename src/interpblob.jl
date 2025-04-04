struct InterpBlob
    p::AbstractArray
    A
    sz
    asz
    symmetries
    conv
end
Base.size(m::InterpBlob) = m.sz
@functor InterpBlob (p, A, conv,)

function (m::InterpBlob)()
    @unpack p, A, symmetries, sz, asz, conv = m
    @nograd (A, conv,)
    T = eltype(p)
    p = apply_symmetries(p, symmetries)
    p = vec(p)

    a = reshape(A * p, asz)

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a) .|> T
    a = dropdims(a, dims=(N + 1, N + 2))
    stepfunc.(a)
end
