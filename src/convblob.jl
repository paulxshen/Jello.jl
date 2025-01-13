struct ConvBlob
    p::AbstractArray
    sz
    sesolid
    sevoid
    frame
    margin::Int
    symmetries
    conv
end
Base.size(m::ConvBlob) = m.sz
@functor ConvBlob (p, conv)

function (m::ConvBlob)(sharp=true;)
    @unpack p, symmetries, sz, frame, margin, sevoid, sesolid, conv = m
    @nograd (frame, conv,)
    T = eltype(p)
    α = T(1 - sharp) / 2

    a = p
    a = apply_symmetries(a, symmetries, sz)

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    if sharp
        a = stepfunc(a)
    end
    a = imframe(a, frame, margin)
    if sharp > 0
        a = smooth(a, sesolid, sevoid)
    end
    a
end
