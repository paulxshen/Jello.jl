struct InterpBlob
    p::AbstractArray
    A
    sz
    asz
    sesolid
    sevoid
    frame
    margin::Int
    symmetries
    conv
end
Base.size(m::InterpBlob) = m.sz
@functor InterpBlob (p, A, conv,)

function (m::InterpBlob)(sharp=true;)
    @unpack p, A, symmetries, sz, asz, frame, margin, sevoid, sesolid, conv = m
    @nogradvars (A, frame, conv,)

    a = reshape(A * p, asz)
    a = apply_symmetries(a, symmetries, sz)

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))
    a = stepfunc.(a)
    # if sharp
    #     a = stepfunc(a)
    # end
    a = imframe(a, frame, margin)
    # # if sharp
    # #     a = smooth(a, sesolid, sevoid)
    # # end
    # a
end
