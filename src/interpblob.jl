struct InterpBlob
    p::AbstractArray
    A
    sz
    asz
    lmin
    lvoid
    lsolid
    frame
    margin
    symmetries
    conv
end
@functor InterpBlob (p,)

function (m::InterpBlob)(sharpness::Real=0.995; withloss=false)
    @unpack p, A, symmetries, sz, asz, lmin, frame, margin, lvoid, lsolid, conv = m
    @ignore_derivatives_vars (A, frame, conv)

    T = eltype(p)
    α = T(1 - sharpness)

    p = abs.(p)
    # Z = maximum(p)
    # if Z > 0
    #     p = p / Z
    # end

    a = reshape(A * p, asz)
    a = apply_symmetries(a, symmetries, sz)

    N = ndims(a)
    Rf = (size(conv.weight, 1) - 1) ÷ 2
    a = pad(a, :replicate, Rf)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    a = step(a, α)
    a = imframe(a, frame, margin)

    if withloss
        p = 1
        l = loss(a, p * lsolid, p * lvoid)
        a, l
    else
        a
    end
end
