const Δ = 2

struct InterpBlob
    p::AbstractArray
    A
    sz
    asz
    sesolid
    sevoid
    jump
    frame
    margin::Int
    symmetries
    conv
end
Base.size(m::InterpBlob) = m.sz
@functor InterpBlob (p, A, conv,)

function (m::InterpBlob)(sharpness::Real=0.998;)
    @unpack p, A, symmetries, sz, asz, frame, margin, sevoid, sesolid, conv = m
    @nogradvars (A, frame, conv,)


    p = min.(1 + Δ, p)
    p = max.(-Δ, p)

    T = eltype(p)
    α = T(1 - sharpness) / 2

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
    if sharpness > 0
        a = smooth(a, sesolid, sevoid)
    end
    a
end
