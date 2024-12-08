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

function (m::InterpBlob)(sharpness::Real=0.995;)
    @unpack p, A, symmetries, sz, asz, frame, margin, sevoid, sesolid, conv = m
    @ignore_derivatives_vars (A, frame, conv,)

    T = eltype(p)
    α = T(1 - sharpness)

    p = abs.(p)

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
    smooth(a, α, sesolid, sevoid)
end
