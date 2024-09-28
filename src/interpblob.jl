struct InterpBlob
    a::AbstractArray
    A
    sz
    lmin
    lvoid
    lsolid
    frame
    symmetries
end
@functor InterpBlob (a, A)
Zygote.Params(m::InterpBlob) = Params([m.a])
Flux.trainable(m::InterpBlob) = (; a=m.a)
# Base.size(m::InterpBlob) = size(m.a)

function (m::InterpBlob)(sharpness::Real=0.99, lvoid=m.lvoid, lsolid=m.lsolid,)
    @unpack a, A, symmetries, sz, lmin, frame = m
    T = eltype(a)
    α = T(1 - sharpness)

    if !isnothing(A)
        a = reshape((A) * a, sz)
    end
    a = apply(symmetries, a)
    a = step(a, α)

    margin = Int.((size(frame, 1) - sz[1]) / 2)
    a = smooth(a, α, lvoid, lsolid, frame, margin)
end
