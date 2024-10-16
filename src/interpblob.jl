struct InterpBlob
    a::AbstractArray
    A
    sz
    lmin
    lvoid
    lsolid
    vvoid
    vsolid
    frame
    symmetries
    ratio
end
@functor InterpBlob (a, A)
Flux.trainable(m::InterpBlob) = (; a=m.a)
# Base.size(m::InterpBlob) = size(m.a)

function _InterpBlob(m, f, vvoid=m.vvoid, vsolid=m.vsolid, sharpness::Real=0.99)
    #  vvoid::Real,vsolid,sharpness::Real=0.99)
    @unpack a, A, symmetries, sz, lmin, frame, ratio, lvoid, lsolid = m
    T = eltype(a)
    α = T(1 - sharpness)
    A = ignore_derivatives() do
        A
    end
    a = reshape((A) * a, ratio * sz)
    a = apply(symmetries, a)
    a = step(a, α)

    margin = Int.((size(frame, 1) - ratio * sz[1]) / 2)
    a = smooth(a, α, ratio * lvoid, ratio * lsolid, frame, margin)
    a = vvoid + (vsolid - vvoid) * a
    # if !withtensor
    f(a, ratio)
end

(m::InterpBlob)(f::Function, args...) = _InterpBlob(m, f, args...)
(m::InterpBlob)(args...) = _InterpBlob(m, downsample, args...)