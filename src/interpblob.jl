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
    start
    symmetries
    ratio
    ker
    levels
end
@functor InterpBlob (a, A)
Flux.trainable(m::InterpBlob) = (; a=m.a)
# Base.size(m::InterpBlob) = size(m.a)

function _InterpBlob(m, f, vvoid=m.vvoid, vsolid=m.vsolid, sharpness::Real=0.99; withloss=false)
    #  vvoid::Real,vsolid,sharpness::Real=0.99)
    @unpack a, A, symmetries, sz, lmin, frame, start, ratio, lvoid, lsolid, ker, levels = m
    @ignore_derivatives_vars (A, ker, levels, frame)

    T = eltype(a)
    α = T(1 - sharpness)
    a = reshape((A) * a, ratio * sz)
    a = apply(symmetries, a)
    a = step(a, α)

    # margin = Int.((size(frame, 1) - ratio * sz[1]) / 2)
    # a = smooth(a, α, ratio * lvoid, ratio * lsolid, frame, margin)
    a = vvoid + (vsolid - vvoid) * a
    # a = [norm(collect(Tuple(i)) - [25, 25]) < 9.5 for i = CartesianIndices((50, 50))]

    if withloss
        l = loss(levels, ker, a, frame, start)
    end
    if !isnothing(f)
        a = f(a, ratio)
    end
    if withloss
        return a, l
    end
    a
end

(m::InterpBlob)(f::Union{Function,Nothing}, args...; kw...) = _InterpBlob(m, f, args...; kw...)
(m::InterpBlob)(args...; kw...) = _InterpBlob(m, downsample, args...; kw...)