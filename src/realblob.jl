mutable struct RealBlob
    a::AbstractArray
    A
    contrast::Real
    sz
    ose
    cse
    symmetry_dims
end
@functor RealBlob (a,)
Zygote.Params(m::RealBlob) = Params([m.a])
using Zygote
# Zygote.params
Flux.trainable(m::RealBlob) = (; a=m.a)
Base.size(m::RealBlob) = m.sz


function (m::RealBlob)()
    @unpack a, contrast, A, sz, ose, cse, symmetry_dims, = m
    # itp = interpolate(a, BSpline(Linear()))
    # a = σ.(a)
    # r = map(nn) do (k, w)
    #     sum(zip(k, w)) do (k, w)
    #         a[k...] * w
    #     end
    # end
    T = typeof(a)
    # r = sum([T(getindex.((Array(a),), k)) .* w for (k, w) = zip(nn, w)])

    r = reshape(A * a / 1000, sz)
    # r = map(CartesianIndices(Tuple(sz))) do i
    #     i = Tuple(i)
    #     i = 1 + (i - 1) .* (size(a) - 1) ./ (sz - 1)
    #     i = Float32.(i)
    #     interp(a, i)
    # end

    # r = Buffer(a, sz)
    # imresize!(r, a)
    # r = copy(r)
    r = apply(symmetry_dims, r)
    T = eltype(a)
    r = r - T(0.5)
    v = mean(abs.(r))
    if v != 0
        r /= v
    end
    r = NNlib.σ.(contrast * r)
    r = apply(ose, cse, r)
end


# function RealBlob(m::RealBlob, sz...; contrast=m.contrast,)
#     RealBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetry_dims, m.diagonal_symmetry)
# end
