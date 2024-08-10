struct RealBlob
    a::AbstractArray
    A
    contrast::Real
    sz
    ose
    cse
    symmetries
end
@functor RealBlob (a, A)
Zygote.Params(m::RealBlob) = Params([m.a])
using Zygote
# Zygote.params
Flux.trainable(m::RealBlob) = (; a=m.a)
Base.size(m::RealBlob) = m.sz


function (m::RealBlob)()
    @unpack a, contrast, A, sz, ose, cse, symmetries, = m
    # itp = interpolate(a, BSpline(Linear()))
    # a = σ.(a)
    # r = map(nn) do (k, w)
    #     sum(zip(k, w)) do (k, w)
    #         a[k...] * w
    #     end
    # end
    # r = sum([T(getindex.((Array(a),), k)) .* w for (k, w) = zip(nn, w)])

    T = eltype(a)
    # A = ignore_derivatives() do
    #     A / T(1000)
    # end
    A, contrast = ignore_derivatives() do
        A, contrast
    end
    r = reshape((A) * a, sz)
    # r = map(CartesianIndices(Tuple(sz))) do i
    #     i = Tuple(i)
    #     i = 1 + (i - 1) .* (size(a) - 1) ./ (sz - 1)
    #     i = Float32.(i)
    #     interp(a, i)
    # end

    # r = Buffer(a, sz)
    # imresize!(r, a)
    # r = copy(r)
    r = apply(symmetries, r)
    r = r - T(0.5)
    v = mean(abs.(r))
    if v != 0
        r /= v
    end
    r = NNlib.σ.(contrast * r)
    r = apply(ose, cse, r)
end


# function RealBlob(m::RealBlob, sz...; contrast=m.contrast,)
#     RealBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
# end
