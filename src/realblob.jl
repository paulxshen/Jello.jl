struct RealBlob
    a::AbstractArray
    nn
    w
    contrast::Real
    sz
    ose
    cse
    symmetry_dims
end
@functor RealBlob (a,)
Base.size(m::RealBlob) = m.sz


function (m::RealBlob)()
    @unpack a, contrast, nn, w, sz, ose, cse, symmetry_dims, = m
    # itp = interpolate(a, BSpline(Linear()))
    # a = σ.(a)
    # r = map(nn) do (k, w)
    #     sum(zip(k, w)) do (k, w)
    #         a[k...] * w
    #     end
    # end
    r = sum([getindex.((a,), k) .* w for (k, w) = zip(nn, w)])
    # r = Buffer(a, sz)
    # imresize!(r, a)
    # r = copy(r)
    r = apply(symmetry_dims, r)
    v = mean(abs.(r))
    if v != 0
        r /= v
    end
    r = tanh.(contrast * r)
    r = max.(r, 0)
    r = min.(r, 1)
    r = apply(ose, cse, r)
end


# function RealBlob(m::RealBlob, sz...; contrast=m.contrast,)
#     RealBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetry_dims, m.diagonal_symmetry)
# end
