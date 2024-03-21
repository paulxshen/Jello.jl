struct RealBlob
    a::AbstractArray
    nn
    contrast::Real
    sz
    ose
    cse
    symmetries
end
@functor RealBlob (a,)
Base.size(m::RealBlob) = m.sz


function (m::RealBlob)()
    @unpack a, contrast, nn, sz, ose, cse, symmetries, = m
    # itp = interpolate(a, BSpline(Linear()))
    # a = σ.(a)
    r = map(nn) do (k, w)
        sum(zip(k, w)) do (k, w)
            a[k...] * w
        end
    end
    # r = Buffer(a, sz)
    # imresize!(r, a)
    # r = copy(r)
    r = apply(symmetries, r)
    # r = σ.(contrast * r)
    r = apply(σ, contrast, r)
    r = apply(ose, cse, r)
end


# function RealBlob(m::RealBlob, sz...; contrast=m.contrast,)
#     RealBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
# end
