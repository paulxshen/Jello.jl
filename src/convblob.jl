struct ConvBlob
    a::AbstractArray

    conv
    rmin
    symmetries
end
@functor ConvBlob (a, conv)
Zygote.Params(m::ConvBlob) = Params([m.a])
Flux.trainable(m::ConvBlob) = (; a=m.a)
# Base.size(m::ConvBlob) = size(m.a)

function (m::ConvBlob)(α::Real=0.03, rmin=0, rvalssolid=rmin, rvalsvoid=rmin,)
    @unpack a, conv, symmetries, = m
    T = eltype(a)
    α = T(α)

    # v = mean(abs.(a))
    # if v != 0
    #     a *= T(0.5) / v
    # end

    N = ndims(a)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))


    # a = tanh.(α * m.rmin * a)
    a = step(a, α)

    a = (a + 1) / 2
    a = smooth(a, rvalssolid, rvalsvoid)
    a = apply(symmetries, a)
end
