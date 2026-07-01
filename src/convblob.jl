mutable struct ConvBlob
    p::AbstractArray
    sz
    lmin
    strict
    symmetries
    W
    contrast
end
Base.size(m::ConvBlob) = m.sz
@functor ConvBlob (p,)

function (m::ConvBlob)()
    @unpack p, symmetries, sz, W, contrast, lmin, strict = m
    p = _ConvBlob(p, symmetries, sz, W, contrast, lmin, strict)
    # @debug p |> extrema
end

function _ConvBlob(a::AbstractArray{T,N}, symmetries, sz, W, contrast, lmin, strict) where {T,N}
    @nograd W, lmin
    if !isnothing(symmetries)
        a = apply_symmetries(a, symmetries, sz)
    end
    # @debug a |> extrema

    a = conv(reshape(a, size(a)..., 1, 1), reshape(W, size(W)..., 1, 1))
    a = dropdims(a, dims=(N + 1, N + 2))
    a = resize(a, sz)

    contrast==0 && return a

    contrast = T(contrast)
    m = b = 0
    ignore_derivatives() do
        b = a .> 0.5
        r = abs.(a - 0.5)
        m = pad(zeros(Bool, size(a) - 2), 1, 1)
        for dims = 1:N
            db = diff(b; dims)
            dr = diff(r; dims)
            s = dims .== 1:N

            I = ifelse.(s, (1:(size(a, dims)-1),), (:,))
            m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .> 0))
            I = ifelse.(s, (2:size(a, dims),), (:,))
            m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .< 0))
        end
    end
    a = a .* m + .!(m) .* (contrast * b + (1 - contrast) * a)
end
