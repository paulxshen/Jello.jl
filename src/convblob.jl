mutable struct ConvBlob
    p::AbstractArray
    sz
    σ
    lmin
    symmetries
    W
    contrast
end
Base.size(m::ConvBlob) = m.sz
@functor ConvBlob (p,)

function (m::ConvBlob)()
    @unpack p, σ, symmetries, sz, W, contrast, lmin = m
    p = _ConvBlob(p, symmetries, sz, W, contrast, σ, lmin)
    # @debug p |> extrema
    # foo(p)
end

function _ConvBlob(a::AbstractArray{T,N}, symmetries, sz, W, contrast, σ, lmin) where {T,N}
    @nograd W, σ, lmin
    a0 = a
    if !isnothing(symmetries)
        a = apply_symmetries(a, symmetries, sz)
    end
    # @debug a |> extrema

    a = conv(reshape(a, size(a)..., 1, 1), reshape(W, size(W)..., 1, 1))
    a = dropdims(a, dims=(N + 1, N + 2))
    a = resize(a, sz)
    contrast = T(contrast)

    if !isnothing(lmin) && (lmin = round(Int, lmin)) > 0
        a += ignore_derivatives() do
            b = a .> 0.5
            if iseven(lmin)
                lmin += 1
            end
            R = (lmin - 1) / 2
            C = ntuple(_ -> R + 1, N)
            se = map(CartesianIndices(ntuple(_ -> lmin, N))) do I
                norm(Tuple(I) - C) <= R + 0.01
            end |> centered
            closing(b, se) - b
        end / 2
    end

    m = b = 0
    ignore_derivatives() do
        b = a .> 0.5
        r = abs.(a - 0.5)
        m = pad(zeros(Bool, size(a) - 2), 1, 1)
        for dims = 1:N
            db = diff(b; dims)
            dr = diff(r; dims)
            s = dims .== 1:N

            I = ifelse.(s, (1:size(a, dims)-1,), (:,))
            m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .> 0))
            I = ifelse.(s, (2:size(a, dims),), (:,))
            m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .< 0))
        end
    end
    a = a .* m + .!(m) .* (contrast * b + (1 - contrast) / 2 * a)
    a = max.(a, 0)
    a = min.(a, 1)
end
