abstract type AbstractBlob end
struct ConvBlob<:AbstractBlob
    p::AbstractArray
    W::AbstractArray
    sz::Tuple
    repdims
    symdims
    meta
end
Base.size(m::ConvBlob) = m.sz
@functor ConvBlob (p,)

function (m::ConvBlob)()
    @unpack p, symdims, sz, repdims, W, meta = m
    @unpack contrast=meta
    ignore_derivatives() do
        p.=clamp.(p, 0, 1)
    end
    _ConvBlob(p, W, sz, repdims, symdims, contrast)
end

function _ConvBlob(a::AbstractArray{T,N}, W, sz, repdims, symdims, contrast) where {T,N}
    !isempty(symdims) && (a = apply_symdims(a, symdims, ))
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

    for dims=repdims
        a = repeat(a, outer=ifelse.(dims .== (1:N), sz, 1))
    end
    a
end
