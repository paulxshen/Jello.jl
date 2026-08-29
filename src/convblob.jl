abstract type AbstractBlob end
struct ConvBlob<:AbstractBlob
    p::AbstractArray
    W::AbstractArray
    sz::Tuple
    repdims
    symdims
    lopen
    ropen
    meta
end
Base.size(m::ConvBlob) = m.sz
@functor ConvBlob (p,)

function (m::ConvBlob)(; rep=true)
    @unpack p, symdims, sz, repdims, W, meta, lopen, ropen = m
    @unpack contrast=meta
    ignore_derivatives() do
        p.=clamp.(p, 0, 1)
    end
    _ConvBlob(p, W, sz, repdims, symdims, lopen, ropen, contrast, rep)
end

function _ConvBlob(a::AbstractArray{T,n}, W, sz, repdims, symdims, lopen, ropen, contrast, rep) where {T,n}
    N=length(sz)

    a = apply_symdims(a, symdims)

    R=(size(W)-1) .÷ 2
    a=pad(a, :replicate, R)

    a = conv(reshape(a, size(a)..., 1, 1), reshape(W, size(W)..., 1, 1))
    a = dropdims(a, dims=(n + 1, n + 2))

    if contrast>0
        contrast = T(contrast)
        m = b = 0
        ignore_derivatives() do
            b = a .> 0.5
            r = abs.(a - 0.5)
            m = pad(zeros(Bool, (size(a) - (lopen + ropen))...), true, lopen, ropen)
            for dims = 1:n
                db = diff(b; dims)
                dr = diff(r; dims)
                s = dims .== 1:n

                I = ifelse.(s, (1:(size(a, dims)-1),), (:,))
                m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .> 0))
                I = ifelse.(s, (2:size(a, dims),), (:,))
                m[I...] = m[I...] .|| ((db .!= 0) .&& (dr .< 0))
            end
        end

        a = a .* m + .!(m) .* (contrast * b + (1 - contrast) * a)
    end

    (!rep || isempty(repdims)) && return a

    v=1:N .∈ (repdims,)
    a=reshape(a, (ifelse.(v, 1, sz))...)
    repeat(a, outer=ifelse.(v, sz, 1))
end
