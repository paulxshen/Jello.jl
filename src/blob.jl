function Blob(sz::Tuple;
    lmin,
    periodic=false,
    init=1,
    topopt=false,
    repdims=[],
    symdims=[],
    anchordims=[],
    F=Float32)

    N = length(sz)
    contrast=1
    meta=Dict{Symbol,Any}(pairs((; contrast, topopt)))

    repdims, symdims, anchordims = unique.([repdims, symdims, anchordims])

    if !periodic
        σ = lmin / 4
        R = round(Int, 2σ)

        # if :diagonal ∈ symdims
        #     psz = Tuple(fill(maximum(sz), N))
        # else
        #     psz = sz
        # end

        isa(repdims, Int) && (repdims = [repdims])
        I=collect(1:N)
        if isempty(repdims)
            psz=sz
        else
            for i=sort(repdims, rev=true)
                for (j, x) = enumerate(anchordims)
                    abs(x) == i && (anchordims[j] = sign(x) * (abs(x) - 1))
                end
                for (j, x) = enumerate(symdims)
                    x == i && (symdims[j] -= 1)
                end
                deleteat!(I, i)
            end
            psz=sz[I]
        end
        n=length(psz)

        if init==1
            w = 0.99
            p = rand(F, psz)
            p = w * init + (1 - w) * p
        else
            @assert size(init)==sz
            p=init[ifelse.((1:N) .∈ (I,), (:,), 1)...]
        end
        p = F.(p)
        # p = pad(p, :replicate, R)

        W = ball(R, n; normalized=true) do r
            exp(-(r / (σ))^2 / 2)
        end |> F

        lopen=(1:n) .∉ (-anchordims,)
        ropen=(1:n) .∉ (anchordims,)

        return ConvBlob(p, W, sz, repdims, symdims, lopen, ropen, meta)
    else
    end
end
Blob(sz::AbstractVector; kw...) = Blob(Tuple(sz); kw...)
Blob(sz...; kw...) = Blob(sz; kw...)

function set!(m::AbstractBlob, k, v)
    @unpack meta=m
    if k==:contrast
        !meta[:topopt] && error("cannot set contrast when topopt=false")
    end
    meta[k]=v
end