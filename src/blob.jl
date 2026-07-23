function Blob(sz::Tuple;
    lmin,
    periodic=false,
    init=1,
    optimize_shape_only=false,
    symdims=[],
    repdims=[],
    F=Float32)

    N = length(sz)
    contrast=1
    meta=Dict{Symbol,Any}(pairs((; contrast, optimize_shape_only)))

    if !periodic
        σ = lmin / 2
        R = round(Int, 2σ)

        # if :diagonal ∈ symdims
        #     psz = Tuple(fill(maximum(sz), N))
        # else
        #     psz = sz
        # end

        isa(repdims, Int) && (repdims = [repdims])
        if isempty(repdims)
            psz=sz
        else
            I=collect(1:N)
            [deleteat!(I, i) for i=sort(repdims, rev=true)]
            psz=sz[I]
        end

        w = 0.99
        n = rand(F, psz)
        p = w * init + (1 - w) * n
        p = F.(p)
        p = pad(p, :replicate, R)

        W = ball(R, N; normalized=true) do r
            exp(-(r / (σ))^2 / 2)
        end |> F

        return ConvBlob(p, W, sz, repdims, symdims, meta)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)

function set!(m::AbstractBlob, k, v)
    @unpack meta=m
    if k==:contrast
        if !meta[:optimize_shape_only]
            meta[k]=v
        end
    end
end