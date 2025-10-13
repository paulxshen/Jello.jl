function Blob(sz::Base.AbstractVecOrTuple;
    lmin=nothing, σ=0.5lmin,
    symmetries=[], periodic=false,
    init=0.5,
    contrast=1,
    F=Float32)

    sz = Tuple(sz)
    N = length(sz)
    symmetries = Symbol.(symmetries)

    if !periodic
        R = round(Int, 2σ)
        if :diagonal ∈ symmetries
            psz = Tuple(fill(maximum(sz), N))
        else
            psz = sz
        end

        w = 0.99
        n = rand(F, psz)
        p = w * init + (1 - w) * n
        p = F.(p)
        p = pad(p, :replicate, R)

        W = ball(R, N; normalized=true) do r
            exp(-(r / (σ))^2 / 2)
        end |> F

        return ConvBlob(p, sz, σ, lmin, symmetries, W, contrast)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)