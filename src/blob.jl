function Blob(sz::Base.AbstractVecOrTuple;
    lmin,
    symmetries=(), periodic=false,
    init=0.5,
    init_pattern_level=nothing, init_pattern_spacing=nothing, init_pattern_diameter=nothing,
    contrast=1,
    F=Float32)

    init = F(init)
    sz = Tuple(sz)
    N = length(sz)
    symmetries = Symbol.(symmetries)
    if !isnothing(init_pattern_level)
        if isnothing(init_pattern_spacing)
            init_pattern_spacing = 2lmin
        end
        if isnothing(init_pattern_diameter)
            init_pattern_diameter = init_pattern_spacing / 2
        end
    end
    if !periodic
        if lmin == 1
            psz = sz
        else
            σ = 0.5lmin
            R2 = round(Int, 2σ)
            R1 = round(Int, lmin / 2)
            R = R1

            if :diagonal ∈ symmetries
                psz = Tuple(fill(maximum(sz), N))
            else
                psz = sz
            end
        end
        n = rand(F, psz)
        if init == 0.5
            p = n
        elseif init == 0
            p = 0.5n
        elseif init == 1
            p = 0.5n + 0.5
        else
            # p = ifelse.(init .> 0.5, init - 0.1, init + 0.1)
            p = 0.5init + 0.5n
        end
        p = F.(p)
        # b = zeros(Bool, size(p))
        # b[(:).(3, size(b) - 2)...] .= 1
        # p = ifelse.(b, p, 1 - p)
        p = pad(p, :replicate, R)
        if Symbol(init_pattern_level) == :invert
            b = zeros(Bool, size(p))
            perforate!(b, 1, init_pattern_spacing, init_pattern_diameter, R)
            p = ifelse.(b, 1 .- p, p)

        elseif !isnothing(init_pattern_level)
            perforate!(p, init_pattern_level, init_pattern_spacing, init_pattern_diameter, R)
        end

        n = 2R1 + 1
        W = ball(R1, N; normalized=true) do r
            (R1 - r + 1) / R1
            # exp(-(r / σ)^2 / 2)
        end |> F

        return ConvBlob(p, sz, symmetries, W, contrast)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)