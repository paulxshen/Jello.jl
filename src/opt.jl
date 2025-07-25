mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    m
    change
    η
    ρ
    x̄
    xs
    losses
    function AreaChangeOptimiser(m, change=0.001, η=1, ρ=0.1)
        new(m, change, η, ρ, 0, [], [])
    end
end

function invf(f, y; init=1, maxiters=100, reltol=0.2, abstol=1.0f-4, lims=(1.0f-20, 1.0f20))
    x = init
    a = b = nothing
    for i = 1:maxiters
        ϵ = f(x) - y
        (abs(ϵ) < max(reltol * abs(y), abstol) || x < lims[1] || x > lims[2]) && return x
        if ϵ > 0
            b = x
            if isnothing(a)
                x /= 2
            else
                x = (x + a) / 2
            end
        else
            a = x
            if isnothing(b)
                x *= 2
            else
                x = (x + b) / 2
            end
        end
    end
    @debug "invf: maxiters reached"
    x
end
function η_from_area_change(m::Base.AbstractVecOrTuple, dldm, change=0.01)
    A = sum(prod.(size.(m)))
    p0 = deepcopy(getfield.(m, :p))
    f0 = [m() for m in m]
    η = invf(change) do η
        sum(map(m, p0, f0, dldm) do m, p0, f0, dldm
            m.p .= p0 - η * dldm.p
            r = sum(abs, m() - f0)
            m.p .= p0
            r
        end) / A
    end
    η
end
η_from_area_change(m, dldm, change=0.01) = η_from_area_change((m,), (dldm,), change)
function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, losses, xs = o
    push!(xs, x)

    @debug extrema(m.p)
    @debug extrema(x̄)
    @assert !any(isnan, x̄)

    @assert all(m.p .== x)
    @assert length(xs) == length(losses)

    if length(xs) > 1
        if losses[end] >= losses[end-1]
            # w = 0.85
            # m.p .== w * xs[end-1] + (1 - w) * xs[end]
            o.η /= 1.2
            # o.ρ = 0.8o.ρ + 0.2
        else
            o.η *= 1.1
            # o.ρ *= 0.95
        end
    end

    x̄ .*= (0 .< x .< 1) .|| ((x .<= 0) .&& (x̄ .< 0)) .|| ((x .>= 1) .&& (x̄ .> 0))
    o.x̄ .= x̄
    # Z = maximum(abs, x̄)
    # heatmap(x̄ / Z, colormap=:seismic, colorrange=(-1, 1)) |> display
    # o.x̄ = o.ρ * o.x̄ + (1 - o.ρ) * x̄
    A = prod(size(m))
    a0 = m()
    x0 = deepcopy(x)

    if length(xs) == 1
        dA = 0
        invf(o.change * A; init=o.η) do η
            o.η = η

            x̄ = A * o.η * o.x̄
            m.p .= x0 - x̄
            m.p .= min.(1, m.p)
            m.p .= max.(0, m.p)
            dA = sum(abs, m() - a0)
        end
    else
        x̄ = A * o.η * o.x̄
        m.p .= x0 - x̄
        m.p .= min.(1, m.p)
        m.p .= max.(0, m.p)
        dA = sum(abs, m() - a0)
    end

    x̄ = x0 - m.p
    m.p .= x0

    @debug (; o.η, o.ρ)
    println("fractional change: $(dA/A)")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    nothing
end
