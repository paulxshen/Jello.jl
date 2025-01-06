# Define a container to hold any optimiser specific parameters (if any):
mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    opt
    m
    η
    A
    minchange
    maxchange
    ls
    function AreaChangeOptimiser(m;
        opt=Momentum(1, 0.0),
        #  opt=Adam(1, (0.8, 0.9)),
        minchange=0.001,
        maxchange=0.1)
        @show minchange, maxchange
        η = nothing
        A = sum(prod(size(m)))
        new(opt, m, η, A, minchange, maxchange, [])
    end
end

function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, A, opt, maxchange, minchange = o
    @assert all(m.p .== x)
    a0 = m() .> 0.5

    T = eltype(x)
    dA = 0
    x0 = deepcopy(x)
    s, x̄ = Optimisers.apply!(opt, s, x, x̄)
    x̄0 = deepcopy(x̄)

    maxdA = max(1, maxchange * A)
    mindA = max(1, minchange * A)
    if isnothing(o.η)
        o.η = 1
        mindA = maxdA
    end

    i = 0
    overshot = undershot = false
    while (1.0f-9 < o.η < 1.0f9) && (i == 0 || (dA > maxdA && !undershot) || (dA < mindA && !overshot))
        if i > 0
            c = T(1.05)
            if dA > maxdA
                o.η /= c
                overshot = true
            else
                o.η *= c
                undershot = true
            end
        end

        x̄ = T(o.η) * x̄0
        m.p .= x0 - x̄
        a = m() .> 0.5
        dA = sum(abs, a - a0)
        i += 1
    end

    m.p .= min.(1 + Δ, m.p)
    m.p .= max.(0 - Δ, m.p)
    x̄ = x0 - m.p
    m.p .= x0

    print("debug: ")
    @show o.η, dA
    println("fractional change in design: $(dA/A)")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    # @show jump = maximum(getfield.(x, :jump))

    opt_state = Optimisers.init(o.opt, x)
end

function update_loss!(o::AreaChangeOptimiser, l)
    push!(o.ls, l)
    if length(o.ls) > 1
        if o.ls[end] > o.ls[end-1]
            o.η /= 1.5
        else
            o.η *= 1.5
        end
    end
    # repair!(o.m)
end
update_loss!(a...) = 0

function repair!(m::InterpBlob)
    p = m.p
    p .= max.(0, p)
    p .= min.(1, p)
    m
end
function repair!(m)
    m
end