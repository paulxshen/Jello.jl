# Define a container to hold any optimiser specific parameters (if any):
mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    opt
    m
    η
    A
    minchange
    maxchange
    function AreaChangeOptimiser(m; minchange=0, maxchange=1)
        η = 1
        opt = Adam(1)
        # jump = m.jump * (length(m.symmetries) + 1)
        A = sum(prod(size(m)))
        new(opt, m, η, A, minchange, maxchange)
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

    i = 0
    maxdA = max(1, maxchange * A)
    mindA = max(1, minchange * A)
    overshot = undershot = false
    while i < 100 && (i == 0 || (dA > maxdA && !undershot) || (dA < mindA && !overshot))
        if i > 0
            if dA > maxdA
                o.η /= T(1.05)
                overshot = true
            else
                o.η *= T(1.05)
                undershot = true
            end
        end

        x̄ = o.η * x̄0
        m.p .= x0 - x̄
        a = m() .> 0.5
        dA = sum(abs, a - a0)
        i += 1
    end
    m.p .= x0

    print("debug: ")
    @show o.η, dA
    println("fractional change in design: $(dA/A)")
    println("")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    # @show jump = maximum(getfield.(x, :jump))

    opt_state = Optimisers.init(o.opt, x)
end