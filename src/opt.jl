# Define a container to hold any optimiser specific parameters (if any):
mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    opt
    m
    η
    A
    maxchange
    function AreaChangeOptimiser(m; maxchange=1)
        η = 10000
        opt = Adam()
        # jump = m.jump * (length(m.symmetries) + 1)
        A = sum(prod(size(m)))
        new(opt, m, η, A, maxchange)
    end
end

function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, A, opt, maxchange = o
    m.p .= x
    a0 = m() .> 0.5

    T = eltype(x)
    dA = 0
    x̄0 = deepcopy(x̄)
    x0 = deepcopy(x)
    s, x̄ = Optimisers.apply!(opt, s, x, x̄)

    i = 0
    maxdA = max(4, maxchange * A)
    # mindA = max(1,)
    while i == 0 || dA > maxdA
        if i > 0
            if dA > maxdA
                o.η /= T(1.05)
            else
                o.η *= T(1.05)
            end
        end

        global _a = x̄, x̄0
        x̄ = o.η * x̄0
        x = x0 - x̄
        m.p .= x
        a = m() .> 0.5
        dA = sum(a - a0) do a
            sum(abs, a)
        end
        i += 1
    end

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