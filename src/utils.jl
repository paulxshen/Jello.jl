const α = 1.0f-6
function stepfunc(a::T, β) where {T}
    T(α) * tanh(β * (a - T(0.5))) + (a > 0.5 ? 1 - T(α) : T(α))
    # 2T(α) * (a - T(0.5)) + (a > 0.5 ? 1 - T(α) : T(α))
end
_ceil(x) = x == floor(Int, x) ? Int(x) + 1 : ceil(Int, x)
function ball(f, R, N=2; normalized=false)
    a = map(Base.product(fill(-R:R, N)...)) do r
        r = norm(collect(r))
        r > R + 0.001 ? 0 : f(r)
    end
    if normalized
        a /= sum(a)
    end
    a
end
function circle(r, d=2; normalized=false)
    n = round(Int, r - 0.01)
    a = [
        begin
            d = norm(v) - r
            if d < -0.5
                1
            elseif d > 0.5
                0
            else
                0.5 - d
            end
        end for v = Base.product(fill(-n:n, d)...)
    ] # circle
    if normalized
        a /= sum(a)
    end
    a
end
function cone(r, d; normalized=false)
    r = round(Int, r - 0.01)
    a = [1 - norm(v) / (r + 1) for v = Base.product(fill(-r:r, d)...)] # circle
    a = max.(a, 0)
    if normalized
        a /= sum(a)
    end
    a
end
function se(r, d=2)
    a = ball(x -> 1, r, d) .> 0
    # display(heatmap(a))
    centered(a)
end
function apply_symmetries(a, symmetries)
    if isempty(symmetries)
        return a
    end
    for s = symmetries
        if Symbol(s) == :diagonal
            a += a'
            a /= 2
        else
            dims = findfirst(==(string(s)), ["x", "y", "z"])
            if !isnothing(dims)
                a += reverse(a; dims)
                a /= 2
            end
        end
    end
    a
end

function perforate!(a::AbstractArray{T,N}, R, r) where {T,N}
    sz = size(a)
    n = prod(round.(Int, sz / R))
    for i = 1:n
        c = round.(Int, rand(T, N) * sz)
        lb = max.(round.(Int, c - r), 1)
        ub = min.(round.(Int, c + r), sz)
        a[(:).(lb, ub)...] .= 0.2
    end
    a
end