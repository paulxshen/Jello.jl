_ceil(x) = x == floor(Int, x) ? Int(x) + 1 : ceil(Int, x)
function circle(r, d)
    r = round(Int, r)
    [norm(v) <= r for v = Base.product(fill(-r:r, d)...)] # circle
end
function se(rminfill, rminvoid, d=2)
    ose = cse = nothing
    if !isnothing(rminfill,)
        # if rminfill > v
        #     # @warn w
        # end
        ose = circle(rminfill, d) |> centered
    end
    if !isnothing(rminvoid)
        # if rminvoid > v
        #     # @warn w
        # end
        cse = circle(rminvoid, d) |> centered
    end
    ose, cse
end
function apply(symmetries, r)
    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
        # elseif diagonal_symmetry == true
        #     r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge
    end
    r
end
function apply(σ, contrast::Real, r)
    if !isinf(contrast)
        r /= mean(abs.(r))
        # r = σ.(contrast * σ.(r))
        r = σ.(contrast * r)
    else
        r = r .> 0.5
    end
end
function apply(ose, cse, r)
    isnothing(ose) && isnothing(cse) && return r
    m = m0 = 0
    ignore_derivatives() do
        m = r .> 0.5
        m0 = m
        if !isnothing(ose)
            m = opening(m, ose)
        end
        if !isnothing(cse)
            m = closing(m, cse)
        end
    end
    r .* (m .== m0) + (m - m0 .> 0)
end
function imresize(a, sz)

end

