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
function apply(symmetry_dims, r)
    if !isempty(symmetry_dims)
        for d = symmetry_dims
            if d=="diag"
                    r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge
            elseif d=="antidiag"
                    r = (r + reverse(r, dims=1)') / 2
            else
            r += reverse(r, dims=Int.(d))
            r /= 2
        end
        end
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
function resize(a, sz)
    if length(sz) == 1
        return imresize(a, sz, method=ImageTransformations.Lanczos4OpenCV())
    end
    imresize(a, sz)
end

