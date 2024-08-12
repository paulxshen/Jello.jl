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
        for d = symmetries
            if d == "diagonal"
                r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge
            elseif d == "anti-diagonal"
                r = (r + reverse(r, dims=1)') / 2
            elseif d == "inversion"
                r += reverse(r, dims=Tuple(1:ndims(r)))
                r /= 2
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
    A = B = 0
    ignore_derivatives() do
        T = typeof(r)
        m = Array(r) .> 0.5
        m0 = m
        if !isnothing(ose)
            mo = opening(m, ose)
        end
        if !isnothing(cse)
            mc = .!(closing(m, cse))
        end
        A = mo .| mc
        B = mc .> m0
        B = B .& (.!A)
        # A = T(m .== m0)
        # B = T(m .> m0)
    end
    # @show size(r), size(A), size(B), size(ose), size(cse)
    r .* A + B
end
function resize(a, sz)
    if length(sz) == 1
        return imresize(a, sz, method=ImageTransformations.Lanczos4OpenCV())
    end
    imresize(a, sz)
end

