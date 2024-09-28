# - `alg`: `:interpolation` `:fourier`
"""
    Blob(sz; contrast=20, alg=:interpolation, T=Float32, rmin=nothing, lsolid=rmin, lvoid=rmin, symmetries=[])
    (m::ConvBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `ConvBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `lmin`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array

Keyword Args
- `contrast`: edge sharpness
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering)
- `lsolid`: same as `rmin` but only applied to solid (bright) features
- `lvoid`: ditto
- `symmetries`: symmetry dimensions
"""
function Blob(sz::Base.AbstractVecOrTuple;
    lmin=0, lvoid=0, lsolid=0,
    symmetries=[], alg=:interp, init=nothing,
    frame=0,
    F=Float32, T=F, verbose=false)

    N = length(sz)
    # lmin, lsolid, lvoid = round.(Int, [lmin, lsolid, lvoid])
    lmin = max(lmin, lsolid, lvoid)
    lmin /= 1.2sqrt(N)
    lmin = max(1, lmin)

    rmin = round((lmin - 1) / 2 + 0.01)
    if isa(frame, Number)
        margin = maximum(round.((lsolid, lvoid)))
        frame = fill(frame, sz + 2margin)
    end

    if alg == :interp
        psz = round.(Int, sz ./ lmin) + 1
    elseif alg == :conv
        rfilter = round(1.6rmin)
        psz = sz + 2rfilter

    end

    da = 0.02
    if isnothing(init)
        a = da * randn(psz)
    end
    if isa(init, Number)
        a = fill((-da) + 2 * da * init, psz)
    end

    if alg == :conv
        if isa(init, Number)
            a += 0.5randn(psz) * rmin^(N / 2)
        end
        a = T.(a)
        n = 2rfilter + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= conic(rfilter, 2)
        return ConvBlob(a, conv, rmin, symmetries)
    elseif alg == :interp
        if isa(init, Number)
            # a += -0.9sign(init) * rand(T, psz)
        end
        a = T.(a)

        #     resize(T.(init), nbasis)
        # end

        # n = length(a)
        # N = prod(sz)
        # A = zeros(Int, 3, 2^d * N)
        # if lmin == 1
        if any(sz .< psz)
            A = nothing
        else
            J = LinearIndices(a)
            I = LinearIndices(sz)
            A = map(CartesianIndices(Tuple(sz))) do i
                _i = I[i]
                i = Tuple(i)
                i = 1 + (i - 1) .* (size(a) - 1) ./ (sz - 1)
                i = Float32.(i)
                p = floor(i)
                q = ceil(i)
                stack(vec([Int32[_i, J[j...], round(1000prod(1 - abs.(i - j)))] for j = Base.product([p[i] == q[i] ? (p[i],) : (p[i], q[i]) for i = 1:length(i)]...)]))
            end
            A = reduce(hcat, vec(A))'
            i, j, v = eachcol(A)
            A = sparse(i, j, T(v / 1000))
            a = vec(a)
        end
        # nn = [
        #     map(getindex.(getindex.(t, 1), i)) do c
        #         c = min.(c, size(a))
        #         l[c...]
        #     end for i = 1:2^d
        # ]
        # w = [getindex.(getindex.(t, 2), i) for i = 1:2^d]
        # nn = w = 0

        return InterpBlob(a, A, sz, lmin, lvoid, lsolid, frame, symmetries,)
    elseif alg == :fourier
        if isnothing(init)
            ar = randn(T, nbasis...)
            ai = randn(T, nbasis...)
        else
            ar = zeros(T, nbasis...)
            ai = zeros(T, nbasis...)
            if init == 1
                ar[1] = 1
            end
        end

        if verbose
            @info """
        Blob configs
        
        Geometry generation 
        - algorithm: Fourier basis
        - Fourier k-space size (# of Fourier basis per dimension): $nbasis
        $com
        """
        end

        return FourierBlob(ar, ai, T(contrast), sz, lsolid, lvoid, symmetries, diagonal_symmetry)
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)

