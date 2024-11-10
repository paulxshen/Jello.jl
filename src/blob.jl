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
    lvoid=0, lsolid=0,
    vvoid=0, vsolid=1,
    symmetries=[], alg=:interp, init=nothing,
    ratio=1, frame=0, margin=0,
    F=Float32, T=F, verbose=false)

    N = length(sz)
    lmin = max(lsolid, lvoid)
    lmin /= sqrt(N)
    lmin = max(1, lmin)

    rmin = round(Int, (lmin - 1) / 2 + 0.01)
    if isa(frame, Number)
        margin = maximum(round.(Int, (lsolid, lvoid)))
        frame = fill(frame, sz + 2margin)
    end
    frame = T(frame)
    o = fill(ratio * margin + 1, N)
    frame = imresize(frame; ratio) .> 0.5

    if alg == :interp
        paramsize = round.(Int, sz ./ lmin) + 1
    elseif alg == :conv
        rfilter = round(Int, 1.6rmin)
        paramsize = sz + 2rfilter

    end

    da = 0.5
    if isnothing(init)
        a = 0.5 + da * randn(paramsize)
    end
    if isa(init, Number)
        a = fill(0.5 + 2da * (init - 0.5), paramsize)
        a += 0.01randn(paramsize)
    end
    a = T.(a)

    dl = 1 / ratio
    Rsolid = lsolid / 2 / dl
    Rvoid = lvoid / 2 / dl
    R = 4
    ker = circle(R)
    # @show A = sum(ker)
    A = π * R^2
    levels = [0, ica(R, Rsolid, Rsolid - 0.5), A - ica(R, Rvoid, Rvoid + 0.5), A]
    if alg == :interp
        supersize = ratio * sz
        J = LinearIndices(a)
        I = LinearIndices(supersize)
        A = map(CartesianIndices(Tuple(supersize))) do i
            _i = I[i]
            i = Tuple(i)
            i = 1 + (i - 1) .* (paramsize - 1) ./ (supersize - 1)
            i = Float32.(i)
            p = floor.(Int, i)
            q = ceil.(Int, i)
            stack(vec([Int32[_i, J[j...], round.(Int, 1000prod(1 - abs.(i - j)))] for j = Base.product([p[i] == q[i] ? (p[i],) : (p[i], q[i]) for i = 1:length(i)]...)]))
        end
        A = reduce(hcat, vec(A))'
        i, j, v = eachcol(A)
        A = sparse(i, j, convert.(T, v / 1000))
        a = vec(a)
        # nn = [
        #     map(getindex.(getindex.(t, 1), i)) do c
        #         c = min.(c, size(a))
        #         l[c...]
        #     end for i = 1:2^d
        # ]
        # w = [getindex.(getindex.(t, 2), i) for i = 1:2^d]
        # nn = w = 0
        vvoid, vsolid = T(vvoid), T(vsolid)
        return InterpBlob(a, A, sz, lmin, lvoid, lsolid, vvoid, vsolid, frame, o, symmetries, ratio, ker, levels)
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

