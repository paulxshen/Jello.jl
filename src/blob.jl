footer = "Jello.jl 2024 (c) Paul Shen at Luminescent AI\n<pxshen@alumni.stanford.edu>"
"""
    Blob(sz...; nbasis=4, contrast=1, alg=:interpolation, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[])
    (m::RealBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `RealBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `edge length / nbasis`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array
- `alg`: `:interpolation` `:fourier`
- `contrast`: edge sharpness
- `nbasis`: # of basis along each dimension
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering) or `:auto` (automatically set wrt `nbasis`)
- `rminfill`: same as `rmin` but only applied to fill (bright) features
- `rminvoid`: ditto
- `symmetries`: symmetry dimensions
"""
function Blob(sz...;
    lmin=nothing, nbasis=4, alg=:interpolation, init=nothing, contrast=1, T=Float32,
    rmin=nothing, rminfill=rmin, rminvoid=rmin,
    symmetries=[], diagonal_symmetry=false,
    verbose=true)
    d = length(sz)
    if lmin != nothing
        nbasis = round.(Int, sz ./ lmin) + 1
    end
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
    end
    com = """
          - output size: $sz
          - edge contrast : $contrast
          
          Morphological filtering (skipped if nothing )
          - min fill radii: $rminfill
          - min void radii: $rminvoid
          
          Symmetry dimensions: $symmetries
          
          Suppress this message by verbose=false
          $footer
          """

    if alg == :interpolation
        a = if isnothing(init)
            rand(T, nbasis...)
        elseif isa(init, Number)
            T(init) * ones(T, nbasis...)
        else
            resize(T.(init), nbasis)
        end

        d = ndims(a)
        n = length(a)
        N = prod(sz)
        # A = zeros(Int, 3, 2^d * N)
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
        # nn = [
        #     map(getindex.(getindex.(t, 1), i)) do c
        #         c = min.(c, size(a))
        #         l[c...]
        #     end for i = 1:2^d
        # ]
        # w = [getindex.(getindex.(t, 2), i) for i = 1:2^d]
        nn = w = 0
        ose, cse = se(rminfill, rminvoid)

        if verbose
            @info """
        Blob configs
        
        Geometry generation 
        - algorithm: real space interpolation
        - interpolation grid: $nbasis
        $com
        """
        end
        return RealBlob(a, A, T(contrast), sz, ose, cse, symmetries,)
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

        ose = cse = nothing
        v = minimum(round.(Int, sz ./ nbasis ./ 4))
        rminfill == :auto && (rminfill = v)
        rminvoid == :auto && (rminvoid = v)

        w = "rmin too high relative to nbasis. much of generated geometry may get erased by morphological filtering . consider setting to `:auto` which in this case evaluates to $v"

        if !isnothing(rminfill,)
            if rminfill > v
                @warn w
            end
            ose = circle(rminfill, d) |> centered
        end
        if !isnothing(rminvoid)
            if rminvoid > v
                @warn w
            end
            cse = circle(rminvoid, d) |> centered
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

        return FourierBlob(ar, ai, T(contrast), sz, ose, cse, symmetries, diagonal_symmetry)
    end
end
Blob(sz::Tuple; kw...) = Blob(sz...; kw...)

