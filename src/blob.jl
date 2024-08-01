footer = "Jello.jl 2024 (c) Paul Shen at Luminescent AI\n<pxshen@alumni.stanford.edu>"
"""
    Blob(sz...; nbasis=4, contrast=1, alg=:interpolation, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetry_dims=[])
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
- `symmetry_dims`: symmetry dimensions
"""
function Blob(sz...;
    lmin=nothing, nbasis=4, alg=:interpolation, init=nothing, contrast=1, T=Float32,
    rmin=nothing, rminfill=rmin, rminvoid=rmin,
    symmetry_dims=[], diagonal_symmetry=false,
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
          
          Symmetry dimensions: $symmetry_dims
          
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

        t = map(CartesianIndices(Tuple(sz))) do i
            i = collect(T.(1 .+ (Tuple(i) .- 1) .* (size(a) .- 1) ./ (sz .- 1)))
            nn = collect(Base.product(range.(floor.(Int, i), _ceil.(i))...))

            w = map(nn) do k
                prod(1 .- abs.(i .- k))
            end
            nn, w
        end
        l = LinearIndices(a)
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
        return RealBlob(a, nn, w, T(contrast), sz, ose, cse, symmetry_dims,)
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

        return FourierBlob(ar, ai, T(contrast), sz, ose, cse, symmetry_dims, diagonal_symmetry)
    end
end
Blob(sz::Tuple; kw...) = Blob(sz...; kw...)

