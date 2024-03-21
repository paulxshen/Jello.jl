struct FourierBlob
    ar::AbstractArray
    ai::AbstractArray
    contrast
    sz
    ose
    cse
    symmetries
    diagonal_symmetry
end
@functor FourierBlob (ar, ai)
Base.size(m::FourierBlob) = m.sz
function FourierBlob(m::FourierBlob, sz...; contrast=m.contrast,)
    FourierBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
end
