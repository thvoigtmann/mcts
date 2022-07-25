# (tv) 2022-07-22
# First demo code using the moment algorithm to solve the MCT equations,
# schematic-model version (F12 model).

using DelimitedFiles
using LoopVectorization

const maxIter = 1000000
const accuracy = 1e-9

abstract type MCTmodel end

struct F12Model <: MCTmodel
    v1::Float64
    v2::Float64
    nu::Float64
    function F12Model(v1::Real, v2::Real)
        new(v1, v2, 1.0)
    end
end

function kernel(model::F12Model, ϕ::Real)::Real
    return model.v1 * ϕ + model.v2 * ϕ^2
end
function cortype(model::F12Model)::DataType
    return Float64
end

abstract type MCTresults end

struct MomentSolutions{T<:Vector{Float64}} <: MCTresults
    ϕ::T
    m::T
    dΦ::T
    dM::T
    function MomentSolutions(model::MCTmodel, blocksize::Integer)
        halfsize = (blocksize+1) ÷ 2
        new{Vector{cortype(model)}}(Vector{cortype(model)}(undef,blocksize), Vector{cortype(model)}(undef,blocksize), Vector{cortype(model)}(undef,halfsize), Vector{cortype(model)}(undef,halfsize))
    end
end
function cortype(sol::MCTresults)
    return eltype(sol.ϕ)
end

function initial_values!(model::MCTmodel, sol::MomentSolutions, h::Float64, imax::Integer)::Nothing
    @inbounds begin
    for i in 1:imax
        t = h*(i-1)
        sol.ϕ[i] = one(cortype(model)) - t/model.nu
        sol.m[i] = kernel(model, sol.ϕ[i])
    end
    @turbo for i in 1:imax-1
        sol.dΦ[i] = 0.5 * (sol.ϕ[i] + sol.ϕ[i+1])
        sol.dM[i] = 0.5 * (sol.m[i] + sol.m[i+1])
    end
    sol.dΦ[imax] = sol.ϕ[imax]
    sol.dM[imax] = sol.m[imax]
    end
    return
end

function decimize!(sol::MomentSolutions, h::Float64, halfblocksize::Integer)::Float64
    imid = halfblocksize ÷ 2
    @inbounds begin
    for i in 1:imid-1
        di = i+i
        sol.dΦ[i] = 0.5 * (sol.dΦ[di-1] + sol.dΦ[di])
        sol.dM[i] = 0.5 * (sol.dM[di-1] + sol.dM[di])
    end
    @turbo for i in imid:halfblocksize-1
        di = i+i
        sol.dΦ[i] = 0.25 * (sol.ϕ[di-1] + 2*sol.ϕ[di] + sol.ϕ[di+1])
        sol.dM[i] = 0.25 * (sol.m[di-1] + 2*sol.m[di] + sol.m[di+1])
    end
    sol.dΦ[halfblocksize] = sol.ϕ[2halfblocksize]
    sol.dM[halfblocksize] = sol.m[2halfblocksize]
    for i in 1:halfblocksize
        di = i+i
        sol.ϕ[i] = sol.ϕ[di-1]
        sol.m[i] = sol.m[di-1]
    end
    end
    return 2h
end

function iterator(model::MCTmodel, ϕ::T, B::T, C::T)::Tuple{T,T} where {T}
    m = 0.0
    for _ in 1:maxIter
        ϕtmp = copy(ϕ)
        m = kernel(model, ϕtmp)
        ϕ = m*B - C
        if isapprox(ϕ, ϕtmp, rtol=accuracy, atol=accuracy)
            break
        end
    end
    return ϕ, m
end

function solve_block_body(sol::MomentSolutions, i::Integer, h::Float64)::cortype(sol)
    @inbounds begin
    ibar = (i-1)÷2
    C = -sol.m[i-1]*sol.dΦ[1] - sol.ϕ[i-1]*sol.dM[1]
    for k in 2:ibar
        C += (sol.m[i-k+1] - sol.m[i-k]) * sol.dΦ[k]
        C += (sol.ϕ[i-k+1] - sol.ϕ[i-k]) * sol.dM[k]
    end
    if i-ibar > ibar+1
        C += (sol.ϕ[i-ibar] - sol.ϕ[i-ibar-1]) * sol.dM[ibar+1]
    end
    C += sol.m[i-ibar] * sol.ϕ[ibar+1]
    end
    return C
end


function solve_block!(model::MCTmodel, sol::MomentSolutions, h::Float64, istart::Integer, iend::Integer; lazy_moments::Bool=true)::Nothing
    A = sol.dM[1] + 1.0 + 1.5*model.nu/h
    B = (1.0 - sol.dΦ[1]) / A
    @inbounds begin
        for i in istart:iend
            C = solve_block_body(sol, i, h) / A
            C += (-2.0*sol.ϕ[i-1] + 0.5*sol.ϕ[i-2]) * model.nu/h/A
            sol.ϕ[i],sol.m[i] = iterator(model, sol.ϕ[i-1], B, C)
            if !lazy_moments
                sol.dΦ[i-1] = 0.5 * (sol.ϕ[i-1] + sol.ϕ[i])
                sol.dM[i-1] = 0.5 * (sol.m[i-1] + sol.m[i])
            end
        end
        if !lazy_moments
            sol.dΦ[iend] = 0.5 * (sol.ϕ[iend+1] + sol.ϕ[iend])
            sol.dM[iend] = 0.5 * (sol.m[iend+1] + sol.m[iend])
        end
    end
    return
end

#


function solve(model::MCTmodel, blocksize::Integer, blocks::Integer; h0::Float64=1e-9, maxinit::Integer=10, io=stdout)::Nothing
    h = h0
    halfblocksize = blocksize÷2
    iend = (maxinit <= halfblocksize ? maxinit : halfblocksize)
    sol = MomentSolutions(model, blocksize)
    @inbounds begin
        initial_values!(model, sol, h, iend)
        if iend<halfblocksize
            solve_block!(model, sol, h, iend+1, halfblocksize, lazy_moments=false)
        end
        t = collect(h*(i-1) for i in 1:halfblocksize)
        writedlm(io, [t sol.ϕ[1:halfblocksize] sol.m[1:halfblocksize]])
        for d in 1:blocks
            solve_block!(model, sol, h, halfblocksize+1, blocksize)
            t = collect(h*(i-1) for i in halfblocksize+1:blocksize)
            writedlm(io, [t sol.ϕ[halfblocksize+1:blocksize] sol.m[halfblocksize+1:blocksize]])
            h = decimize!(sol, h, halfblocksize)
        end
    end
    return
end
 
open("/tmp/ref.dat", "w") do io
    @time solve(F12Model(0.99,0.), 512, 60; maxinit=50, h0=1e-9, io=io)
end
open("/tmp/1", "w") do io
    @time solve(F12Model(0.99,0.), 512, 60; maxinit=50, h0=1e-3, io=io)
end
