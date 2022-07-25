# (tv) 2022-07-22
# First demo code using the moment algorithm to solve the MCT equations,
# schematic-model version (F12 model).

using DelimitedFiles
using LoopVectorization

const maxIter = 1000000
const accuracy = 1e-9

abstract type MCTmodel end

link!(model::MCTmodel, dict) = nothing

abstract type MCTresults end

struct F12Model <: MCTmodel
    v1::Float64
    v2::Float64
    nu::Float64
    function F12Model(v1::Real, v2::Real)
        new(v1, v2, 1.0)
    end
end
function kernel(model::F12Model, ϕ::Real, i::Int)::Real
    return model.v1 * ϕ + model.v2 * ϕ^2
end
cortype(model::F12Model)::DataType = Float64

mutable struct SjoegrenModel <: MCTmodel
    vs::Float64
    nu::Float64
    base::MCTmodel
    ϕ::MCTresults
    function SjoegrenModel(vs::Real, base::MCTmodel)
        new(vs, 1.0, base)
    end
end
function kernel(model::SjoegrenModel, ϕs::Real, i::Int)::Real
    ϕ = model.ϕ[i]
    return model.vs * ϕ * ϕs
end
cortype(model::SjoegrenModel)::DataType = Float64

link!(model::SjoegrenModel, dict) = (model.ϕ = dict[model.base])


struct MomentSolutions{T<:Vector{Float64}} <: MCTresults
    ϕ::T
    m::T
    dΦ::T
    dM::T
    function MomentSolutions(model::MCTmodel, blocksize::Int)
        halfsize = (blocksize+1) ÷ 2
        new{Vector{cortype(model)}}(Vector{cortype(model)}(undef,blocksize), Vector{cortype(model)}(undef,blocksize), Vector{cortype(model)}(undef,halfsize), Vector{cortype(model)}(undef,halfsize))
    end
end
function cortype(sol::MCTresults)
    return eltype(sol.ϕ)
end
Base.getindex(s::MomentSolutions{T}, i::Int) where {T} = getindex(s.ϕ, i)
Base.getindex(s::MomentSolutions{T}, I::Vararg{Int,N}) where {T,N} = getindex(s.ϕ, I)

struct SolutionStack{T<:MCTresults} <: AbstractArray{T,1}
    solutions::Vector{T}
    blocksize::Int
    function SolutionStack{T}(modelStack::Vector{MCTmodel},blocksize::Int) where {T}
        n = new([T(model,blocksize) for model in modelStack],blocksize)
        dict = Dict{MCTmodel,T}()
        for (model,sol) in zip(modelStack,n.solutions)
            dict[model] = sol
            link!(model, dict)
        end
        return n
    end
end
# methods to define the AbstractArray interface (read-only)
Base.size(s::SolutionStack{T}) where {T} = size(s.solutions)
Base.getindex(s::SolutionStack{T}, i::Int) where {T} = getindex(s.solutions, i)
Base.getindex(s::SolutionStack{T}, I::Vararg{Int,N}) where {T,N} = getindex(s.solutions, I)


function silentWriter(io::Union{Base.TTY,IOStream,Base.PipeEndpoint}, solStack::SolutionStack{T}, imin::Int, imax::Int, h::Float64) where T<:MCTresults
end

function solutionWriter(io::Union{Base.TTY,IOStream,Base.PipeEndpoint}, solStack::SolutionStack{T}, imin::Int, imax::Int, h::Float64) where T<:MCTresults
    halfblocksize = solStack.blocksize ÷ 2
    t = collect(h*(i-1) for i in imin:imax)
    v = [(collect((sol.ϕ[imin:imax],sol.m[imin:imax]) for sol in solStack)...)...]
    writedlm(io, [t v...])
end



function initial_values!(model::MCTmodel, sol::MomentSolutions, h::Float64, imax::Int)::Nothing
    @inbounds begin
    for i in 1:imax
        t = h*(i-1)
        sol.ϕ[i] = one(cortype(model)) - t/model.nu
        sol.m[i] = kernel(model, sol.ϕ[i], i)
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

function decimize!(sol::MomentSolutions, h::Float64, halfblocksize::Int)::Float64
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

function iterator(model::MCTmodel, ϕ::T, B::T, C::T, i::Int)::Tuple{T,T} where {T}
    m = 0.0
    for _ in 1:maxIter
        ϕtmp = copy(ϕ)
        m = kernel(model, ϕtmp, i)
        ϕ = m .* B - C
        if isapprox(ϕ, ϕtmp, rtol=accuracy, atol=accuracy)
            break
        end
    end
    return ϕ, m
end

function solve_block_body(sol::MomentSolutions, i::Int, h::Float64)::cortype(sol)
    @inbounds begin
    ibar = (i-1)÷2
    C = -sol.m[i-1] .* sol.dΦ[1] - sol.ϕ[i-1] .* sol.dM[1]
    for k in 2:ibar
        C += (sol.m[i-k+1] - sol.m[i-k]) .* sol.dΦ[k]
        C += (sol.ϕ[i-k+1] - sol.ϕ[i-k]) .* sol.dM[k]
    end
    if i-ibar > ibar+1
        C += (sol.ϕ[i-ibar] - sol.ϕ[i-ibar-1]) .* sol.dM[ibar+1]
    end
    C += sol.m[i-ibar] .* sol.ϕ[ibar+1]
    end
    return C
end


function solve_block!(model::MCTmodel, sol::MomentSolutions, h::Float64, istart::Int, iend::Int; lazy_moments::Bool=true)::Nothing
    A = sol.dM[1] + 1.0 + 1.5*model.nu/h
    B = (1.0 - sol.dΦ[1]) ./ A
    @inbounds begin
        for i in istart:iend
            C = solve_block_body(sol, i, h) ./ A
            C += (-2.0*sol.ϕ[i-1] + 0.5*sol.ϕ[i-2]) * model.nu/h ./ A
            sol.ϕ[i],sol.m[i] = iterator(model, sol.ϕ[i-1], B, C, i)
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

"""
    solve(model, blocksize, blocks)

Solve the MCT model given by `model`.
"""
function solve(model::MCTmodel, blocksize::Int, blocks::Int; h0::Float64=1e-9, maxinit::Int=10, io=stdout)::Nothing
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
 
function solve!(modelStack::Vector{MCTmodel}, solStack::SolutionStack{MomentSolutions}, blocks::Int; h0::Float64=1e-9, maxinit::Int=10, io=stdout, writer=silentWriter)::Nothing
    h = h0
    blocksize = solStack.blocksize
    halfblocksize = blocksize ÷ 2
    iend = (maxinit <= halfblocksize ? maxinit : halfblocksize)
    @inbounds begin
        for (model,sol) in zip(modelStack,solStack)
            initial_values!(model, sol, h, iend)
            if iend < halfblocksize
                solve_block!(model, sol, h, iend+1, halfblocksize, lazy_moments=false)
            end
        end
        writer(io, solStack, 1, halfblocksize, h)
        for d in 1:blocks
            hnew = h
            for (model,sol) in zip(modelStack,solStack)
                solve_block!(model, sol, h, halfblocksize+1, blocksize)
            end
            writer(io, solStack, halfblocksize+1, blocksize, h)
            for (model,sol) in zip(modelStack,solStack)
                hnew = decimize!(sol, h, halfblocksize)
            end
            h = hnew
        end
    end
end


open("/tmp/ref.dat", "w") do io
    @time solve(F12Model(0.99,0.), 512, 60; maxinit=50, h0=1e-9, io=io)
end
open("/tmp/1", "w") do io
    @time solve(F12Model(0.99,0.), 512, 60; maxinit=50, h0=1e-3, io=io)
end

f12 = F12Model(0.,3.95)
models = Vector{MCTmodel}([f12,SjoegrenModel(30.,f12)])
solutions = SolutionStack{MomentSolutions}(models,512)

open("/tmp/1v", "w") do io
    @time solve!(models,solutions,60;maxinit=50,io=io,writer=solutionWriter)
end
open("/tmp/1v2", "w") do io
    @time solve!(models,solutions,60;maxinit=50,io=io,writer=silentWriter)
end
