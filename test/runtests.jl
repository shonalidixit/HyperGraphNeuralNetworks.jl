using Random
using StatsBase
using LinearAlgebra
using Test
using Graphs
using GNNGraphs
import GNNGraphs: getn, getdata, normalize_graphdata, cat_features, shortsummary
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks
include("layers/DirectedHypergraphLayer.jl")
include("layers/DirectedHypergraphAttentionLayer.jl")
include("layers/DirectedHypergraphConvolutionLayer.jl") 
include("layers/AsymmetricDirectedHypergraphLayer.jl")
include("layers/ResidualDirectedHypergraphLayer.jl")
include("layers/GatedDirectedHypergraphLayer.jl")
include("layers/Spatial_hypergraph_layer_test.jl")
include("layers/UniGNNHypergraphLayer_test.jl")
include("layers/HypergraphAttentionLayer_test.jl")
# Necessary for MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Example undirected hypergraph
uh1 = Hypergraph{Float64, Int, String}(11,5)
#1st graph
uh1[1, 1] = 1.0
uh1[2, 1] = 2.0
uh1[4, 1] = 4.0
uh1[2, 2] = 3.0
uh1[5, 2] = 12.0
uh1[3, 2] = 0.0
uh1[4, 3] = 1.0
uh1[6, 3] = 4.0
#2nd graph
uh1[7, 4] = 3.5
uh1[10, 4] = 1.0
uh1[11, 4] = 4.0
uh1[8, 5] = 1.0
uh1[9, 5] = 5.0
uh1[10, 5] = 7.0

uid1 = [1,1,1,1,1,1,2,2,2,2,2]
uhedata1 = [10, 20, 30, 40, 50]

# Example directed hypergraph
dh1 = DirectedHypergraph{Float64, Int, String}(11,5)
dh1[1,1,1] = 1.0
dh1[1,2,1] = 2.0
dh1[2,4,1] = 4.0
dh1[1,2,2] = 3.0
dh1[1,5,2] = 12.0
dh1[2,3,2] = 0.0
dh1[1,4,3] = 1.0
dh1[2,6,3] = 4.0
#2nd graph
dh1[1,7,4] = 3.5
dh1[1,10,4] = 1.0
dh1[2,11,4] = 4.0
dh1[2,8,5] = 1.0
dh1[2,9,5] = 5.0
dh1[1,10,5] = 7.0
did1 = [1,1,1,1,1,1,2,2,2,2,2]
dhedata1 = [10, 20, 30, 40, 50]


@testset "HyperGraphNeuralNetworks HGNNHypergraph" begin

    # Direct Construction
    HGNN0 = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())
    @test size(HGNN0) == (11, 5)
    @test nhv(HGNN0) == 11
    @test nhe(HGNN0) == 5
    @test HGNN0.hypergraph_ids == uid1
    @test HGNN0.vdata == DataStore()
    @test HGNN0.hedata == DataStore() 
    @test HGNN0.hgdata == DataStore()

    # Test type equality
    @test HGNN0 == HGNNHypergraph{Float64, Dict{Int, Float64}}(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())

    # Construct using existing hypergraph
    HGNN1 = HGNNHypergraph(uh1; hypergraph_ids = uid1, hedata = uhedata1)
    @test size(HGNN1) == (11, 5)
    @test nhv(HGNN1) == 11
    @test nhe(HGNN1) == 5
    @test HGNN1.hypergraph_ids == uid1
    @test HGNN1.hedata == DataStore(e = uhedata1) 
    @test HGNN1.hgdata == DataStore(2)

    # Test type equality
    @test HGNN1 == HGNNHypergraph{Float64}(uh1; hypergraph_ids = uid1, hedata=uhedata1)
    @test HGNN1 == HGNNHypergraph{Float64, Dict{Int, Float64}}(uh1; hypergraph_ids = uid1, hedata=uhedata1)

    # Construct using matrix
    m = Matrix(uh1)
    @test m == uh1
    @test m == [1.0     nothing nothing nothing nothing
                2.0     3.0     nothing nothing nothing
                nothing 0.0     nothing nothing nothing
                4.0     nothing 1.0     nothing nothing
                nothing 12.0    nothing nothing nothing
                nothing nothing 4.0     nothing nothing
                nothing nothing nothing 3.5     nothing
                nothing nothing nothing nothing 1.0
                nothing nothing nothing nothing 5.0
                nothing nothing nothing 1.0     7.0
                nothing nothing nothing 4.0     nothing]
    HGNN2 = HGNNHypergraph(m; hypergraph_ids = uid1, hedata = uhedata1)
    @test HGNN2 == HGNN1

    # Test type equality
    @test HGNN2 == HGNNHypergraph{Float64}(m; hypergraph_ids = uid1, hedata = uhedata1)
    @test HGNN2 == HGNNHypergraph{Float64, Dict{Int, Float64}}(m; hypergraph_ids = uid1, hedata = uhedata1)

    # Construct with no hypergraph and num_nodes vertices
    HGNN3 = HGNNHypergraph(3)
    @test HGNN3.num_vertices == 3
    @test HGNN3.num_hyperedges == 0

    #construct with minimal information
    HGNN4 = HGNNHypergraph()
    @test HGNN4.num_vertices == 0

    #hasvertexmeta and hashyperedgemeta
    @test hasvertexmeta(HGNN1) == true 
    @test hashyperedgemeta(HGNN1) == true
    @test hasvertexmeta(HGNNHypergraph) == true
    @test hashyperedgemeta(HGNNHypergraph) == true
end

@testset "HyperGraphNeuralNetworks HGNNHypergraph modification" begin
    incident = [1.0     2.0
                1.0     nothing
                nothing 1.0
                nothing nothing]
    HGNN1 = HGNNHypergraph(incident)
    @test HGNN1.num_vertices == 4

    #add/remove single vertex or hyperedge
    features2 = DataStore(1)
    hyperedges2 = Dict(2 => 4.0) #connect the new vertex to hyperedge 2
    HGNN2 = add_vertex(HGNN1, features2; hyperedges = hyperedges2)
    @test HGNN2.num_vertices == 5
    @test HGNN2.v2he[5] == Dict(2 => 4.0)
    @test HGNN2 != HGNN1

    HGNN3 = remove_vertex(HGNN2, 5)
    @test HGNN3.num_vertices == 4
    @test HGNN1 == HGNN3

    features4 = DataStore(1)
    vertices4 = Dict(2 => 4.0, 4 => 5.0) #connect the new hyperedge to vertices 2 and 4
    HGNN4 = add_hyperedge(HGNN3, features4; vertices = vertices4)
    @test HGNN4.num_hyperedges == 3
    @test HGNN4.he2v[3] == Dict(2 => 4.0, 4 => 5.0)
    @test HGNN4 != HGNN3

    HGNN5 = remove_hyperedge(HGNN4, 3)
    @test HGNN5.num_hyperedges == 2
    @test HGNN5 == HGNN3

    h = Hypergraph{Float64, Int, String}(7,4)
    h[1, 1] = 1.0
    h[2, 1] = 1.0
    h[3, 1] = 1.0
    h[3, 2] = 1.0
    h[4, 2] = 1.0
    h[4, 3] = 1.0
    h[5, 3] = 1.0
    h[6, 3] = 1.0
    h[7, 4] = 1.0
    vdata6 = (a = [1,2,3,4,5,6,7],)
    hedata6 = (b = [1,2,3,4],)
    HGNN6 = HGNNHypergraph(h; vdata = vdata6, hedata = hedata6)

    #add/remove multiple vertices or hyperedges
    HGNN7 = remove_vertices(HGNN6, [2, 5, 6, 7])
    @test HGNN7.num_vertices == 3
    @test HGNN7.num_hyperedges == 4
    @test HGNN7.v2he == [Dict(1 => 1.0), 
                        Dict(1 => 1.0, 2 => 1.0),  
                        Dict(2 => 1.0, 3 => 1.0)]
    @test HGNN7.he2v == [Dict(1 => 1.0, 2 => 1.0)
                        Dict(2 => 1.0, 3 => 1.0)
                        Dict(3 => 1.0)
                        Dict{Int64, Float64}()]
    
    HGNN8 = remove_hyperedges(HGNN7, [2, 4])
    @test HGNN8.num_vertices == 3
    @test HGNN8.num_hyperedges == 2
    @test HGNN8.v2he == [Dict(1 => 1.0), Dict(1 => 1.0), Dict(2 => 1.0)]
    @test HGNN8.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(3 => 1.0)]
    
    features9 = DataStore(a = [[8], [9]])
    hyperedges9 = [Dict(1 => 2.0), Dict(2 => 3.0)] #connect the new vertex to hyperedges 1 and 2
    HGNN9 = add_vertices(HGNN8, 2, features9; hyperedges = hyperedges9)
    @test HGNN9.num_vertices == 5
    @test HGNN9.v2he[4] == Dict(1 => 2.0)
    @test HGNN9.v2he[5] == Dict(2 => 3.0)
    @test HGNN9.vdata == DataStore(a = [1,3,4,8,9])

    features10 = DataStore(b = [[5], [6]])
    vertices10 = [Dict(1 => 1.0, 4 => 1.0), Dict(3 => 2.0, 5 => 2.0)]
    HGNN10 = add_hyperedges(HGNN9, 2, features10; vertices = vertices10)
    @test HGNN10.num_hyperedges == 4
    @test HGNN10.he2v[3] == Dict(1 => 1.0, 4 => 1.0)
    @test HGNN10.he2v[4] == Dict(3 => 2.0, 5 => 2.0)
    @test HGNN10.hedata == DataStore(b = [1,3,5,6])

    #These functions are not implemented
    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.add_vertex!(HGNN1)
    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.remove_vertex!(HGNN1, 1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.add_hyperedge!(HGNN1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.remove_hyperedge!(HGNN1, 1)

end

@testset "HyperGraphNeuralNetworks HGNNHypergraph Base functions" begin
    # Base.zero
    zeroHGNN = zero(HGNNHypergraph)
    @test zeroHGNN.num_vertices == 0
    @test zeroHGNN.num_hyperedges == 0
    @test zeroHGNN.num_hypergraphs == 1

    h = Hypergraph{Float64, Int, String}(2, 1)
    h[1, 1] = 1.0
    h[2, 1] = 2.0
    vdata = (a = [[1,2],[3,4]], b = [1, -1])
    hedata = (b = [1],)
    hgdata = [3]
    HGNN = HGNNHypergraph(h; vdata = vdata, hedata = hedata, hgdata = hgdata)

    # Base.copy
    copyHGNN = copy(HGNN; deep = false)
    @test copyHGNN == HGNN
    @test copyHGNN.v2he === HGNN.v2he
    @test copyHGNN.he2v === HGNN.he2v
    deepcopyHGNN = copy(HGNN; deep = true)
    @test deepcopyHGNN !== HGNN
    @test deepcopyHGNN.v2he !== HGNN.v2he
    @test deepcopyHGNN.he2v !== HGNN.he2v

    # Base.show
    normalize_str(s::AbstractString) = replace(s, r"\s+" => " ") |> strip
    @test normalize_str(sprint(show, HGNN)) == normalize_str("HGNNHypergraph(2, 1, 1) with 
    vertex features: DataStore(2) with 2 elements:
        a = 2-element Vector{Vector{Int64}}
        b = 2-element Vector{Int64}, 
    hyperedge features: DataStore(1) with 1 element:
        b = 1-element Vector{Int64}, 
    hypergraph features: DataStore() with 1 element: 
    u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN; context=IOContext(stdout, :compact=>true))
        ) == normalize_str("HGNNHypergraph(2, 1, 1) with 
        vertex features: DataStore(2) with 2 elements:
            a = 2-element Vector{Vector{Int64}}
            b = 2-element Vector{Int64}, 
        hyperedge features: DataStore(1) with 1 element:
            b = 1-element Vector{Int64}, 
        hypergraph features: DataStore() with 1 element: 
            u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN)
        ) == normalize_str("HGNNHypergraph: 
        num_vertices: 2 
        num_hyperedges: 1 
        vdata (vertex data): a = 2-element Vector{Vector{Int64}} 
                            b = 2-element Vector{Int64} 
        hedata (hyperedge data): b = 1-element Vector{Int64}
        hgdata (hypergraph data): u = 1-element Vector{Int64}")

    # MLUtils.numobs
    #TODO: probably move this elsewhere
    @test numobs(HGNN) == HGNN.num_hypergraphs 

    # Base.hash
    newHGNN = add_vertex(HGNN, DataStore(a = [[1, 2]], b = [3]))
    @test newHGNN.vdata == DataStore(a = [[1,2],[3,4], [1,2]], b = [1, -1, 3])
    @test hash(HGNN) == hash(copyHGNN)
    @test hash(HGNN) != hash(newHGNN)

    # Base.getproperty
    @test getproperty(HGNN, :v2he) == HGNN.v2he
    @test_throws ArgumentError getproperty(HGNN, :b)
    @test getproperty(HGNN, :a) == vdata.a
    @test_throws ArgumentError getproperty(HGNN, :foo) 

end

@testset "HyperGraphNeuralNetworks HGNNDiHypergraph" begin
    #construct using exsiting directedhypergraph
    HGNN1 = HGNNDiHypergraph(dh1, hypergraph_ids = did1, hedata = dhedata1)
    @test size(HGNN1) == (11, 5)
    @test nhv(HGNN1) == 11
    @test nhe(HGNN1) == 5
    @test HGNN1.hypergraph_ids == did1
    @test HGNN1.hedata == DataStore(e = dhedata1) 
    @test HGNN1.hgdata == DataStore(2)

    #construct using matrix
    m = Matrix(dh1)
    @test m == dh1
    tailMatrix = getindex.(m, 1)
    headMatrix = getindex.(m, 2)
    @test tailMatrix == [1.0     nothing nothing nothing nothing 
                         2.0     3.0     nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing 1.0     nothing nothing
                         nothing 12.0    nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 3.5     nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 1.0     7.0
                         nothing nothing nothing nothing nothing]
    @test headMatrix == [nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing 0.0     nothing nothing nothing
                         4.0     nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing 4.0     nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing 1.0
                         nothing nothing nothing nothing 5.0
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 4.0     nothing]
    HGNN2 = HGNNDiHypergraph(tailMatrix, headMatrix; hypergraph_ids = did1, hedata = dhedata1)
    @test HGNN2 == HGNN1

    #construct with no hypergraph and num_nodes vertices
    HGNN3 = HGNNDiHypergraph(3)
    @test HGNN3.num_vertices == 3
    @test HGNN3.num_hyperedges == 0

    #construct with minimal information
    HGNN4 = HGNNDiHypergraph()
    @test HGNN4.num_vertices == 0
    
    #hasvertexmeta and hashyperedgemeta
    @test hasvertexmeta(HGNN1) == true 
    @test hashyperedgemeta(HGNN1) == true
    @test hasvertexmeta(HGNNDiHypergraph) == true
    @test hashyperedgemeta(HGNNDiHypergraph) == true

    # Base.zero
    zeroHGNN = zero(HGNNDiHypergraph)
    @test zeroHGNN.num_vertices == 0
    @test zeroHGNN.num_hyperedges == 0
    @test zeroHGNN.num_hypergraphs == 1
end

@testset "HGNNDiHypergraph modification functions" begin
    tailMatrix = [1.0     nothing
                  1.0     nothing
                  nothing nothing
                  nothing 1.0]
    headMatrix = [nothing nothing
                  nothing 1.0
                  1.0     1.0
                  nothing nothing]
    vdata1 = (a = [1, 2, 3, 4], b = [1, -1, 1, -1])
    hedata1 = (c = [2.0, 4.0],)
    HGNN1 = HGNNDiHypergraph(tailMatrix, headMatrix; vdata = vdata1, hedata = hedata1)
    
    #add_vertices, add_vertex, remove_vertex, remove_hyperedge
    @test HGNN1.num_vertices == 4
    features1 = DataStore(a = [[5], [6]], b = [[1], [-1]])
    hyperedges_tail1 = [Dict(2 => 2.0), Dict{Int64, Float64}()]
    hyperedges_head1 = [Dict{Int64, Float64}(), Dict(1=>3.0)]
    HGNN2 = add_vertices(HGNN1, 2, features1; hyperedges_tail = hyperedges_tail1,
                            hyperedges_head = hyperedges_head1)
    @test HGNN2.hg_tail.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(4 => 1.0, 5 => 2.0)]
    @test HGNN2.hg_head.he2v == [Dict(3 => 1.0, 6 => 3.0), Dict(2 => 1.0, 3 => 1.0)]
    @test HGNN2.vdata == DataStore(a = [1, 2, 3, 4, 5, 6], b = [1, -1, 1, -1, 1, -1])

    HGNN3 = remove_vertex(HGNN2, 5)
    @test HGNN3.num_vertices == 5
    @test HGNN3.num_hyperedges == 2
    @test HGNN3.hg_tail.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(4 => 1.0)]
    @test HGNN3.hg_head.he2v == [Dict(3 => 1.0, 5=> 3.0), Dict(2 => 1.0, 3 => 1.0)]

    features4 = DataStore(c = [[1.0], [2.0]])
    vertices_tail4 = [Dict(3 => 2.0), Dict(5 => 3.0)]
    vertices_head4 = [Dict(2 => 3.0), Dict(4 => 6.0)]
    HGNN4 = add_hyperedges(HGNN3, 2, features4; vertices_tail = vertices_tail4, 
                            vertices_head = vertices_head4)
    @test HGNN4.num_hyperedges == 4
    @test HGNN4.num_vertices == 5
    @test HGNN4.hg_tail.he2v[3] == Dict(3 => 2.0)
    @test HGNN4.hg_tail.he2v[4] == Dict(5 => 3.0)
    @test HGNN4.hg_head.he2v[3] == Dict(2 => 3.0)
    @test HGNN4.hg_head.he2v[4] == Dict(4 => 6.0)
    @test HGNN4.hedata == DataStore(c = [2.0, 4.0, 1.0, 2.0])
    
    HGNN5 = remove_hyperedge(HGNN4, 2)
    @test HGNN5.num_hyperedges == 3
    @test HGNN5.num_vertices == 5
    @test HGNN5.hg_tail.v2he == [Dict(1 => 1.0), Dict(1 => 1.0), Dict(2 => 2.0), 
                                Dict{Int64, Float64}(), Dict(2 => 3.0)]
    @test HGNN5.hg_head.v2he == [Dict{Int64, Float64}(), Dict(2 => 3.0), Dict(1 => 1.0), 
                                Dict(2 => 6.0), Dict(1 => 3.0)]

    h = DirectedHypergraph{Float64, Int, String}(7,4)
    h[1, 1, 1] = 1.0
    h[2, 2, 1] = 1.0
    h[2, 3, 1] = 1.0
    h[1, 3, 2] = 1.0
    h[2, 4, 2] = 1.0
    h[1, 4, 3] = 1.0
    h[1, 5, 3] = 1.0
    h[2, 6, 3] = 1.0
    h[1, 7, 4] = 1.0
    HGNN6 = HGNNDiHypergraph(h)

    #remove_hyperedges
    HGNN7 = remove_vertices(HGNN6, [2, 5, 6, 7])
    @test HGNN7.num_vertices == 3
    @test HGNN7.num_hyperedges == 4
    @test HGNN7.hg_tail.v2he == [Dict(1 => 1.0), 
                                Dict(2 => 1.0),
                                Dict(3 => 1.0)]
    @test HGNN7.hg_head.v2he == [Dict{Int64, Float64}(),
                                Dict(1 => 1.0),
                                Dict(2 => 1.0)]
    @test HGNN7.hg_tail.he2v == [Dict(1 => 1.0),
                                Dict(2 => 1.0),
                                Dict(3 => 1.0),
                                Dict{Int64, Float64}()]
    @test HGNN7.hg_head.he2v == [Dict(2 => 1.0),
                                Dict(3 => 1.0),
                                Dict{Int64, Float64}(),
                                Dict{Int64, Float64}()]

    HGNN8 = remove_hyperedges(HGNN7, [2, 4])
    @test HGNN8.num_vertices == 3
    @test HGNN8.num_hyperedges == 2
    @test HGNN8.hg_tail.v2he == [Dict(1 => 1.0),
                                Dict{Int64, Float64}(),
                                Dict(2 => 1.0)]
    @test HGNN8.hg_head.v2he == [Dict{Int64, Float64}(),
                                Dict(1 => 1.0),
                                Dict{Int64, Float64}()]
    @test HGNN8.hg_tail.he2v == [Dict(1 => 1.0),
                                Dict(3 => 1.0)]
    @test HGNN8.hg_head.he2v == [Dict(2 => 1.0),
                                Dict{Int64, Float64}()]
    
    #These functions are not implemented
    @test_throws "Not implemented! Number of vertices in HGNNDiHypergraph is fixed." SimpleHypergraphs.add_vertex!(HGNN1)
    @test_throws "Not implemented! Number of vertices in HGNNDiHypergraph is fixed." SimpleHypergraphs.remove_vertex!(HGNN1, 1)
    @test_throws "Not implemented! Number of hyperedges in HGNNDiHypergraph is fixed." SimpleHypergraphs.add_hyperedge!(HGNN1)
    @test_throws "Not implemented! Number of hyperedges in HGNNDiHypergraph is fixed." SimpleHypergraphs.remove_hyperedge!(HGNN1, 1)
end

@testset "HyperGraphNeuralNetworks HGNNDiHypergraph Base functions" begin
    h = DirectedHypergraph{Float64, Int, String}(2,1)
    h[1, 1, 1] = 1.0
    h[2, 2, 1] = 2.0
    vdata = (a = [[1,2],[3,4]], b = [1, -1])
    hedata = (b = [1],)
    hgdata = [3]
    HGNN = HGNNDiHypergraph(h; vdata = vdata, hedata = hedata, hgdata = hgdata)

    #base.show
    normalize_str(s::AbstractString) = replace(s, r"\s+" => " ") |> strip
    @test normalize_str(sprint(show, HGNN)) == normalize_str("
        HGNNDiHypergraph(2, 1, 1) with 
        vertex features: DataStore(2) with 2 elements:
            a = 2-element Vector{Vector{Int64}}
            b = 2-element Vector{Int64}, 
        hyperedge features: DataStore(1) with 1 element:
            b = 1-element Vector{Int64}, 
        hypergraph features: DataStore() with 1 element:
            u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN; context=IOContext(stdout, :compact=>true))
        ) == normalize_str("HGNNDiHypergraph(2, 1, 1) with 
                vertex features: DataStore(2) with 2 elements: 
                    a = 2-element Vector{Vector{Int64}} 
                    b = 2-element Vector{Int64}, 
                hyperedge features: DataStore(1) with 1 element: 
                    b = 1-element Vector{Int64}, 
                hypergraph features: DataStore() with 1 element: 
                    u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN)
        ) == normalize_str("HGNNDiHypergraph: num_vertices: 2 num_hyperedges: 1 
                            vdata (vertex data): 
                                a = 2-element Vector{Vector{Int64}} 
                                b = 2-element Vector{Int64} 
                            hedata (hyperedge data): 
                                b = 1-element Vector{Int64} 
                            hgdata (hypergraph data): 
                                u = 1-element Vector{Int64}")
    
    #base.copy
    copyHGNN = copy(HGNN; deep = false)
    @test copyHGNN == HGNN
    @test copyHGNN.hg_tail === HGNN.hg_tail
    @test copyHGNN.hg_head === HGNN.hg_head
    deepcopyHGNN = copy(HGNN; deep = true)
    @test deepcopyHGNN !== HGNN
    @test deepcopyHGNN.hg_tail !== HGNN.hg_tail
    @test deepcopyHGNN.hg_head !== HGNN.hg_head

    #MLUtils.numobs
    @test numobs(HGNN) == HGNN.num_hypergraphs 

    #Bese.hash
    newHGNN = add_vertex(HGNN, DataStore(a = [[1, 2]], b = [3]))
    @test newHGNN.vdata == DataStore(a = [[1,2],[3,4], [1,2]], b = [1, -1, 3])
    @test hash(HGNN) == hash(copyHGNN)
    @test hash(HGNN) != hash(newHGNN)

    #Base.getproperty
    @test getproperty(HGNN, :hg_tail) == HGNN.hg_tail
    @test_throws ArgumentError getproperty(HGNN, :b)
    @test getproperty(HGNN, :a) == vdata.a
    @test_throws ArgumentError getproperty(HGNN, :foo) 

end

@testset "HyperGraphNeuralNetworks random generation" begin
    # Erdos-Renyi random hypergraphs
    
    # Undirected
    Her_un = erdos_renyi_hypergraph(5, 5, HGNNHypergraph)
    @test nhv(Her_un) == 5
    @test nhe(Her_un) == 5
    @test all(length.(Her_un.v2he) .> 0)
    @test all(length.(Her_un.v2he) .<= 5)

    # With specified seed
    Her_un = erdos_renyi_hypergraph(5, 5, HGNNHypergraph; seed=1)
    @test Matrix(Her_un) == [
        nothing   nothing  1         1         1
        1         1         1          nothing  1
        nothing  1         1         1         1
        nothing  1         1          nothing  1
        nothing  1          nothing   nothing   nothing
    ]

    # Directed
    Her_di = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph)
    @test nhv(Her_un) == 5
    @test nhe(Her_un) == 5
    @test all(length.(Her_di.hg_tail.v2he) .> 0)
    @test all(length.(Her_di.hg_head.v2he) .> 0)
    @test all(length.(Her_di.hg_tail.v2he) .<= 5)
    @test all(length.(Her_di.hg_head.v2he) .<= 5)

    # With specified seed
    Her_di = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph; seed=42)
    @test Matrix(Her_di) == [
        (1, 1)              (nothing, 1)  (nothing, 1)        (1, 1)  (1, 1)
        (nothing, nothing)  (1, 1)        (nothing, nothing)  (1, 1)  (1, 1)
        (nothing, 1)        (1, 1)        (1, nothing)        (1, 1)  (1, nothing)
        (nothing, 1)        (nothing, 1)  (nothing, nothing)  (1, 1)  (1, 1)
        (1, 1)              (1, 1)        (1, 1)              (1, 1)  (1, 1)
    ]

    # With no self-loops
    DHr_nsl = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph; no_self_loops=true)
    for i in 1:5
        @test length(intersect(keys(DHr_nsl.hg_tail.v2he[i]), keys(DHr_nsl.hg_head.v2he[i]))) == 0
    end

    # Random k-uniform hypergraph

    # Undirected
    Hk = random_kuniform_hypergraph(5, 5, 3, HGNNHypergraph)
    @test nhv(Hk) == 5
    @test nhe(Hk) == 5
    @test all(length.(Hk.he2v) .== 3)

    # With specified seed
    Hk = random_kuniform_hypergraph(5, 5, 3, HGNNHypergraph; seed=42)
    @test Matrix(Hk) == [
        1         1         1         1          nothing
        nothing   nothing  1         1         1
        1         1          nothing   nothing  1
        1         1         1         1         1
        nothing   nothing   nothing   nothing   nothing
    ]

    # Directed
    DHk = random_kuniform_hypergraph(5, 5, 3, HGNNDiHypergraph)
    @test nhv(DHk) == 5
    @test nhe(DHk) == 5
    @test all(length.(DHk.hg_tail.he2v) .+ length.(DHk.hg_head.he2v) .== 3)

    # With specified seed
    DHk = random_kuniform_hypergraph(5, 5, 3, HGNNDiHypergraph; seed=42)
    @test Matrix(DHk) == [
        (nothing, nothing)  (1, nothing)        (nothing, nothing)  (nothing, 1)        (1, nothing)
        (1, nothing)        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)
        (1, nothing)        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)
        (nothing, 1)        (1, nothing)        (nothing, 1)        (nothing, nothing)  (nothing, nothing)
        (nothing, nothing)  (1, nothing)        (nothing, nothing)  (nothing, nothing)  (nothing, nothing)
    ]

    # Random d-regular hypergraph

    # Undirected
    Hd = random_dregular_hypergraph(5, 5, 3, HGNNHypergraph)
    @test nhv(Hd) == 5
    @test nhe(Hd) == 5
    @test all(length.(Hd.v2he) .== 3)

    # With specified seed
    Hd = random_dregular_hypergraph(5, 5, 3, HGNNHypergraph; seed=42)
    @test Matrix(Hd) == [
        1          nothing  1         1  nothing
        1          nothing  1         1  nothing
        1         1          nothing  1  nothing
        1         1          nothing  1  nothing
         nothing  1         1         1  nothing
    ]

    # Directed
    DHd = random_dregular_hypergraph(5, 5, 3, HGNNDiHypergraph)
    @test nhv(DHd) == 5
    @test nhe(DHd) == 5
    @test all(length.(DHd.hg_tail.v2he) .+ length.(DHd.hg_head.v2he) .== 3)

    # With specified seed
    DHd = random_dregular_hypergraph(5, 5, 3, HGNNDiHypergraph; seed=42)
    @test Matrix(DHd) == [
        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)        (nothing, nothing)
        (1, nothing)        (nothing, nothing)  (nothing, nothing)  (1, nothing)        (1, nothing)
        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)        (nothing, nothing)
        (nothing, 1)        (1, nothing)        (1, nothing)        (nothing, nothing)  (nothing, nothing)
        (1, nothing)        (nothing, 1)        (nothing, 1)        (nothing, nothing)  (nothing, nothing)
    ]

    # Random hypergraph with preferential attachment (undirected only, for now)

    H∂ = random_preferential_hypergraph(20, 0.5, HGNNHypergraph)
    @test nhv(H∂) == 20

    uh2 = Hypergraph{Bool}(5,5)
    uh2[1, 1] = true
    uh2[2, 1] = true
    uh2[4, 1] = true
    uh2[2, 2] = true
    uh2[5, 2] = true
    uh2[4, 3] = true
    uh2[2, 3] = true
    uh2[2, 4] = true
    uh2[4, 4] = true
    uh2[5, 4] = true
    uh2[4, 5] = true
    uh2[5, 5] = true

    # With specified seed
    H∂ = random_preferential_hypergraph(8, 0.5, HGNNHypergraph; seed=42, hg=uh2)
    @test Matrix(H∂) == [
        1          nothing   nothing   nothing   nothing   nothing   nothing  1          nothing   nothing
        1         1         1         1          nothing  1         1         1          nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing
        1          nothing  1         1         1         1         1         1         1          nothing
         nothing  1          nothing  1         1         1         1         1          nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing  1          nothing   nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing  1          nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing  1       
    ]
end

@testset "HyperGraphNeuralNetworks query" begin
    hgnn = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())
    dhgnn = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    # hyperedge_index
    @test hyperedge_index(hgnn) == [
        [1, 2, 4],
        [2, 3, 5],
        [4, 6],
        [7, 10, 11],
        [8, 9, 10],
    ]
    @test hyperedge_index(dhgnn) == (
        [[1, 2], [2, 5], [4], [7, 10], [10]],
        [[4], [3], [6], [11], [8, 9]]
    )
    
    # get_hyperedge_weights
    @test get_hyperedge_weights(hgnn) == [
        [1.0, 2.0, 4.0],
        [3.0, 0.0, 12.0],
        [1.0, 4.0],
        [3.5, 1.0, 4.0],
        [1.0, 5.0, 7.0]
    ]
    @test get_hyperedge_weights(hgnn, sum) == [7.0, 15.0, 5.0, 8.5, 13.0]

    dweights = (
        [[1.0, 2.0], [3.0, 12.0], [1.0], [3.5, 1.0], [7.0]],
        [[4.0], [0.0], [4.0], [4.0], [1.0, 5.0]]
    )

    @test get_hyperedge_weights(dhgnn) == dweights
    @test get_hyperedge_weights(dhgnn; side=:both) == dweights
    @test get_hyperedge_weights(dhgnn; side=:tail) == dweights[1]
    @test get_hyperedge_weights(dhgnn; side=:head) == dweights[2]
    @test get_hyperedge_weights(dhgnn, sum) == ([3.0, 15.0, 1.0, 4.5, 7.0], [4.0, 0.0, 4.0, 4.0, 6.0])

    # get_hyperedge_weight
    @test get_hyperedge_weight(hgnn, 2) == [3.0, 0.0, 12.0]
    @test get_hyperedge_weight(hgnn, 2, sum) == 15.0

    @test get_hyperedge_weight(dhgnn, 2) == ([3.0, 12.0], [0.0])
    @test get_hyperedge_weight(dhgnn, 2; side=:both) == ([3.0, 12.0], [0.0])
    @test get_hyperedge_weight(dhgnn, 2; side=:tail) == [3.0, 12.0]
    @test get_hyperedge_weight(dhgnn, 2; side=:head) == [0.0]
    @test get_hyperedge_weight(dhgnn, 2, sum) == (15.0, 0.0)

    # has_vertex
    @test has_vertex(hgnn, 10)
    @test !(has_vertex(hgnn, 12))

    @test has_vertex(dhgnn, 10)
    @test !(has_vertex(dhgnn, 25))

    # vertices
    @test vertices(hgnn) == 1:11
    @test vertices(dhgnn) == 1:11

    # degree
    @test degree(hgnn) == [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    @test degree(hgnn, 2) == 2
    @test degree(hgnn, [1,3,5]) == [1, 1, 1]

    @test degree(dhgnn) == [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    @test degree(dhgnn, 1) == 1
    @test degree(dhgnn, [1,2,3]) == [1, 2, 1]

    # indegree
    @test indegree(dhgnn) == [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    @test indegree(dhgnn, 5) == 0
    @test indegree(dhgnn, [1,2,4,8]) == [0, 0, 1, 1]

    # outdegree
    @test outdegree(dhgnn) == [1, 2, 0, 1, 1, 0, 1, 0, 0, 2, 0]
    @test outdegree(dhgnn, 2) == 2
    @test outdegree(dhgnn, [5, 10]) == [1, 2]

    # all_neighbors
    @test all_neighbors(hgnn) == [
        [2, 4],
        [1, 3, 4, 5],
        [2, 5],
        [1, 2, 6],
        [2, 3],
        [4],
        [10, 11],
        [9, 10],
        [8, 10],
        [7, 8, 9, 11],
        [7, 10]
    ]
    @test all_neighbors(hgnn, 2) == [1,3,4,5]

    @test all_neighbors(dhgnn) == [
        [4],
        [3, 4],
        [2, 5],
        [1, 2, 6],
        [3],
        [4],
        [11],
        [10],
        [10],
        [8, 9, 11],
        [7, 10]
    ]
    @test all_neighbors(dhgnn; same_side=true) == [
        [2, 4],
        [1, 3, 4, 5],
        [2, 5],
        [1, 2, 6],
        [2, 3],
        [4],
        [10, 11],
        [9, 10],
        [8, 10],
        [7, 8, 9, 11],
        [7, 10]
    ]

    @test all_neighbors(dhgnn, 2) == [3, 4]
    @test all_neighbors(dhgnn, 2; same_side=true) == [1, 3, 4, 5]

    # inneighbors
    @test inneighbors(dhgnn) == [
        [],
        [],
        [2, 5],
        [1, 2],
        [],
        [4],
        [],
        [10],
        [10],
        [],
        [7, 10]
    ]
    @test inneighbors(dhgnn; same_side=true) == [
        [],
        [],
        [2, 5],
        [1, 2],
        [],
        [4],
        [],
        [9, 10],
        [8, 10],
        [],
        [7, 10]
    ]

    @test inneighbors(dhgnn, 9) == [10]
    @test inneighbors(dhgnn, 9; same_side=true) == [8, 10]

    # outneighbors
    @test outneighbors(dhgnn) == [
        [4],
        [3, 4],
        [],
        [6],
        [3],
        [],
        [11],
        [],
        [],
        [8, 9, 11],
        []
    ]
    @test outneighbors(dhgnn; same_side=true) == [
        [2, 4],
        [1, 3, 4, 5],
        [],
        [6],
        [2, 3],
        [],
        [10, 11],
        [],
        [],
        [7, 8, 9, 11],
        []
    ]

    @test outneighbors(dhgnn, 2) == [3, 4]
    @test outneighbors(dhgnn, 2; same_side=true) == [1, 3, 4, 5]

    # hyperedge_neighbors
    @test hyperedge_neighbors(hgnn) == [[2, 3], [1], [1], [5], [4]]
    @test hyperedge_neighbors(hgnn, 4) == [5]

    # isolated_vertices
    @test length(isolated_vertices(hgnn)) == 0
    @test isolated_vertices(HGNNHypergraph(Hypergraph(5,0))) == [1,2,3,4,5]

    # incidence_matrix
    @test incidence_matrix(hgnn) == [
        1.0  0.0  0.0  0.0  0.0
        1.0  1.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        1.0  0.0  1.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  1.0  1.0
        0.0  0.0  0.0  1.0  0.0
    ]

    inc = incidence_matrix(dhgnn)
    @test inc[1] == [
        1.0  0.0  0.0  0.0  0.0
        1.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  1.0
        0.0  0.0  0.0  0.0  0.0
    ]
    @test inc[2] == [
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        1.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
    ]

    # complex_incidence_matrix
    @test complex_incidence_matrix(dhgnn) == [
        0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-1.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  1.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        1.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  1.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-1.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im  0.0-0.0im
    ]
    
    # vertex_weight_matrix
    @test vertex_weight_matrix(hgnn) == Diagonal([1.0, 5.0, 0.0, 5.0, 12.0, 4.0, 3.5, 1.0, 5.0, 8.0, 4.0])
    # Non-standard weighting function
    @test vertex_weight_matrix(hgnn; weighting_function=prod) == Diagonal(zeros(11))

    @test vertex_weight_matrix(dhgnn)[1] == Diagonal([1.0, 5.0, 0.0, 1.0, 12.0, 0.0, 3.5, 0.0, 0.0, 8.0, 0.0])
    @test vertex_weight_matrix(dhgnn; weighting_function=prod)[1] == Diagonal(zeros(11))

    @test vertex_weight_matrix(dhgnn)[2] == Diagonal([0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 1.0, 5.0, 0.0, 4.0])
    @test vertex_weight_matrix(dhgnn; weighting_function=prod)[2] == Diagonal(zeros(11))

    # hyperedge_weight_matrix
    @test hyperedge_weight_matrix(hgnn) == Diagonal([7.0, 15.0, 5.0, 8.5, 13.0])
    # Non-standard weighting function
    @test hyperedge_weight_matrix(hgnn; weighting_function=prod) == Diagonal(zeros(5))

    @test hyperedge_weight_matrix(dhgnn)[1] == Diagonal([3.0, 15.0, 1.0, 4.5, 7.0])
    @test hyperedge_weight_matrix(dhgnn; weighting_function=prod)[1] == Diagonal(zeros(5))

    @test hyperedge_weight_matrix(dhgnn)[2] == Diagonal([4.0, 0.0, 4.0, 4.0, 6.0])
    @test hyperedge_weight_matrix(dhgnn; weighting_function=prod)[2] == Diagonal(zeros(5))

    # vertex_degree_matrix
    @test vertex_degree_matrix(hgnn) == Diagonal([1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1])

    @test vertex_degree_matrix(dhgnn)[1] == Diagonal([1, 2, 0, 1, 1, 0, 1, 0, 0, 2, 0])
    @test vertex_degree_matrix(dhgnn)[2] == Diagonal([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

    # hyperedge_degree_matrix
    @test hyperedge_degree_matrix(hgnn) == Diagonal([3, 3, 2, 3, 3])

    @test hyperedge_degree_matrix(dhgnn)[1] == Diagonal([2, 2, 1, 2, 1])
    @test hyperedge_degree_matrix(dhgnn)[2] == Diagonal([1, 1, 1, 1, 2])

    # normalized_laplacian
    L = normalized_laplacian_matrix(hgnn)
    @test size(L) == (11,11)
    @test L[1,3] == 0.0
    @test isapprox(L, L'; rtol=1e-5)

    L = normalized_laplacian_matrix(dhgnn)
    @test size(L) == (11, 11)
    @test L[1,3] == 0.0-0.0im
    @test isapprox(L, L'; rtol=1e-5)

    # hypergraph_ids
    @test hypergraph_ids(hgnn) == uid1
    @test hypergraph_ids(dhgnn) == did1

    # Case with no hypergraph_ids
    hgnn2 = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 1, nothing, DataStore(), DataStore(), DataStore())
    @test hypergraph_ids(hgnn2) == ones(11)

    dhgnn2 = HGNNDiHypergraph(dh1; hypergraph_ids = nothing)
    @test hypergraph_ids(dhgnn2) == ones(11)

    # has_self_loops
    @test !(has_self_loops(hgnn))
    @test !(has_self_loops(dhgnn))

    # has_multi_hyperedges
    @test !(has_multi_hyperedges(hgnn))
    hg3 = Hypergraph{Bool}(2,2)
    hg3[:,:] .= true
    hgnn3 = HGNNHypergraph(hg3)
    @test has_multi_hyperedges(hgnn3)

    @test !(has_multi_hyperedges(dhgnn))
    dhg3 = DirectedHypergraph{Bool}(2, 3)
    dhg3.hg_tail[:,:] .= true
    dhg3.hg_head[:,:] .= true
    dhgnn3 = HGNNDiHypergraph(dhg3)
    @test has_multi_hyperedges(dhgnn3)

end

@testset "HyperGraphNeuralNetworks transforms" begin
    # TODO: directed hyperedges

    hgnn = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())
    
    dhgnn = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = nothing,
        hedata = nothing,
        hgdata = nothing
    )

    # add_selfloops
    hgnn0 = add_selfloops(hgnn)
    @test hgnn0.num_hyperedges == 16
    for i in 1:hgnn0.num_vertices
        @test Dict{Int, Float64}(i => 1.0) in hgnn0.he2v
    end

    hgnn1 = add_hyperedge(hgnn, DataStore(); vertices=Dict{Int, Float64}(1 => 1.0))
    hgnn2 = add_selfloops(hgnn1)
    @test hgnn2.num_hyperedges == 16
    hgnn2 = add_selfloops(hgnn1; add_repeated_hyperedge=true)
    @test hgnn2.num_hyperedges == 17

    dhgnn0 = add_selfloops(dhgnn)
    @test dhgnn0.num_hyperedges == 16
    he_verts = collect(zip(Set.(keys.(dhgnn0.hg_tail.he2v)), Set.(keys.(dhgnn0.hg_head.he2v))))
    for i in 1:dhgnn0.num_vertices
        @test (Set(i), Set(i)) in he_verts
    end

    dhgnn1 = add_hyperedge(
        dhgnn,
        DataStore();
        vertices_tail=Dict{Int, Float64}(1 => 1.0),
        vertices_head=Dict{Int, Float64}(1 => 1.0)
    )
    dhgnn2 = add_selfloops(dhgnn1)
    @test dhgnn2.num_hyperedges == 16
    dhgnn2 = add_selfloops(dhgnn1; add_repeated_hyperedge=true)
    @test dhgnn2.num_hyperedges == 17

    # remove_selfloops
    hgnn1 = remove_selfloops(hgnn2)
    @test hgnn1.num_hyperedges == 5

    dhgnn1 = remove_selfloops(dhgnn2)
    @test dhgnn1.num_hyperedges == 5

    # remove_multihyperedges
    hgnn2 = add_hyperedge(hgnn, DataStore(); vertices=Dict{Int, Float64}(1 => 1.0, 2 => 2.0, 4 => 4.0))
    @test remove_multihyperedges(hgnn2).num_hyperedges == 5

    dhgnn2 = add_hyperedge(
        dhgnn,
        DataStore();
        vertices_tail=Dict{Int, Float64}(1 => 1.0, 2 => 2.0),
        vertices_head=Dict{Int, Float64}(4 => 4.0)
    )
    @test remove_multihyperedges(dhgnn2).num_hyperedges == 5

    # to_undirected
    @test Set.(keys.(to_undirected(dhgnn).he2v)) == Set.(keys.(hgnn.he2v))

    # combine_hypergraphs / MLUtils.batch
    hg1 = HGNNHypergraph(
        [
            1.0     nothing
            nothing 2.0
            nothing 3.0
        ];
        hypergraph_ids=[1,2,2],
        vdata = rand(Float64, 5, 3),
        hedata = rand(Float64, 5, 2),
        hgdata = rand(Float64, 5, 2)
    )
    hg2 = HGNNHypergraph(
        [
            1.0     nothing     1.0
            nothing 2.0         3.0
        ];
        hypergraph_ids=[1,1],
        vdata = rand(Float64, 5, 2),
        hedata = rand(Float64, 5, 3),
        hgdata = rand(Float64, 5, 1)
    )

    hg_comb1 = combine_hypergraphs(hg1, hg2)
    @test hg_comb1.num_vertices == 5
    @test hg_comb1.num_hyperedges == 5
    @test hg_comb1.num_hypergraphs == 3
    @test hg_comb1.hypergraph_ids == [1, 2, 2, 3, 3]
    @test hg_comb1.vdata == cat_features(hg1.vdata, hg2.vdata)
    @test hg_comb1.hedata == cat_features(hg1.hedata, hg2.hedata)
    @test hg_comb1.hgdata == cat_features(hg1.hgdata, hg2.hgdata)

    hg_comb2 = combine_hypergraphs(hg1, hg1, hg2, hg2)
    @test hg_comb2.num_vertices == 10
    @test hg_comb2.num_hyperedges == 10
    @test hg_comb2.num_hypergraphs == 6
    @test hg_comb2.hypergraph_ids == [1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    @test hg_comb2.vdata == cat_features([hg1.vdata, hg1.vdata, hg2.vdata, hg2.vdata])
    @test hg_comb2.hedata == cat_features([hg1.hedata, hg1.hedata, hg2.hedata, hg2.hedata])
    @test hg_comb2.hgdata == cat_features([hg1.hgdata, hg1.hgdata, hg2.hgdata, hg2.hgdata])

    @test isnothing(combine_hypergraphs(HGNNHypergraph{Float64, Dict{Int,Float64}}[]))
    @test combine_hypergraphs([hg1]) == hg1
    @test combine_hypergraphs([hg1, hg1, hg2, hg2]) == hg_comb2

    @test combine_hypergraphs([hg1, hg2]) == batch([hg1, hg2])

    dhg1 = HGNNDiHypergraph(
        [
            1.0     nothing
            nothing 2.0
            nothing 3.0
        ],
        [
            nothing nothing
            2.0     nothing
            nothing 6.0
        ];
        hypergraph_ids=[1,2,2],
        vdata = rand(Float64, 5, 3),
        hedata = rand(Float64, 5, 2),
        hgdata = rand(Float64, 5, 2)
    )
    dhg2 = HGNNDiHypergraph(
        [
            1.0     nothing     nothing
            nothing 2.0         3.0
        ],
        [
            nothing nothing 1.0
            2.0     4.0     nothing
        ];
        hypergraph_ids=[1,1],
        vdata = rand(Float64, 5, 2),
        hedata = rand(Float64, 5, 3),
        hgdata = rand(Float64, 5, 1)
    )

    dhg_comb1 = combine_hypergraphs(dhg1, dhg2)
    @test dhg_comb1.num_vertices == 5
    @test dhg_comb1.num_hyperedges == 5
    @test dhg_comb1.num_hypergraphs == 3
    @test dhg_comb1.hypergraph_ids == [1, 2, 2, 3, 3]
    @test dhg_comb1.vdata == cat_features(dhg1.vdata, dhg2.vdata)
    @test dhg_comb1.hedata == cat_features(dhg1.hedata, dhg2.hedata)
    @test dhg_comb1.hgdata == cat_features(dhg1.hgdata, dhg2.hgdata)

    dhg_comb2 = combine_hypergraphs(dhg1, dhg1, dhg2, dhg2)
    @test dhg_comb2.num_vertices == 10
    @test dhg_comb2.num_hyperedges == 10
    @test dhg_comb2.num_hypergraphs == 6
    @test dhg_comb2.hypergraph_ids == [1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    @test dhg_comb2.vdata == cat_features([dhg1.vdata, dhg1.vdata, dhg2.vdata, dhg2.vdata])
    @test dhg_comb2.hedata == cat_features([dhg1.hedata, dhg1.hedata, dhg2.hedata, dhg2.hedata])
    @test dhg_comb2.hgdata == cat_features([dhg1.hgdata, dhg1.hgdata, dhg2.hgdata, dhg2.hgdata])

    @test isnothing(combine_hypergraphs(HGNNDiHypergraph{Float64, Dict{Int,Float64}}[]))
    @test combine_hypergraphs([dhg1]) == dhg1
    @test combine_hypergraphs([dhg1, dhg1, dhg2, dhg2]) == dhg_comb2

    @test combine_hypergraphs([dhg1, dhg2]) == batch([dhg1, dhg2])

    # get_hypergraph / MLUtils.unbatch
    hg1_1 = get_hypergraph(hg1, 1)
    @test hg1_1 == get_hypergraph(hg1, [1])
    @test get_hypergraph(hg1, [1,2]) == hg1
    @test hg1_1.num_vertices == 1
    @test hg1_1.num_hyperedges == 1
    @test hg1_1.num_hypergraphs == 1
    @test get_hypergraph(hg1, 2; map_vertices=true)[2] == [2, 3]
    @test unbatch(hg1) == [get_hypergraph(hg1, 1), get_hypergraph(hg1, 2)]

    dhg1_1 = get_hypergraph(dhg1, 1)
    @test dhg1_1 == get_hypergraph(dhg1, [1])
    @test get_hypergraph(dhg1, [1,2]) == dhg1
    @test dhg1_1.num_vertices == 1
    @test dhg1_1.num_hyperedges == 0
    @test dhg1_1.num_hypergraphs == 1
    @test get_hypergraph(dhg1, 2; map_vertices=true)[2] == [2, 3]
    @test unbatch(dhg1) == [get_hypergraph(dhg1, 1), get_hypergraph(dhg1, 2)]

    start_he_keys = Set.(keys.(hgnn.he2v))

    # uniform_negative_sample
    hgnn_u = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), UniformSample(); max_trials=100)
    @test hgnn_u.num_vertices == 11
    @test hgnn_u.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph
    for he in hgnn_u.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    # All hyperedges should be unique
    @test length(Set(Set.(keys.(hgnn_u.he2v)))) == 3

    start_he_keys = collect(
        zip(
            Set.(keys.(dhgnn.hg_tail.he2v)),
            Set.(keys.(dhgnn.hg_head.he2v))
        )
    )

    dhgnn_u = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), UniformSample(); max_trials=100)
    @test dhgnn_u.num_vertices == 11
    @test dhgnn_u.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph
    
    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_u.hg_tail.he2v, dhgnn_u.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3


    # sized_negative_sample
    hgnn_s = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), SizedSample(); max_trials=100)
    @test hgnn_s.num_vertices == 11
    @test hgnn_s.num_hyperedges == 3
    for he in hgnn_s.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_s.he2v)))) == 3

    dhgnn_s = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), SizedSample(); max_trials=100)
    @test dhgnn_s.num_vertices == 11
    @test dhgnn_s.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph

    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_s.hg_tail.he2v, dhgnn_s.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3


    # motif_negative_sample
    hgnn_m = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), MotifSample(); max_trials=100)
    @test hgnn_m.num_vertices == 11
    @test hgnn_m.num_hyperedges == 3
    for he in hgnn_m.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_m.he2v)))) == 3

    dhgnn_m = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), MotifSample(); max_trials=100)
    @test dhgnn_m.num_vertices == 11
    @test dhgnn_m.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph

    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_m.hg_tail.he2v, dhgnn_m.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3

    # clique_negative_sample
    hgnn_c = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), CliqueSample(); max_trials=100)
    @test hgnn_c.num_vertices == 11
    @test hgnn_c.num_hyperedges == 3
    for he in hgnn_c.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_c.he2v)))) == 3

    @test_throws "negative_sample not implemented for strategy of type CliqueSample" negative_sample_hyperedge(
        dhgnn,
        1,
        Xoshiro(42),
        CliqueSample()
    )

    # negative_sample_hyperedge
    struct NewSample <: AbstractNegativeSamplingStrategy end
    @test_throws "negative_sample not implemented for strategy of type NewSample" negative_sample_hyperedge(
        hgnn,
        1,
        Xoshiro(42),
        NewSample()
    )

    @test_throws "negative_sample not implemented for strategy of type NewSample" negative_sample_hyperedge(
        dhgnn,
        1,
        Xoshiro(42),
        NewSample()
    )

end

@testset "HyperGraphNeuralNetworks split vertices" begin    
    # Split vertices of undirected hypergraphs
    hgnn1 = HGNNHypergraph(
        uh1;
        hypergraph_ids = uid1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    vmasks = [
        BitVector((false, true, true, false, true, false, true, false, false, false, true)),
        BitVector((false, false, false, true, false, true, false, true, false, true, false)),
        BitVector((true, false, false, false, false, false, false, false, true, false, false))
    ]

    # Split vertices using masks
    hgnns = split_vertices(hgnn1, vmasks)
    @test length(hgnns) == 3
    @test hgnns[1].num_vertices == 5
    @test hgnns[1].num_hyperedges == 3
    @test hgnns[1].num_hypergraphs == 2
    @test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 2).x
    @test getobs(hgnns[1].hedata, 1).e == getobs(hgnn1.hedata, 1).e
    @test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
    @test hgnns[2].num_vertices == 4
    @test hgnns[2].num_hyperedges == 4
    @test hgnns[2].num_hypergraphs == 2
    @test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 4).x
    @test getobs(hgnns[2].hedata, 2).e == getobs(hgnn1.hedata, 3).e
    @test getobs(hgnns[2].hgdata, 2).u == getobs(hgnn1.hgdata, 2).u
    @test hgnns[3].num_vertices == 2
    @test hgnns[3].num_hyperedges == 2
    @test hgnns[3].num_hypergraphs == 2
    @test getobs(hgnns[3].vdata, 2).x == getobs(hgnn1.vdata, 9).x
    @test getobs(hgnns[3].hedata, 2).e == getobs(hgnn1.hedata, 5).e
    @test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u

    # Split vertices by train-val-test labeled masks
    hgnns_tvt = split_vertices(hgnn1, vmasks[1], vmasks[3]; val_mask=vmasks[2])
    @test hgnns_tvt.train == hgnns[1]
    @test hgnns_tvt.val == hgnns[2]
    @test hgnns_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_tvt_noval = split_vertices(hgnn1, vmasks[1], vmasks[3])
    @test hgnns_tvt_noval.train == hgnns[1]
    @test hgnns_tvt_noval.val === nothing
    @test hgnns_tvt_noval.test == hgnns[3]

    vinds = [
        [2, 3, 5, 7, 11],
        [4, 6, 8, 10],
        [1, 9]
    ]

    # Split vertices using vertex indices
    hgnns_ind = split_vertices(hgnn1, vinds)
    @test length(hgnns_ind) == 3
    @test hgnns_ind[1] == hgnns[1]
    @test hgnns_ind[2] == hgnns[2]
    @test hgnns_ind[3] == hgnns[3]

    # Split vertices by train-val-test labeled indices
    hgnns_ind_tvt = split_vertices(hgnn1, vinds[1], vinds[3]; val_inds=vinds[2])
    @test hgnns_ind_tvt.train == hgnns[1]
    @test hgnns_ind_tvt.val == hgnns[2]
    @test hgnns_ind_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_ind_tvt_noval = split_vertices(hgnn1, vinds[1], vinds[3])
    @test hgnns_ind_tvt_noval.train == hgnns[1]
    @test hgnns_ind_tvt_noval.val === nothing
    @test hgnns_ind_tvt_noval.test == hgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    hgnns_rand = random_split_vertices(hgnn1, [0.7, 0.1, 0.2], rng)
    @test length(hgnns_rand) == 3
    @test hgnns_rand[1].num_vertices == 8
    @test hgnns_rand[2].num_vertices == 1
    @test hgnns_rand[3].num_vertices == 2

    # Split vertices of directed hypergraphs
    dhgnn1 = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    vmasks = [
        BitVector((false, true, true, false, true, false, true, false, false, false, true)),
        BitVector((false, false, false, true, false, true, false, true, false, true, false)),
        BitVector((true, false, false, false, false, false, false, false, true, false, false))
    ]

    # Split vertices using masks
    dhgnns = split_vertices(dhgnn1, vmasks)
    @test length(dhgnns) == 3
    @test dhgnns[1].num_vertices == 5
    @test dhgnns[1].num_hyperedges == 3
    @test dhgnns[1].num_hypergraphs == 2
    @test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 2).x
    @test getobs(dhgnns[1].hedata, 1).e == getobs(dhgnn1.hedata, 1).e
    @test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
    @test dhgnns[2].num_vertices == 4
    @test dhgnns[2].num_hyperedges == 4
    @test dhgnns[2].num_hypergraphs == 2
    @test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 4).x
    @test getobs(dhgnns[2].hedata, 2).e == getobs(dhgnn1.hedata, 3).e
    @test getobs(dhgnns[2].hgdata, 2).u == getobs(dhgnn1.hgdata, 2).u
    @test dhgnns[3].num_vertices == 2
    @test dhgnns[3].num_hyperedges == 2
    @test dhgnns[3].num_hypergraphs == 2
    @test getobs(dhgnns[3].vdata, 2).x == getobs(dhgnn1.vdata, 9).x
    @test getobs(dhgnns[3].hedata, 2).e == getobs(dhgnn1.hedata, 5).e
    @test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u

    # Split vertices by train-val-test labeled masks
    dhgnns_tvt = split_vertices(dhgnn1, vmasks[1], vmasks[3]; val_mask=vmasks[2])
    @test dhgnns_tvt.train == dhgnns[1]
    @test dhgnns_tvt.val == dhgnns[2]
    @test dhgnns_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_tvt_noval = split_vertices(dhgnn1, vmasks[1], vmasks[3])
    @test dhgnns_tvt_noval.train == dhgnns[1]
    @test dhgnns_tvt_noval.val === nothing
    @test dhgnns_tvt_noval.test == dhgnns[3]

    vinds = [
        [2, 3, 5, 7, 11],
        [4, 6, 8, 10],
        [1, 9]
    ]

    # Split vertices using vertex indices
    dhgnns_ind = split_vertices(dhgnn1, vinds)
    @test length(dhgnns_ind) == 3
    @test dhgnns_ind[1] == dhgnns[1]
    @test dhgnns_ind[2] == dhgnns[2]
    @test dhgnns_ind[3] == dhgnns[3]

    # Split vertices by train-val-test labeled indices
    dhgnns_ind_tvt = split_vertices(dhgnn1, vinds[1], vinds[3]; val_inds=vinds[2])
    @test dhgnns_ind_tvt.train == dhgnns[1]
    @test dhgnns_ind_tvt.val == dhgnns[2]
    @test dhgnns_ind_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_ind_tvt_noval = split_vertices(dhgnn1, vinds[1], vinds[3])
    @test dhgnns_ind_tvt_noval.train == dhgnns[1]
    @test dhgnns_ind_tvt_noval.val === nothing
    @test dhgnns_ind_tvt_noval.test == dhgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    dhgnns_rand = random_split_vertices(dhgnn1, [0.7, 0.1, 0.2], rng)
    @test length(dhgnns_rand) == 3
    @test dhgnns_rand[1].num_vertices == 8
    @test dhgnns_rand[2].num_vertices == 1
    @test dhgnns_rand[3].num_vertices == 2
end

@testset "HyperGraphNeuralNetworks split hyperedges" begin    
    # Split hyperedges of undirected hypergraphs
    hgnn1 = HGNNHypergraph(
        uh1;
        hypergraph_ids = uid1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    hemasks = [
        BitVector((false, true, true, false, true)),
        BitVector((false, false, false, true, false)),
        BitVector((true, false, false, false, false))
    ]

    # Split hyperedges using masks
    hgnns = split_hyperedges(hgnn1, hemasks)
    @test length(hgnns) == 3
    @test hgnns[1].num_vertices == 8
    @test hgnns[1].num_hyperedges == 3
    @test hgnns[1].num_hypergraphs == 2
    @test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 2).x
    @test getobs(hgnns[1].hedata, 1).e == getobs(hgnn1.hedata, 2).e
    @test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
    @test hgnns[2].num_vertices == 3
    @test hgnns[2].num_hyperedges == 1
    @test hgnns[2].num_hypergraphs == 1
    @test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 7).x
    @test getobs(hgnns[2].hedata, 1).e == getobs(hgnn1.hedata, 4).e
    @test getobs(hgnns[2].hgdata, 1).u == getobs(hgnn1.hgdata, 2).u
    @test hgnns[3].num_vertices == 3
    @test hgnns[3].num_hyperedges == 1
    @test hgnns[3].num_hypergraphs == 1
    @test getobs(hgnns[3].vdata, 1).x == getobs(hgnn1.vdata, 1).x
    @test getobs(hgnns[3].hedata, 1).e == getobs(hgnn1.hedata, 1).e
    @test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u

    # Split hyperedges by train-val-test labeled masks
    hgnns_tvt = split_hyperedges(hgnn1, hemasks[1], hemasks[3]; val_mask=hemasks[2])
    @test hgnns_tvt.train == hgnns[1]
    @test hgnns_tvt.val == hgnns[2]
    @test hgnns_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_tvt_noval = split_hyperedges(hgnn1, hemasks[1], hemasks[3])
    @test hgnns_tvt_noval.train == hgnns[1]
    @test hgnns_tvt_noval.val === nothing
    @test hgnns_tvt_noval.test == hgnns[3]

    heinds = [
        [2, 3, 5],
        [4],
        [1]
    ]

    # Split hyperedges using hyperedge indices
    hgnns_ind = split_hyperedges(hgnn1, heinds)
    @test length(hgnns_ind) == 3
    @test hgnns_ind[1] == hgnns[1]
    @test hgnns_ind[2] == hgnns[2]
    @test hgnns_ind[3] == hgnns[3]

    # Split hyperedges by train-val-test labeled indices
    hgnns_ind_tvt = split_hyperedges(hgnn1, heinds[1], heinds[3]; val_inds=heinds[2])
    @test hgnns_ind_tvt.train == hgnns[1]
    @test hgnns_ind_tvt.val == hgnns[2]
    @test hgnns_ind_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_ind_tvt_noval = split_hyperedges(hgnn1, heinds[1], heinds[3])
    @test hgnns_ind_tvt_noval.train == hgnns[1]
    @test hgnns_ind_tvt_noval.val === nothing
    @test hgnns_ind_tvt_noval.test == hgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    hgnns_rand = random_split_hyperedges(hgnn1, [0.7, 0.3], rng)
    @test length(hgnns_rand) == 2
    @test hgnns_rand[1].num_hyperedges == 4
    @test hgnns_rand[2].num_hyperedges == 1

    # Split hyperedges of directed hypergraphs
    dhgnn1 = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    hemasks = [
        BitVector((false, true, true, false, true)),
        BitVector((false, false, false, true, false)),
        BitVector((true, false, false, false, false))
    ]

    # Split hyperedges using masks
    dhgnns = split_hyperedges(dhgnn1, hemasks)
    @test length(dhgnns) == 3
    @test dhgnns[1].num_vertices == 8
    @test dhgnns[1].num_hyperedges == 3
    @test dhgnns[1].num_hypergraphs == 2
    @test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 2).x
    @test getobs(dhgnns[1].hedata, 1).e == getobs(dhgnn1.hedata, 2).e
    @test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
    @test dhgnns[2].num_vertices == 3
    @test dhgnns[2].num_hyperedges == 1
    @test dhgnns[2].num_hypergraphs == 1
    @test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 7).x
    @test getobs(dhgnns[2].hedata, 1).e == getobs(dhgnn1.hedata, 4).e
    @test getobs(dhgnns[2].hgdata, 1).u == getobs(dhgnn1.hgdata, 2).u
    @test dhgnns[3].num_vertices == 3
    @test dhgnns[3].num_hyperedges == 1
    @test dhgnns[3].num_hypergraphs == 1
    @test getobs(dhgnns[3].vdata, 1).x == getobs(dhgnn1.vdata, 1).x
    @test getobs(dhgnns[3].hedata, 1).e == getobs(dhgnn1.hedata, 1).e
    @test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u

    # Split hyperedges by train-val-test labeled masks
    dhgnns_tvt = split_hyperedges(dhgnn1, hemasks[1], hemasks[3]; val_mask=hemasks[2])
    @test dhgnns_tvt.train == dhgnns[1]
    @test dhgnns_tvt.val == dhgnns[2]
    @test dhgnns_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_tvt_noval = split_hyperedges(dhgnn1, hemasks[1], hemasks[3])
    @test dhgnns_tvt_noval.train == dhgnns[1]
    @test dhgnns_tvt_noval.val === nothing
    @test dhgnns_tvt_noval.test == dhgnns[3]

    heinds = [
        [2, 3, 5],
        [4],
        [1]
    ]

    # Split hyperedges using hyperedge indices
    dhgnns_ind = split_hyperedges(dhgnn1, heinds)
    @test length(dhgnns_ind) == 3
    @test dhgnns_ind[1] == dhgnns[1]
    @test dhgnns_ind[2] == dhgnns[2]
    @test dhgnns_ind[3] == dhgnns[3]

    # Split hyperedges by train-val-test labeled indices
    dhgnns_ind_tvt = split_hyperedges(dhgnn1, heinds[1], heinds[3]; val_inds=heinds[2])
    @test dhgnns_ind_tvt.train == dhgnns[1]
    @test dhgnns_ind_tvt.val == dhgnns[2]
    @test dhgnns_ind_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_ind_tvt_noval = split_hyperedges(dhgnn1, heinds[1], heinds[3])
    @test dhgnns_ind_tvt_noval.train == dhgnns[1]
    @test dhgnns_ind_tvt_noval.val === nothing
    @test dhgnns_ind_tvt_noval.test == dhgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    dhgnns_rand = random_split_hyperedges(dhgnn1, [0.7, 0.3], rng)
    @test length(dhgnns_rand) == 2
    @test dhgnns_rand[1].num_hyperedges == 4
    @test dhgnns_rand[2].num_hyperedges == 1
end

@testset "HyperGraphNeuralNetworks split hypergraphs" begin
    uid2 = [1,1,1,1,2,2,3,3,3,3,3]

    # Split hypergraphs of undirected hypergraphs
    hgnn1 = HGNNHypergraph(
        uh1;
        hypergraph_ids = uid2,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 3)
    )

    hgmasks = [
        BitVector((true, false, false)),
        BitVector((false, true, false)),
        BitVector((false, false, true))
    ]

    # Split hypergraphs using masks
    hgnns = split_hypergraphs(hgnn1, hgmasks)
    @test length(hgnns) == 3
    @test hgnns[1].num_vertices == 4
    @test hgnns[1].num_hyperedges == 3
    @test hgnns[1].num_hypergraphs == 1
    @test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 1).x
    @test getobs(hgnns[1].hedata, 2).e == getobs(hgnn1.hedata, 2).e
    @test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
    @test hgnns[2].num_vertices == 2
    @test hgnns[2].num_hyperedges == 2
    @test hgnns[2].num_hypergraphs == 1
    @test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 5).x
    @test getobs(hgnns[2].hedata, 2).e == getobs(hgnn1.hedata, 3).e
    @test getobs(hgnns[2].hgdata, 1).u == getobs(hgnn1.hgdata, 2).u
    @test hgnns[3].num_vertices == 5
    @test hgnns[3].num_hyperedges == 2
    @test hgnns[3].num_hypergraphs == 1
    @test getobs(hgnns[3].vdata, 1).x == getobs(hgnn1.vdata, 7).x
    @test getobs(hgnns[3].hedata, 2).e == getobs(hgnn1.hedata, 5).e
    @test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 3).u

    # Split hypergraphs by train-val-test labeled masks
    hgnns_tvt = split_hypergraphs(hgnn1, hgmasks[1], hgmasks[3]; val_mask=hgmasks[2])
    @test hgnns_tvt.train == hgnns[1]
    @test hgnns_tvt.val == hgnns[2]
    @test hgnns_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_tvt_noval = split_hypergraphs(hgnn1, hgmasks[1], hgmasks[3])
    @test hgnns_tvt_noval.train == hgnns[1]
    @test hgnns_tvt_noval.val === nothing
    @test hgnns_tvt_noval.test == hgnns[3]

    hginds = [[1], [2], [3]]

    # Split hypergraphs using vertex indices
    hgnns_ind = split_hypergraphs(hgnn1, hginds)
    @test length(hgnns_ind) == 3
    @test hgnns_ind[1] == hgnns[1]
    @test hgnns_ind[2] == hgnns[2]
    @test hgnns_ind[3] == hgnns[3]

    # Split hypergraphs by train-val-test labeled indices
    hgnns_ind_tvt = split_hypergraphs(hgnn1, hginds[1], hginds[3]; val_inds=hginds[2])
    @test hgnns_ind_tvt.train == hgnns[1]
    @test hgnns_ind_tvt.val == hgnns[2]
    @test hgnns_ind_tvt.test == hgnns[3]

    # Split without validation set
    hgnns_ind_tvt_noval = split_hypergraphs(hgnn1, hginds[1], hginds[3])
    @test hgnns_ind_tvt_noval.train == hgnns[1]
    @test hgnns_ind_tvt_noval.val === nothing
    @test hgnns_ind_tvt_noval.test == hgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    hgnns_rand = random_split_hypergraphs(hgnn1, [0.34, 0.33, 0.33], rng)
    @test length(hgnns_rand) == 3
    @test hgnns_rand[1].num_hypergraphs == 1
    @test hgnns_rand[2].num_hypergraphs == 1
    @test hgnns_rand[3].num_hypergraphs == 1


    # Split hypergraphs of directed hypergraphs
    dhgnn1 = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = uid2,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 3)
    )

    hgmasks = [
        BitVector((true, false, false)),
        BitVector((false, true, false)),
        BitVector((false, false, true))
    ]

    # Split hypergraphs using masks
    dhgnns = split_hypergraphs(dhgnn1, hgmasks)
    @test length(dhgnns) == 3
    @test dhgnns[1].num_vertices == 4
    @test dhgnns[1].num_hyperedges == 3
    @test dhgnns[1].num_hypergraphs == 1
    @test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 1).x
    @test getobs(dhgnns[1].hedata, 2).e == getobs(dhgnn1.hedata, 2).e
    @test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
    @test dhgnns[2].num_vertices == 2
    @test dhgnns[2].num_hyperedges == 2
    @test dhgnns[2].num_hypergraphs == 1
    @test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 5).x
    @test getobs(dhgnns[2].hedata, 2).e == getobs(dhgnn1.hedata, 3).e
    @test getobs(dhgnns[2].hgdata, 1).u == getobs(dhgnn1.hgdata, 2).u
    @test dhgnns[3].num_vertices == 5
    @test dhgnns[3].num_hyperedges == 2
    @test dhgnns[3].num_hypergraphs == 1
    @test getobs(dhgnns[3].vdata, 1).x == getobs(dhgnn1.vdata, 7).x
    @test getobs(dhgnns[3].hedata, 2).e == getobs(dhgnn1.hedata, 5).e
    @test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 3).u

    # Split hypergraphs by train-val-test labeled masks
    dhgnns_tvt = split_hypergraphs(dhgnn1, hgmasks[1], hgmasks[3]; val_mask=hgmasks[2])
    @test dhgnns_tvt.train == dhgnns[1]
    @test dhgnns_tvt.val == dhgnns[2]
    @test dhgnns_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_tvt_noval = split_hypergraphs(dhgnn1, hgmasks[1], hgmasks[3])
    @test dhgnns_tvt_noval.train == dhgnns[1]
    @test dhgnns_tvt_noval.val === nothing
    @test dhgnns_tvt_noval.test == dhgnns[3]

    hginds = [[1], [2], [3]]

    # Split hypergraphs using vertex indices
    dhgnns_ind = split_hypergraphs(dhgnn1, hginds)
    @test length(dhgnns_ind) == 3
    @test dhgnns_ind[1] == dhgnns[1]
    @test dhgnns_ind[2] == dhgnns[2]
    @test dhgnns_ind[3] == dhgnns[3]

    # Split hypergraphs by train-val-test labeled indices
    dhgnns_ind_tvt = split_hypergraphs(dhgnn1, hginds[1], hginds[3]; val_inds=hginds[2])
    @test dhgnns_ind_tvt.train == dhgnns[1]
    @test dhgnns_ind_tvt.val == dhgnns[2]
    @test dhgnns_ind_tvt.test == dhgnns[3]

    # Split without validation set
    dhgnns_ind_tvt_noval = split_hypergraphs(dhgnn1, hginds[1], hginds[3])
    @test dhgnns_ind_tvt_noval.train == dhgnns[1]
    @test dhgnns_ind_tvt_noval.val === nothing
    @test dhgnns_ind_tvt_noval.test == dhgnns[3]

    # "Random" split
    rng = Xoshiro(42)
    dhgnns_rand = random_split_hypergraphs(dhgnn1, [0.34, 0.33, 0.33], rng)
    @test length(dhgnns_rand) == 3
    @test dhgnns_rand[1].num_hypergraphs == 1
    @test dhgnns_rand[2].num_hypergraphs == 1
    @test dhgnns_rand[3].num_hypergraphs == 1
end