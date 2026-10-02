module HyperGraphNeuralNetworks

using Random
using DataStructures: counter
using StatsBase: FrequencyWeights, sample
using LinearAlgebra
using InvertedIndices
using Graphs
using GNNGraphs
import GNNGraphs: getn, getdata, normalize_graphdata, cat_features, shortsummary
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs


include("core/abstracttypes.jl")
include("core/hypergraphs.jl")
include("layers/DirectedHypergraphLayer.jl")
include("layers/DirectedHypergraphAttentionLayer.jl")
include("layers/DirectedHypergraphConvolutionLayer.jl")
include("layers/AsymmetricDirectedHypergraphLayer.jl")
include("layers/ResidualDirectedHypergraphLayer.jl")
include("layers/GatedDirectedHypergraphLayer.jl")
include("layers/SpatialHypergraphLayer.jl")
include("layers/UniGNNHypergraphLayer.jl")
include("layers/HypergraphAttentionLayer.jl")

export AbstractHGNNHypergraph, AbstractHGNNDiHypergraph
export HGNNHypergraph, HGNNDiHypergraph
export add_vertex, add_vertices, remove_vertex, remove_vertices
export add_hyperedge, add_hyperedges, remove_hyperedge, remove_hyperedges
export DirectedHypergraphLayer
export DirectedHypergraphAttentionLayer
export DirectedHypergraphConvolutionLayer
export AsymmetricDirectedHypergraphLayer
export ResidualDirectedHypergraphLayer
export GatedDirectedHypergraphLayer
export SpatialHypergraphLayer
export UniGNNHypergraphLayer
export HypergraphAttentionLayer

include("core/generate.jl")

export erdos_renyi_hypergraph, random_kuniform_hypergraph, random_dregular_hypergraph, random_preferential_hypergraph

include("core/query.jl")

export hyperedge_index, get_hyperedge_weight, get_hyperedge_weights
export has_vertex, vertices, isolated_vertices
export hyperedge_neighbors
export hyperedge_weight_matrix, hyperedge_degree_matrix, vertex_weight_matrix, vertex_degree_matrix
export complex_incidence_matrix, normalized_laplacian_matrix
export hypergraph_ids, has_multi_hyperedges

include("core/transform.jl")

export add_selfloops, remove_selfloops, remove_multihyperedges
export rewire_hyperedges, to_undirected
export combine_hypergraphs, get_hypergraph
export AbstractNegativeSamplingStrategy, UniformSample, SizedSample, MotifSample, CliqueSample, negative_sample_hyperedge

include("core/split.jl")

export split_vertices, split_hyperedges, split_hypergraphs
export random_split_vertices, random_split_hyperedges, random_split_hypergraphs

include("core/utils.jl")

export check_num_vertices, check_num_hyperedges
export normalize_graphdata


end