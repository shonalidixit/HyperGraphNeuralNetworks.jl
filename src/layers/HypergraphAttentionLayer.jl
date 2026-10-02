using Lux
using Random

"""
    HypergraphAttentionLayer(
        input_dim,
        output_dim;
        activation = tanh,
        use_bias = true,
        negative_slope = 0.2f0,
        init_weight = Lux.glorot_uniform,
        init_attention = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

Attention-based message-passing layer for undirected hypergraphs.

The layer performs two stages of attention-based propagation:

    vertices -> hyperedges -> vertices

First, each hyperedge computes an attention-weighted representation of its
incident vertices. Second, each vertex computes an attention-weighted
representation of its incident hyperedges.

Unlike mean aggregation, attention allows the layer to learn that different
vertex-hyperedge relationships can contribute different amounts to the
message-passing process.

# Arguments

- `input_dim::Int`: Number of features associated with each input vertex.
- `output_dim::Int`: Number of features in each output vertex representation.

# Keyword Arguments

- `activation = tanh`: Activation applied to the final vertex representations.
- `use_bias::Bool = true`: Whether to include a trainable output bias.
- `negative_slope = 0.2f0`: Negative slope used by the LeakyReLU attention
  scoring function.
- `init_weight = Lux.glorot_uniform`: Initializer for feature transformations.
- `init_attention = Lux.glorot_uniform`: Initializer for attention vectors.
- `init_bias = Lux.zeros32`: Initializer for the output bias.

# Input

The layer expects:

    (X_vertex, incidence_matrix)

where:

- `X_vertex` has shape `n_vertices x input_dim`.
- `incidence_matrix` has shape `n_vertices x n_hyperedges`.
- Any nonzero incidence value represents membership.

# Output

The layer returns a named tuple containing:

- `updated_vertices`: matrix of shape `n_vertices x output_dim`.
- `hyperedge_messages`: matrix of shape `n_hyperedges x output_dim`.
- `vertex_to_hyperedge_attention`: attention matrix with shape
  `n_vertices x n_hyperedges`.
- `hyperedge_to_vertex_attention`: attention matrix with shape
  `n_vertices x n_hyperedges`.

Attention values are zero for vertex-hyperedge pairs that are not incident.

The layer is stateless, so the Lux state is returned unchanged.

# Example

    rng = Random.default_rng()

    layer = HypergraphAttentionLayer(
        4,
        8,
    )

    ps, st = Lux.setup(
        rng,
        layer,
    )

    X_vertex = rand(
        Float32,
        5,
        4,
    )

    incidence_matrix = Float32[
        1  0;
        1  1;
        0  1;
        0  1;
        1  0
    ]

    output, st = layer(
        (
            X_vertex,
            incidence_matrix,
        ),
        ps,
        st,
    )
"""
struct HypergraphAttentionLayer{
    F,
    T,
    IW,
    IA,
    IB,
} <: Lux.AbstractLuxLayer
    input_dim::Int
    output_dim::Int
    activation::F
    use_bias::Bool
    negative_slope::T
    init_weight::IW
    init_attention::IA
    init_bias::IB
end


"""
    HypergraphAttentionLayer(
        input_dim,
        output_dim;
        kwargs...,
    )

Construct a `HypergraphAttentionLayer`.

Both feature dimensions must be positive and `negative_slope` must be
non-negative.
"""
function HypergraphAttentionLayer(
    input_dim::Int,
    output_dim::Int;
    activation = tanh,
    use_bias::Bool = true,
    negative_slope = 0.2f0,
    init_weight = Lux.glorot_uniform,
    init_attention = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    input_dim > 0 ||
        throw(
            ArgumentError(
                "`input_dim` must be positive.",
            ),
        )

    output_dim > 0 ||
        throw(
            ArgumentError(
                "`output_dim` must be positive.",
            ),
        )

    negative_slope >= 0 ||
        throw(
            ArgumentError(
                "`negative_slope` must be non-negative.",
            ),
        )

    return HypergraphAttentionLayer(
        input_dim,
        output_dim,
        activation,
        use_bias,
        negative_slope,
        init_weight,
        init_attention,
        init_bias,
    )
end


"""
    _hypergraph_attention_row_major_weight(
        initializer,
        rng,
        input_dim,
        output_dim,
    )

Initialize an `input_dim x output_dim` transformation matrix.

Lux initializers use output-by-input orientation, so the initialized matrix
is transposed for the row-major feature convention used by this package.
"""
function _hypergraph_attention_row_major_weight(
    initializer,
    rng::AbstractRNG,
    input_dim::Int,
    output_dim::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dim,
            input_dim,
        ),
    )
end


"""
    _hypergraph_attention_vector(
        initializer,
        rng,
        output_dim,
    )

Initialize a trainable attention vector with length `output_dim`.
"""
function _hypergraph_attention_vector(
    initializer,
    rng::AbstractRNG,
    output_dim::Int,
)
    return vec(
        initializer(
            rng,
            output_dim,
            1,
        ),
    )
end


"""
    _hypergraph_attention_bias(
        initializer,
        rng,
        output_dim,
    )

Initialize a `1 x output_dim` output bias.
"""
function _hypergraph_attention_bias(
    initializer,
    rng::AbstractRNG,
    output_dim::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dim,
            1,
        ),
    )
end


"""
    Lux.initialparameters(
        rng,
        layer::HypergraphAttentionLayer,
    )

Initialize the trainable parameters of the attention layer.

The parameters consist of:

- `W_vertex`: input vertex feature transformation.
- `W_output`: final vertex transformation.
- `a_vertex`: attention vector for vertex contributions.
- `a_hyperedge`: attention vector for hyperedge contributions.
- `b`: optional output bias.
"""
function Lux.initialparameters(
    rng::AbstractRNG,
    layer::HypergraphAttentionLayer,
)
    W_vertex =
        _hypergraph_attention_row_major_weight(
            layer.init_weight,
            rng,
            layer.input_dim,
            layer.output_dim,
        )

    W_output =
        _hypergraph_attention_row_major_weight(
            layer.init_weight,
            rng,
            layer.output_dim,
            layer.output_dim,
        )

    a_vertex =
        _hypergraph_attention_vector(
            layer.init_attention,
            rng,
            layer.output_dim,
        )

    a_hyperedge =
        _hypergraph_attention_vector(
            layer.init_attention,
            rng,
            layer.output_dim,
        )

    if layer.use_bias
        b =
            _hypergraph_attention_bias(
                layer.init_bias,
                rng,
                layer.output_dim,
            )

        return (
            W_vertex = W_vertex,
            W_output = W_output,
            a_vertex = a_vertex,
            a_hyperedge = a_hyperedge,
            b = b,
        )
    end

    return (
        W_vertex = W_vertex,
        W_output = W_output,
        a_vertex = a_vertex,
        a_hyperedge = a_hyperedge,
    )
end


"""
    Lux.parameterlength(
        layer::HypergraphAttentionLayer,
    )

Return the number of trainable scalar parameters in the layer.
"""
function Lux.parameterlength(
    layer::HypergraphAttentionLayer,
)
    vertex_transform =
        layer.input_dim *
        layer.output_dim

    output_transform =
        layer.output_dim *
        layer.output_dim

    attention_parameters =
        2 *
        layer.output_dim

    bias_parameters =
        if layer.use_bias
            layer.output_dim
        else
            0
        end

    return vertex_transform +
           output_transform +
           attention_parameters +
           bias_parameters
end


"""
    Lux.initialstates(
        rng,
        layer::HypergraphAttentionLayer,
    )

Return the Lux state associated with the layer.

The attention layer is stateless.
"""
function Lux.initialstates(
    ::AbstractRNG,
    ::HypergraphAttentionLayer,
)
    return NamedTuple()
end


"""
    Lux.statelength(
        layer::HypergraphAttentionLayer,
    )

Return the number of state variables.

The layer is stateless, so the result is always zero.
"""
function Lux.statelength(
    ::HypergraphAttentionLayer,
)
    return 0
end


"""
    _hypergraph_attention_membership(
        incidence_matrix,
    )

Convert the incidence matrix to binary membership.

Every nonzero value represents membership between a vertex and a hyperedge.
"""
function _hypergraph_attention_membership(
    incidence_matrix::AbstractMatrix,
)
    return .!iszero.(
        incidence_matrix,
    )
end


"""
    _hypergraph_attention_leaky_relu(
        value,
        negative_slope,
    )

Apply the LeakyReLU function used for attention logits.
"""
function _hypergraph_attention_leaky_relu(
    value,
    negative_slope,
)
    if value >= zero(value)
        return value
    end

    return negative_slope *
           value
end


"""
    _hypergraph_attention_masked_softmax(
        scores,
        mask,
    )

Compute a numerically stable softmax over the entries selected by `mask`.

Entries outside the mask receive zero probability. If the mask contains no
active entries, an all-zero vector is returned.
"""
function _hypergraph_attention_masked_softmax(
    scores::AbstractVector,
    mask::AbstractVector{Bool},
)
    length(scores) == length(mask) ||
        throw(
            DimensionMismatch(
                "Scores and mask must have the same length.",
            ),
        )

    result =
        zeros(
            eltype(scores),
            length(scores),
        )

    active_indices =
        findall(mask)

    isempty(active_indices) &&
        return result

    active_scores =
        scores[
            active_indices
        ]

    maximum_score =
        maximum(
            active_scores,
        )

    exponentials =
        exp.(
            active_scores .-
            maximum_score
        )

    denominator =
        sum(
            exponentials,
        )

    result[
        active_indices
    ] .= exponentials ./
         denominator

    return result
end


"""
    _hypergraph_vertex_to_hyperedge_attention(
        transformed_vertices,
        incidence_matrix,
        attention_vector,
        negative_slope,
    )

Aggregate transformed vertex representations into hyperedges using learned
attention weights.

Attention is normalized independently inside every hyperedge. Consequently,
the attention coefficients of the incident vertices of each non-empty
hyperedge sum to one.

# Returns

A tuple containing:

- the hyperedge representations;
- the vertex-to-hyperedge attention matrix.
"""
function _hypergraph_vertex_to_hyperedge_attention(
    transformed_vertices::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
    attention_vector::AbstractVector,
    negative_slope,
)
    n_vertices =
        size(
            transformed_vertices,
            1,
        )

    n_hyperedges =
        size(
            incidence_matrix,
            2,
        )

    feature_dim =
        size(
            transformed_vertices,
            2,
        )

    size(
        incidence_matrix,
        1,
    ) == n_vertices ||
        throw(
            DimensionMismatch(
                "The number of incidence-matrix rows must match " *
                "the number of transformed vertices.",
            ),
        )

    length(attention_vector) == feature_dim ||
        throw(
            DimensionMismatch(
                "Attention vector length must match the transformed " *
                "feature dimension.",
            ),
        )

    membership =
        _hypergraph_attention_membership(
            incidence_matrix,
        )

    vertex_scores =
        transformed_vertices *
        attention_vector

    vertex_scores =
        _hypergraph_attention_leaky_relu.(
            vertex_scores,
            Ref(negative_slope),
        )

    attention =
        zeros(
            eltype(transformed_vertices),
            n_vertices,
            n_hyperedges,
        )

    hyperedge_messages =
        zeros(
            eltype(transformed_vertices),
            n_hyperedges,
            feature_dim,
        )

    for hyperedge_index in 1:n_hyperedges
        mask =
            vec(
                membership[
                    :,
                    hyperedge_index,
                ],
            )

        coefficients =
            _hypergraph_attention_masked_softmax(
                vertex_scores,
                mask,
            )

        attention[
            :,
            hyperedge_index,
        ] .= coefficients

        hyperedge_messages[
            hyperedge_index,
            :,
        ] .= vec(
            sum(
                transformed_vertices .*
                reshape(
                    coefficients,
                    :,
                    1,
                );
                dims = 1,
            ),
        )
    end

    return (
        hyperedge_messages,
        attention,
    )
end


"""
    _hypergraph_hyperedge_to_vertex_attention(
        hyperedge_messages,
        incidence_matrix,
        attention_vector,
        negative_slope,
    )

Propagate hyperedge representations back to vertices using learned attention.

For each vertex, attention is normalized across the hyperedges incident to
that vertex. Isolated vertices receive an all-zero propagated representation.

# Returns

A tuple containing:

- the propagated vertex representations;
- the hyperedge-to-vertex attention matrix.
"""
function _hypergraph_hyperedge_to_vertex_attention(
    hyperedge_messages::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
    attention_vector::AbstractVector,
    negative_slope,
)
    n_vertices =
        size(
            incidence_matrix,
            1,
        )

    n_hyperedges =
        size(
            incidence_matrix,
            2,
        )

    feature_dim =
        size(
            hyperedge_messages,
            2,
        )

    size(
        hyperedge_messages,
        1,
    ) == n_hyperedges ||
        throw(
            DimensionMismatch(
                "The number of hyperedge representations must match " *
                "the number of incidence-matrix columns.",
            ),
        )

    length(attention_vector) == feature_dim ||
        throw(
            DimensionMismatch(
                "Attention vector length must match the hyperedge " *
                "feature dimension.",
            ),
        )

    membership =
        _hypergraph_attention_membership(
            incidence_matrix,
        )

    hyperedge_scores =
        hyperedge_messages *
        attention_vector

    hyperedge_scores =
        _hypergraph_attention_leaky_relu.(
            hyperedge_scores,
            Ref(negative_slope),
        )

    attention =
        zeros(
            eltype(hyperedge_messages),
            n_vertices,
            n_hyperedges,
        )

    propagated_vertices =
        zeros(
            eltype(hyperedge_messages),
            n_vertices,
            feature_dim,
        )

    for vertex_index in 1:n_vertices
        mask =
            vec(
                membership[
                    vertex_index,
                    :,
                ],
            )

        coefficients =
            _hypergraph_attention_masked_softmax(
                hyperedge_scores,
                mask,
            )

        attention[
            vertex_index,
            :,
        ] .= coefficients

        propagated_vertices[
            vertex_index,
            :,
        ] .= vec(
            sum(
                hyperedge_messages .*
                reshape(
                    coefficients,
                    :,
                    1,
                );
                dims = 1,
            ),
        )
    end

    return (
        propagated_vertices,
        attention,
    )
end


"""
    _validate_hypergraph_attention_inputs(
        layer,
        X_vertex,
        incidence_matrix,
    )

Validate dimensions before attention-based message passing.

The vertex feature dimension must equal `layer.input_dim`, and the number of
incidence-matrix rows must equal the number of vertices.
"""
function _validate_hypergraph_attention_inputs(
    layer::HypergraphAttentionLayer,
    X_vertex::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
)
    size(
        X_vertex,
        2,
    ) == layer.input_dim ||
        throw(
            DimensionMismatch(
                "X_vertex has " *
                "$(size(X_vertex, 2)) features, " *
                "but the layer expects " *
                "$(layer.input_dim).",
            ),
        )

    size(
        incidence_matrix,
        1,
    ) == size(
        X_vertex,
        1,
    ) ||
        throw(
            DimensionMismatch(
                "The incidence matrix has " *
                "$(size(incidence_matrix, 1)) vertex rows, " *
                "but X_vertex contains " *
                "$(size(X_vertex, 1)) vertices.",
            ),
        )

    return nothing
end


"""
    _hypergraph_attention_forward(
        layer,
        X_vertex,
        incidence_matrix,
        ps,
    )

Perform one complete attention-based undirected hypergraph message-passing
step.

The computation proceeds as follows:

1. Transform input vertex features.
2. Compute vertex-to-hyperedge attention.
3. Aggregate vertex representations into hyperedge representations.
4. Compute hyperedge-to-vertex attention.
5. Aggregate hyperedge representations back into vertices.
6. Apply the output transformation, optional bias, and activation.

# Returns

A named tuple containing the updated vertices, hyperedge representations, and
both attention matrices.
"""
function _hypergraph_attention_forward(
    layer::HypergraphAttentionLayer,
    X_vertex::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
    ps,
)
    _validate_hypergraph_attention_inputs(
        layer,
        X_vertex,
        incidence_matrix,
    )

    transformed_vertices =
        X_vertex *
        ps.W_vertex

    hyperedge_messages,
    vertex_to_hyperedge_attention =
        _hypergraph_vertex_to_hyperedge_attention(
            transformed_vertices,
            incidence_matrix,
            ps.a_vertex,
            layer.negative_slope,
        )

    propagated_vertices,
    hyperedge_to_vertex_attention =
        _hypergraph_hyperedge_to_vertex_attention(
            hyperedge_messages,
            incidence_matrix,
            ps.a_hyperedge,
            layer.negative_slope,
        )

    linear_output =
        propagated_vertices *
        ps.W_output

    if layer.use_bias
        linear_output =
            linear_output .+
            ps.b
    end

    updated_vertices =
        layer.activation.(
            linear_output,
        )

    return (
        updated_vertices =
            updated_vertices,
        hyperedge_messages =
            hyperedge_messages,
        vertex_to_hyperedge_attention =
            vertex_to_hyperedge_attention,
        hyperedge_to_vertex_attention =
            hyperedge_to_vertex_attention,
    )
end


"""
    (layer::HypergraphAttentionLayer)(
        input,
        ps,
        st,
    )

Apply one attention-based message-passing step to an undirected hypergraph.

The input must contain exactly:

    (X_vertex, incidence_matrix)

The returned output contains the updated vertex representations, intermediate
hyperedge representations, and both sets of learned attention coefficients.

The Lux state is returned unchanged because the layer is stateless.
"""
function (layer::HypergraphAttentionLayer)(
    input::Tuple,
    ps,
    st,
)
    length(input) == 2 ||
        throw(
            ArgumentError(
                "Expected `(X_vertex, incidence_matrix)`.",
            ),
        )

    X_vertex,
    incidence_matrix =
        input

    output =
        _hypergraph_attention_forward(
            layer,
            X_vertex,
            incidence_matrix,
            ps,
        )

    return output, st
end