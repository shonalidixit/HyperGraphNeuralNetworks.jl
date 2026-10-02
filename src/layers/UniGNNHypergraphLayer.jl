using Lux
using Random

"""
    UniGNNHypergraphLayer(
        input_dim,
        output_dim;
        activation = tanh,
        use_bias = true,
        include_self = true,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

Undirected hypergraph message-passing layer based on a two-stage
vertex-to-hyperedge-to-vertex propagation scheme.

The layer first aggregates vertex representations into hyperedge
representations. The hyperedge representations are then propagated back to
the vertices.

For an incidence matrix `H`, every nonzero value is interpreted as membership.
The magnitude or sign of a nonzero incidence value does not affect the
aggregation.

The aggregation is degree-normalized:

1. Each hyperedge receives the mean representation of its incident vertices.
2. Each vertex receives the mean representation of its incident hyperedges.

If `include_self` is enabled, the propagated vertex representation is
concatenated with the original vertex representation before the learned
transformation.

Empty hyperedges and isolated vertices are handled safely and produce
zero-valued propagated representations.

# Arguments

- `input_dim::Int`: Number of input features for each vertex.
- `output_dim::Int`: Number of output features for each vertex.

# Keyword Arguments

- `activation = tanh`: Activation applied to the final vertex representations.
- `use_bias::Bool = true`: Whether to include a trainable bias.
- `include_self::Bool = true`: Whether to concatenate the original vertex
  features with the propagated vertex features.
- `init_weight = Lux.glorot_uniform`: Initializer used for the weight matrix.
- `init_bias = Lux.zeros32`: Initializer used for the bias.

# Input

The layer expects a tuple

    (X_vertex, incidence_matrix)

where:

- `X_vertex` has shape `n_vertices x input_dim`;
- `incidence_matrix` has shape `n_vertices x n_hyperedges`.

# Output

The layer returns a named tuple containing:

- `updated_vertices`: matrix of shape `n_vertices x output_dim`;
- `hyperedge_messages`: matrix of shape `n_hyperedges x input_dim`.

The Lux state is returned unchanged because the layer is stateless.

# Example

    rng = Random.default_rng()

    layer = UniGNNHypergraphLayer(
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
struct UniGNNHypergraphLayer{
    F,
    IW,
    IB,
} <: Lux.AbstractLuxLayer
    input_dim::Int
    output_dim::Int
    activation::F
    use_bias::Bool
    include_self::Bool
    init_weight::IW
    init_bias::IB
end


"""
    UniGNNHypergraphLayer(
        input_dim,
        output_dim;
        kwargs...,
    )

Construct a `UniGNNHypergraphLayer`.

Both `input_dim` and `output_dim` must be positive.
"""
function UniGNNHypergraphLayer(
    input_dim::Int,
    output_dim::Int;
    activation = tanh,
    use_bias::Bool = true,
    include_self::Bool = true,
    init_weight = Lux.glorot_uniform,
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

    return UniGNNHypergraphLayer(
        input_dim,
        output_dim,
        activation,
        use_bias,
        include_self,
        init_weight,
        init_bias,
    )
end


"""
    _unignn_row_major_weight(
        initializer,
        rng,
        input_dim,
        output_dim,
    )

Initialize a row-major weight matrix.

The returned matrix has shape

    input_dim x output_dim

Lux initializers conventionally construct matrices in output-by-input
orientation. The initialized matrix is therefore transposed before use.
"""
function _unignn_row_major_weight(
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
    _unignn_row_major_bias(
        initializer,
        rng,
        output_dim,
    )

Initialize a row-major bias with shape

    1 x output_dim
"""
function _unignn_row_major_bias(
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
        layer::UniGNNHypergraphLayer,
    )

Initialize the trainable parameters of a `UniGNNHypergraphLayer`.

When `include_self` is enabled, the propagated representation is concatenated
with the original vertex representation. The learned transformation therefore
receives `2 * input_dim` features.

When `include_self` is disabled, the transformation receives only
`input_dim` propagated features.
"""
function Lux.initialparameters(
    rng::AbstractRNG,
    layer::UniGNNHypergraphLayer,
)
    update_input_dim =
        if layer.include_self
            2 * layer.input_dim
        else
            layer.input_dim
        end

    W =
        _unignn_row_major_weight(
            layer.init_weight,
            rng,
            update_input_dim,
            layer.output_dim,
        )

    if layer.use_bias
        b =
            _unignn_row_major_bias(
                layer.init_bias,
                rng,
                layer.output_dim,
            )

        return (
            W = W,
            b = b,
        )
    end

    return (
        W = W,
    )
end


"""
    Lux.parameterlength(
        layer::UniGNNHypergraphLayer,
    )

Return the number of trainable scalar parameters in the layer.
"""
function Lux.parameterlength(
    layer::UniGNNHypergraphLayer,
)
    update_input_dim =
        if layer.include_self
            2 * layer.input_dim
        else
            layer.input_dim
        end

    weight_parameters =
        update_input_dim *
        layer.output_dim

    bias_parameters =
        if layer.use_bias
            layer.output_dim
        else
            0
        end

    return weight_parameters +
           bias_parameters
end


"""
    Lux.initialstates(
        rng,
        layer::UniGNNHypergraphLayer,
    )

Return the Lux state associated with the layer.

`UniGNNHypergraphLayer` is stateless.
"""
function Lux.initialstates(
    ::AbstractRNG,
    ::UniGNNHypergraphLayer,
)
    return NamedTuple()
end


"""
    Lux.statelength(
        layer::UniGNNHypergraphLayer,
    )

Return the number of state variables maintained by the layer.

The layer is stateless, so the result is always zero.
"""
function Lux.statelength(
    ::UniGNNHypergraphLayer,
)
    return 0
end


"""
    _unignn_membership_matrix(
        incidence_matrix,
    )

Convert an incidence matrix into binary hypergraph membership.

Every nonzero incidence value is interpreted as membership. Zero represents
the absence of a vertex-hyperedge relationship.
"""
function _unignn_membership_matrix(
    incidence_matrix::AbstractMatrix,
)
    return convert.(
        eltype(incidence_matrix),
        .!iszero.(
            incidence_matrix,
        ),
    )
end


"""
    _unignn_safe_inverse(
        values,
    )

Compute element-wise reciprocals while mapping zero values to zero.

This function prevents division by zero when degree normalization encounters
isolated vertices or empty hyperedges.
"""
function _unignn_safe_inverse(
    values::AbstractArray,
)
    result =
        similar(
            values,
        )

    for index in eachindex(values)
        if iszero(
            values[index],
        )
            result[index] =
                zero(
                    eltype(values),
                )
        else
            result[index] =
                one(
                    eltype(values),
                ) /
                values[index]
        end
    end

    return result
end


"""
    _unignn_vertex_to_hyperedge(
        X_vertex,
        incidence_matrix,
    )

Aggregate vertex representations into hyperedge representations.

For each non-empty hyperedge, the output is the mean of the representations
of all incident vertices.

Empty hyperedges receive an all-zero representation.

# Arguments

- `X_vertex`: Vertex-feature matrix with shape
  `n_vertices x feature_dim`.
- `incidence_matrix`: Incidence matrix with shape
  `n_vertices x n_hyperedges`.

# Returns

A matrix with shape

    n_hyperedges x feature_dim
"""
function _unignn_vertex_to_hyperedge(
    X_vertex::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
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
                "The incidence matrix must contain one row per vertex.",
            ),
        )

    membership =
        _unignn_membership_matrix(
            incidence_matrix,
        )

    hyperedge_degrees =
        vec(
            sum(
                membership;
                dims = 1,
            ),
        )

    inverse_hyperedge_degrees =
        _unignn_safe_inverse(
            hyperedge_degrees,
        )

    normalized_membership =
        permutedims(
            membership,
        ) .*
        reshape(
            inverse_hyperedge_degrees,
            :,
            1,
        )

    return normalized_membership *
           X_vertex
end


"""
    _unignn_hyperedge_to_vertex(
        hyperedge_messages,
        incidence_matrix,
    )

Propagate hyperedge representations back to vertices.

For each non-isolated vertex, the output is the mean representation of its
incident hyperedges.

Isolated vertices receive an all-zero propagated representation.

# Arguments

- `hyperedge_messages`: Hyperedge-feature matrix with shape
  `n_hyperedges x feature_dim`.
- `incidence_matrix`: Incidence matrix with shape
  `n_vertices x n_hyperedges`.

# Returns

A matrix with shape

    n_vertices x feature_dim
"""
function _unignn_hyperedge_to_vertex(
    hyperedge_messages::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
)
    size(
        hyperedge_messages,
        1,
    ) == size(
        incidence_matrix,
        2,
    ) ||
        throw(
            DimensionMismatch(
                "The number of hyperedge messages must match " *
                "the number of incidence-matrix columns.",
            ),
        )

    membership =
        _unignn_membership_matrix(
            incidence_matrix,
        )

    vertex_degrees =
        vec(
            sum(
                membership;
                dims = 2,
            ),
        )

    inverse_vertex_degrees =
        _unignn_safe_inverse(
            vertex_degrees,
        )

    normalized_membership =
        membership .*
        reshape(
            inverse_vertex_degrees,
            :,
            1,
        )

    return normalized_membership *
           hyperedge_messages
end


"""
    _validate_unignn_inputs(
        layer,
        X_vertex,
        incidence_matrix,
    )

Validate the dimensions of the vertex-feature matrix and incidence matrix.

The function checks that:

- the number of columns in `X_vertex` matches `layer.input_dim`;
- the number of rows in `incidence_matrix` matches the number of vertices.

A `DimensionMismatch` is thrown before message passing if either condition
is violated.
"""
function _validate_unignn_inputs(
    layer::UniGNNHypergraphLayer,
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
    _unignn_forward(
        layer,
        X_vertex,
        incidence_matrix,
        ps,
    )

Perform one complete UniGNN message-passing step.

The computation proceeds as follows:

1. Aggregate vertex features into hyperedge representations.
2. Propagate hyperedge representations back to vertices.
3. Optionally concatenate the propagated representation with the original
   vertex features.
4. Apply the learned linear transformation.
5. Apply the optional bias.
6. Apply the configured activation.

# Returns

A named tuple containing:

- `updated_vertices`;
- `hyperedge_messages`.
"""
function _unignn_forward(
    layer::UniGNNHypergraphLayer,
    X_vertex::AbstractMatrix,
    incidence_matrix::AbstractMatrix,
    ps,
)
    _validate_unignn_inputs(
        layer,
        X_vertex,
        incidence_matrix,
    )

    hyperedge_messages =
        _unignn_vertex_to_hyperedge(
            X_vertex,
            incidence_matrix,
        )

    propagated_vertices =
        _unignn_hyperedge_to_vertex(
            hyperedge_messages,
            incidence_matrix,
        )

    update_input =
        if layer.include_self
            hcat(
                X_vertex,
                propagated_vertices,
            )
        else
            propagated_vertices
        end

    linear_output =
        update_input *
        ps.W

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
    )
end


"""
    (layer::UniGNNHypergraphLayer)(
        input,
        ps,
        st,
    )

Apply one UniGNN message-passing step to an undirected hypergraph.

The input must contain exactly

    (X_vertex, incidence_matrix)

where:

- `X_vertex` is an `n_vertices x input_dim` feature matrix;
- `incidence_matrix` is an `n_vertices x n_hyperedges` incidence matrix.

The returned output contains the updated vertex representations and the
intermediate hyperedge representations.

The Lux state is returned unchanged because the layer is stateless.

# Throws

- `ArgumentError` if the input tuple does not contain exactly two elements.
- `DimensionMismatch` if the vertex feature dimension does not match
  `input_dim`.
- `DimensionMismatch` if the incidence matrix does not contain one row for
  every vertex.
"""
function (layer::UniGNNHypergraphLayer)(
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
        _unignn_forward(
            layer,
            X_vertex,
            incidence_matrix,
            ps,
        )

    return output, st
end