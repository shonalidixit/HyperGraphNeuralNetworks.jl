using Lux
using Random

"""
    SpatialHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        activation = tanh,
        normalize = true,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

Spatial message-passing layer for undirected hypergraphs.

The layer performs two-stage message passing:

1. Vertices send messages to incident hyperedges.
2. Hyperedges are updated from the aggregated vertex messages and,
   optionally, existing hyperedge features.
3. Updated hyperedges send messages back to incident vertices.
4. Vertices are updated using their original features together with
   the aggregated hyperedge messages.

The incidence matrix is interpreted as a membership matrix: every
nonzero value means that a vertex belongs to a hyperedge.

When `normalize = true`, aggregation uses mean aggregation. When
`normalize = false`, aggregation uses sums.

# Arguments

- `vertex_in_dim::Int`: Number of input features per vertex.
- `hyperedge_in_dim::Int`: Number of input features per hyperedge.
  Use `0` when no hyperedge features are supplied.
- `hidden_dim::Int`: Number of output features for both vertices
  and hyperedges.

# Keyword Arguments

- `activation = tanh`: Activation function applied after learned updates.
- `normalize::Bool = true`: Whether incidence-based aggregation is normalized.
- `init_weight = Lux.glorot_uniform`: Weight initializer.
- `init_bias = Lux.zeros32`: Bias initializer.

# Inputs

If `hyperedge_in_dim == 0`, the layer expects:

    (X_vertex, H)

If `hyperedge_in_dim > 0`, the layer expects:

    (X_vertex, X_hyperedge, H)

where:

- `X_vertex` has shape `n_vertices × vertex_in_dim`.
- `X_hyperedge` has shape `n_hyperedges × hyperedge_in_dim`.
- `H` has shape `n_vertices × n_hyperedges`.

# Output

The layer returns a named tuple containing:

- `updated_vertices`
- `updated_hyperedges`

The layer is stateless, so the Lux state is returned unchanged.
"""
struct SpatialHypergraphLayer{
    F,
    IW,
    IB,
} <: Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hyperedge_in_dim::Int
    hidden_dim::Int
    activation::F
    normalize::Bool
    init_weight::IW
    init_bias::IB
end


"""
    SpatialHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        kwargs...,
    )

Construct a `SpatialHypergraphLayer`.

`vertex_in_dim` and `hidden_dim` must be positive.
`hyperedge_in_dim` must be non-negative.
"""
function SpatialHypergraphLayer(
    vertex_in_dim::Int,
    hyperedge_in_dim::Int,
    hidden_dim::Int;
    activation = tanh,
    normalize::Bool = true,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    vertex_in_dim > 0 ||
        throw(
            ArgumentError(
                "`vertex_in_dim` must be positive.",
            ),
        )

    hyperedge_in_dim >= 0 ||
        throw(
            ArgumentError(
                "`hyperedge_in_dim` must be non-negative.",
            ),
        )

    hidden_dim > 0 ||
        throw(
            ArgumentError(
                "`hidden_dim` must be positive.",
            ),
        )

    return SpatialHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim,
        activation,
        normalize,
        init_weight,
        init_bias,
    )
end


"""
    _spatial_row_major_weight(
        initializer,
        rng,
        input_dim,
        output_dim,
    )

Initialize a weight matrix with shape
`input_dim × output_dim`.

Lux initializers conventionally produce output-by-input matrices,
so the result is transposed for the row-major feature convention used
by this layer.
"""
function _spatial_row_major_weight(
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
    _spatial_row_major_bias(
        initializer,
        rng,
        output_dim,
    )

Initialize a bias matrix with shape `1 × output_dim`.
"""
function _spatial_row_major_bias(
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
        layer::SpatialHypergraphLayer,
    )

Initialize the trainable parameters.

The layer contains three learned transformations:

1. Vertex projection:
   `vertex_in_dim -> hidden_dim`

2. Hyperedge update:
   `(hidden_dim + hyperedge_in_dim) -> hidden_dim`

3. Vertex update:
   `(vertex_in_dim + hidden_dim) -> hidden_dim`
"""
function Lux.initialparameters(
    rng::AbstractRNG,
    layer::SpatialHypergraphLayer,
)
    W_vertex =
        _spatial_row_major_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        )

    b_vertex =
        _spatial_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        )

    hyperedge_update_dim =
        layer.hidden_dim +
        layer.hyperedge_in_dim

    W_hyperedge =
        _spatial_row_major_weight(
            layer.init_weight,
            rng,
            hyperedge_update_dim,
            layer.hidden_dim,
        )

    b_hyperedge =
        _spatial_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        )

    vertex_update_dim =
        layer.vertex_in_dim +
        layer.hidden_dim

    W_vertex_update =
        _spatial_row_major_weight(
            layer.init_weight,
            rng,
            vertex_update_dim,
            layer.hidden_dim,
        )

    b_vertex_update =
        _spatial_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        )

    return (
        W_vertex = W_vertex,
        b_vertex = b_vertex,
        W_hyperedge = W_hyperedge,
        b_hyperedge = b_hyperedge,
        W_vertex_update = W_vertex_update,
        b_vertex_update = b_vertex_update,
    )
end


"""
    Lux.parameterlength(
        layer::SpatialHypergraphLayer,
    )

Return the total number of trainable scalar parameters.
"""
function Lux.parameterlength(
    layer::SpatialHypergraphLayer,
)
    vertex_projection =
        layer.vertex_in_dim *
        layer.hidden_dim +
        layer.hidden_dim

    hyperedge_update =
        (
            layer.hidden_dim +
            layer.hyperedge_in_dim
        ) *
        layer.hidden_dim +
        layer.hidden_dim

    vertex_update =
        (
            layer.vertex_in_dim +
            layer.hidden_dim
        ) *
        layer.hidden_dim +
        layer.hidden_dim

    return vertex_projection +
           hyperedge_update +
           vertex_update
end


"""
    Lux.initialstates(
        rng,
        layer::SpatialHypergraphLayer,
    )

Return the layer state.

`SpatialHypergraphLayer` is stateless.
"""
function Lux.initialstates(
    ::AbstractRNG,
    ::SpatialHypergraphLayer,
)
    return NamedTuple()
end


"""
    Lux.statelength(
        layer::SpatialHypergraphLayer,
    )

Return the number of state variables.

The layer is stateless, so this is always zero.
"""
function Lux.statelength(
    ::SpatialHypergraphLayer,
)
    return 0
end


"""
    _spatial_membership_matrix(H)

Convert an incidence matrix into a binary membership matrix.

Every nonzero entry is interpreted as membership.
"""
function _spatial_membership_matrix(
    H::AbstractMatrix,
)
    return convert.(
        eltype(H),
        .!iszero.(H),
    )
end


"""
    _spatial_safe_inverse(values)

Compute element-wise reciprocals while mapping zero values to zero.

This prevents division-by-zero for isolated vertices and empty
hyperedges.
"""
function _spatial_safe_inverse(
    values::AbstractArray,
)
    result =
        similar(values)

    for index in eachindex(values)
        if iszero(values[index])
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
    _spatial_vertex_to_hyperedge(
        H,
        normalize,
    )

Construct the vertex-to-hyperedge propagation matrix.

The returned matrix has shape:

    n_hyperedges × n_vertices

When `normalize == false`, this is simply the transpose of the binary
membership matrix.

When `normalize == true`, each hyperedge row is divided by the number
of incident vertices, producing mean aggregation.

Empty hyperedges remain all zero.
"""
function _spatial_vertex_to_hyperedge(
    H::AbstractMatrix,
    normalize::Bool,
)
    membership =
        _spatial_membership_matrix(
            H,
        )

    propagation =
        permutedims(
            membership,
        )

    if !normalize
        return propagation
    end

    hyperedge_degrees =
        vec(
            sum(
                membership;
                dims = 1,
            ),
        )

    inverse_degrees =
        _spatial_safe_inverse(
            hyperedge_degrees,
        )

    return propagation .*
           reshape(
               inverse_degrees,
               :,
               1,
           )
end


"""
    _spatial_hyperedge_to_vertex(
        H,
        normalize,
    )

Construct the hyperedge-to-vertex propagation matrix.

The returned matrix has shape:

    n_vertices × n_hyperedges

When `normalize == false`, this is the binary membership matrix.

When `normalize == true`, each vertex row is divided by the number
of incident hyperedges, producing mean aggregation.

Isolated vertices remain all zero.
"""
function _spatial_hyperedge_to_vertex(
    H::AbstractMatrix,
    normalize::Bool,
)
    membership =
        _spatial_membership_matrix(
            H,
        )

    if !normalize
        return membership
    end

    vertex_degrees =
        vec(
            sum(
                membership;
                dims = 2,
            ),
        )

    inverse_degrees =
        _spatial_safe_inverse(
            vertex_degrees,
        )

    return membership .*
           reshape(
               inverse_degrees,
               :,
               1,
           )
end


"""
    _validate_spatial_inputs(
        layer,
        X_vertex,
        X_hyperedge,
        H,
    )

Validate dimensions for the spatial message-passing computation.
"""
function _validate_spatial_inputs(
    layer::SpatialHypergraphLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge,
    H::AbstractMatrix,
)
    size(
        X_vertex,
        2,
    ) == layer.vertex_in_dim ||
        throw(
            DimensionMismatch(
                "X_vertex has $(size(X_vertex, 2)) features, " *
                "but the layer expects $(layer.vertex_in_dim).",
            ),
        )

    size(
        H,
        1,
    ) == size(
        X_vertex,
        1,
    ) ||
        throw(
            DimensionMismatch(
                "The incidence matrix has $(size(H, 1)) vertex rows, " *
                "but X_vertex contains $(size(X_vertex, 1)) vertices.",
            ),
        )

    if layer.hyperedge_in_dim > 0
        X_hyperedge === nothing &&
            throw(
                ArgumentError(
                    "Hyperedge features are required when " *
                    "`hyperedge_in_dim > 0`.",
                ),
            )

        size(
            X_hyperedge,
            1,
        ) == size(
            H,
            2,
        ) ||
            throw(
                DimensionMismatch(
                    "X_hyperedge contains $(size(X_hyperedge, 1)) " *
                    "hyperedges, but the incidence matrix contains " *
                    "$(size(H, 2)).",
                ),
            )

        size(
            X_hyperedge,
            2,
        ) == layer.hyperedge_in_dim ||
            throw(
                DimensionMismatch(
                    "X_hyperedge has $(size(X_hyperedge, 2)) features, " *
                    "but the layer expects $(layer.hyperedge_in_dim).",
                ),
            )
    else
        X_hyperedge === nothing ||
            throw(
                ArgumentError(
                    "Hyperedge features must not be supplied when " *
                    "`hyperedge_in_dim == 0`.",
                ),
            )
    end

    return nothing
end


"""
    _spatial_forward(
        layer,
        X_vertex,
        X_hyperedge,
        H,
        ps,
    )

Perform one spatial hypergraph message-passing step.

The computation is:

1. Project vertex features into the hidden space.
2. Aggregate projected vertices into hyperedges.
3. Concatenate existing hyperedge features when provided.
4. Update hyperedges.
5. Aggregate updated hyperedges back to vertices.
6. Concatenate the original vertex features with received messages.
7. Update vertices.

# Returns

A named tuple containing:

- `updated_vertices`
- `updated_hyperedges`
"""
function _spatial_forward(
    layer::SpatialHypergraphLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge,
    H::AbstractMatrix,
    ps,
)
    _validate_spatial_inputs(
        layer,
        X_vertex,
        X_hyperedge,
        H,
    )

    vertex_hidden =
        layer.activation.(
            X_vertex *
            ps.W_vertex .+
            ps.b_vertex
        )

    vertex_to_hyperedge =
        _spatial_vertex_to_hyperedge(
            H,
            layer.normalize,
        )

    aggregated_vertices =
        vertex_to_hyperedge *
        vertex_hidden

    hyperedge_input =
        if layer.hyperedge_in_dim > 0
            hcat(
                aggregated_vertices,
                X_hyperedge,
            )
        else
            aggregated_vertices
        end

    updated_hyperedges =
        layer.activation.(
            hyperedge_input *
            ps.W_hyperedge .+
            ps.b_hyperedge
        )

    hyperedge_to_vertex =
        _spatial_hyperedge_to_vertex(
            H,
            layer.normalize,
        )

    aggregated_hyperedges =
        hyperedge_to_vertex *
        updated_hyperedges

    vertex_update_input =
        hcat(
            X_vertex,
            aggregated_hyperedges,
        )

    updated_vertices =
        layer.activation.(
            vertex_update_input *
            ps.W_vertex_update .+
            ps.b_vertex_update
        )

    return (
        updated_vertices =
            updated_vertices,
        updated_hyperedges =
            updated_hyperedges,
    )
end


"""
    (layer::SpatialHypergraphLayer)(
        input,
        ps,
        st,
    )

Apply the spatial hypergraph layer.

When `hyperedge_in_dim == 0`, `input` must be:

    (X_vertex, H)

When `hyperedge_in_dim > 0`, `input` must be:

    (X_vertex, X_hyperedge, H)

The Lux state is returned unchanged.
"""
function (layer::SpatialHypergraphLayer)(
    input::Tuple,
    ps,
    st,
)
    if layer.hyperedge_in_dim == 0
        length(input) == 2 ||
            throw(
                ArgumentError(
                    "Expected `(X_vertex, H)` when " *
                    "`hyperedge_in_dim == 0`.",
                ),
            )

        X_vertex,
        H =
            input

        output =
            _spatial_forward(
                layer,
                X_vertex,
                nothing,
                H,
                ps,
            )

        return output, st
    end

    length(input) == 3 ||
        throw(
            ArgumentError(
                "Expected `(X_vertex, X_hyperedge, H)` when " *
                "`hyperedge_in_dim > 0`.",
            ),
        )

    X_vertex,
    X_hyperedge,
    H =
        input

    output =
        _spatial_forward(
            layer,
            X_vertex,
            X_hyperedge,
            H,
            ps,
        )

    return output, st
end